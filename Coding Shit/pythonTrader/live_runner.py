from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import main as backtest


LOGGER = logging.getLogger("live_runner")
UTC = timezone.utc


@dataclass
class HistoryEntry:
    closes: pd.Series
    fetched_at: datetime


def configure_logging(log_path: Optional[str], verbose: bool) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def load_genome(path: str) -> np.ndarray:
    LOGGER.info("Loading genome from %s", path)
    genome = backtest.load_best_genome(path)
    LOGGER.info("Genome loaded (%d weights).", genome.size)
    return genome


def fetch_daily_history(symbol: str, lookback_days: int) -> pd.Series:
    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days + 60)
    try:
        closes = backtest.download_history(symbol, start, end)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to pull daily history for %s: %s", symbol, exc)
        return pd.Series(dtype=float)
    return closes


_TICKER_CACHE: Dict[str, yf.Ticker] = {}


def _get_ticker(symbol: str) -> yf.Ticker:
    cached = _TICKER_CACHE.get(symbol)
    if cached is None:
        cached = yf.Ticker(symbol)
        _TICKER_CACHE[symbol] = cached
    return cached


def fetch_live_price(symbol: str, price_interval: str) -> Optional[float]:
    ticker = _get_ticker(symbol)
    try:
        data = ticker.history(period="1d", interval=price_interval)
        if not data.empty and "Close" in data.columns:
            price = data["Close"].iloc[-1]
            if np.isfinite(price):
                return float(price)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Minute history failed for %s: %s", symbol, exc)
    try:
        info = getattr(ticker, "fast_info", None)
        if info:
            for key in ("lastPrice", "last_price"):
                price = info.get(key)
                if price is not None and np.isfinite(price):
                    return float(price)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("fast_info lookup failed for %s: %s", symbol, exc)
    return None


def refresh_news(now_date: date, cache: Dict[str, Dict[str, object]]) -> Dict[str, Dict[date, List[str]]]:
    if not backtest.FINNHUB_API_KEY:
        return {ticker: {} for ticker in backtest.TICKERS}
    updated: Dict[str, Dict[date, List[str]]] = {}
    window_start = now_date - timedelta(days=backtest.NEWS_LOOKBACK_DAYS)
    for ticker in backtest.TICKERS:
        entry = cache.get(ticker)
        if not entry or entry.get("fetched") != now_date:
            news_map = backtest.load_company_news(ticker, window_start, now_date)
            cache[ticker] = {"fetched": now_date, "data": news_map}
        updated[ticker] = cache[ticker]["data"]  # type: ignore[index]
    return updated


def compute_live_frames(
    ticker: str,
    closes: pd.Series,
    live_price: Optional[float],
) -> pd.DataFrame:
    if closes.empty:
        return pd.DataFrame()
    series = closes.copy()
    if live_price is not None:
        series.iloc[-1] = live_price
    frame = backtest.compute_feature_frame(series)
    return frame.tail(1)


def evaluate_signals(
    genome: np.ndarray,
    frames: Dict[str, pd.DataFrame],
    news_map: Dict[str, Dict[date, List[str]]],
    capital: float,
) -> List[Dict[str, object]]:
    if genome.size != len(backtest.TICKERS) * backtest.FEATURES_PER_TICKER:
        raise ValueError("Genome length mismatch; retrain the model.")

    short_w = genome[0::backtest.FEATURES_PER_TICKER]
    long_w = genome[1::backtest.FEATURES_PER_TICKER]
    vol5_w = genome[2::backtest.FEATURES_PER_TICKER]
    volspike_w = genome[3::backtest.FEATURES_PER_TICKER]

    raw_signals: Dict[str, float] = {}
    meta: Dict[str, Dict[str, object]] = {}
    now_ts = datetime.now(UTC)

    for idx, ticker in enumerate(backtest.TICKERS):
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[-1]
        if row.isnull().any():
            continue

        signal = (
            short_w[idx] * row.get("short_diff", 0.0)
            + long_w[idx] * row.get("long_diff", 0.0)
            + vol5_w[idx] * row.get("vol5_z", 0.0)
            + volspike_w[idx] * row.get("vol_spike_z", 0.0)
            + backtest.SIGNAL_BIAS
        )

        headline = backtest.select_headline(news_map.get(ticker, {}), row.name.date())
        senti = backtest.sentiment_score(headline)
        signal += backtest.SENTIMENT_WEIGHT * senti
        signal = float(np.clip(signal, -backtest.WEIGHT_CLIP, backtest.WEIGHT_CLIP))

        if abs(signal) < backtest.ENTRY_THRESHOLD:
            continue

        price = float(row.get("close", np.nan))
        raw_signals[ticker] = signal
        meta[ticker] = {
            "price": price,
            "headline": headline,
            "sentiment": senti,
            "timestamp": row.name,
        }

    weights = backtest.normalize_weights(raw_signals)
    results: List[Dict[str, object]] = []

    for ticker, weight in weights.items():
        info = meta[ticker]
        price = info["price"]
        notional = capital * abs(weight) if capital > 0 else float("nan")
        shares = notional / price if capital > 0 and price > 0 else float("nan")

        results.append(
            {
                "timestamp": now_ts.isoformat(),
                "data_timestamp": info["timestamp"].isoformat(),
                "ticker": ticker,
                "direction": "LONG" if weight > 0 else "SHORT",
                "weight": float(weight),
                "signal_strength": float(raw_signals[ticker]),
                "price": price,
                "capital_allocated": notional,
                "target_shares": shares,
                "sentiment": info["sentiment"],
                "headline": info["headline"],
            }
        )

    results.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return results


def render_table(records: Iterable[Dict[str, object]]) -> str:
    df = pd.DataFrame(records)
    if df.empty:
        return "No signals."
    display_cols = [
        "ticker",
        "direction",
        "weight",
        "signal_strength",
        "price",
        "capital_allocated",
        "target_shares",
        "sentiment",
    ]
    missing = [col for col in display_cols if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    df = df[display_cols]
    df["weight"] = df["weight"].map(lambda val: f"{val:+.3f}")
    df["signal_strength"] = df["signal_strength"].map(lambda val: f"{val:+.3f}")
    df["price"] = df["price"].map(lambda val: f"{val:,.2f}")
    df["capital_allocated"] = df["capital_allocated"].map(
        lambda val: "n/a" if not np.isfinite(val) else f"{val:,.2f}"
    )
    df["target_shares"] = df["target_shares"].map(
        lambda val: "n/a" if not np.isfinite(val) else f"{val:,.2f}"
    )
    df["sentiment"] = df["sentiment"].map(lambda val: f"{val:+.2f}")
    return df.to_string(index=False)


def append_to_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def run_once(
    genome: np.ndarray,
    history: Dict[str, HistoryEntry],
    news_cache: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    loop_start = datetime.now(UTC)
    LOGGER.info("Polling live data at %s", loop_start.isoformat())

    news_map = refresh_news(loop_start.date(), news_cache)
    frames: Dict[str, pd.DataFrame] = {}

    for ticker in backtest.TICKERS:
        entry = history.get(ticker)
        needs_refresh = True
        if entry:
            age = (loop_start - entry.fetched_at).total_seconds() / 60.0
            needs_refresh = age >= args.refresh_minutes
        if needs_refresh:
            closes = fetch_daily_history(ticker, args.lookback_days)
            if closes.empty:
                LOGGER.warning("Skipping %s due to missing history.", ticker)
                continue
            entry = HistoryEntry(closes=closes, fetched_at=loop_start)
            history[ticker] = entry

        price = fetch_live_price(ticker, args.price_interval)
        if price is None:
            LOGGER.debug("No live price for %s; using last close.", ticker)
        frame = compute_live_frames(ticker, entry.closes, price)
        if frame.empty:
            LOGGER.warning("No feature row for %s.", ticker)
            continue
        frames[ticker] = frame

    records = evaluate_signals(genome, frames, news_map, args.capital)
    LOGGER.info("Generated %d signals.", len(records))
    if records:
        LOGGER.info("\n%s", render_table(records))
    else:
        LOGGER.info("No qualifying signals this cycle.")

    if args.output_csv:
        append_to_csv(Path(args.output_csv), records)

    return records


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live genome signal monitor.")
    parser.add_argument("--genome", type=str, default=backtest.BEST_GENOME_PATH, help="Path to genome npy file.")
    parser.add_argument("--interval", type=float, default=300.0, help="Polling interval in seconds.")
    parser.add_argument(
        "--refresh-minutes",
        type=float,
        default=180.0,
        help="Minutes between full history refreshes per ticker.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=backtest.LOOKBACK_DAYS,
        help="Historical window (days) for feature computation.",
    )
    parser.add_argument(
        "--price-interval",
        type=str,
        default="1m",
        help="Intraday interval passed to yfinance for live prices (e.g. 1m,2m,5m).",
    )
    parser.add_argument("--capital", type=float, default=100_000.0, help="Total deployable capital for sizing.")
    parser.add_argument("--output-csv", type=str, default="live_signals.csv", help="Append signals to this CSV file.")
    parser.add_argument("--log-file", type=str, default="live_runner.log", help="Optional log file path.")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file, args.verbose)

    genome = load_genome(args.genome)
    history: Dict[str, HistoryEntry] = {}
    news_cache: Dict[str, Dict[str, object]] = {}

    stop_requested = False

    def _signal_handler(signum, _frame):
        nonlocal stop_requested
        LOGGER.info("Received signal %s, preparing to shut down.", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while True:
            start_ts = time.time()
            run_once(genome, history, news_cache, args)
            if args.once or stop_requested:
                break
            elapsed = time.time() - start_ts
            sleep_for = max(1.0, args.interval - elapsed)
            LOGGER.debug("Sleeping for %.2f seconds.", sleep_for)
            time.sleep(sleep_for)
    finally:
        LOGGER.info("Live runner stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
