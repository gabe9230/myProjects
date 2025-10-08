from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
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

import matplotlib

_GUI_BACKEND = "Agg"
try:
    matplotlib.use("TkAgg")
    _GUI_BACKEND = "TkAgg"
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOGGER = logging.getLogger("live_runner")
UTC = timezone.utc
GUI_AVAILABLE = _GUI_BACKEND != "Agg"

STATE_VERSION = 1
DEFAULT_STATE_FILE = "live_state.json"

if os.name == "nt":
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
else:
    DETACHED_PROCESS = 0
    CREATE_NEW_PROCESS_GROUP = 0


def _normalize_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def default_state(capital: float) -> Dict[str, object]:
    return {
        "initial_capital": float(capital),
        "balance": float(capital),
        "positions": {},
        "last_pnl": 0.0,
        "equity_history": [],
        "last_update": None,
    }


def _serialize_state(state: Dict[str, object]) -> Dict[str, object]:
    positions = {}
    for ticker, data in state.get("positions", {}).items():
        if not isinstance(data, dict):
            continue
        try:
            positions[str(ticker)] = {
                "weight": float(data.get("weight", 0.0)),
                "price": float(data.get("price", 0.0)),
                "notional": float(data.get("notional", 0.0)),
            }
        except Exception:
            continue

    history_serialized = []
    for item in state.get("equity_history", []):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        ts, balance = item
        if not isinstance(ts, datetime):
            continue
        try:
            history_serialized.append(
                {"timestamp": ts.isoformat(), "balance": float(balance)}
            )
        except Exception:
            continue

    payload = {
        "version": STATE_VERSION,
        "initial_capital": float(state.get("initial_capital", 0.0)),
        "balance": float(state.get("balance", 0.0)),
        "last_pnl": float(state.get("last_pnl", 0.0)),
        "positions": positions,
        "equity_history": history_serialized,
        "last_update": state.get("last_update"),
    }
    return payload


def save_state(path: Path, state: Dict[str, object]) -> None:
    try:
        payload = _serialize_state(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)
        LOGGER.info("Saved session state to %s", path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to save session state to %s: %s", path, exc)


def load_state(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load session state from %s: %s", path, exc)
        return None

    state = default_state(float(payload.get("initial_capital", payload.get("balance", 0.0))))
    try:
        state["initial_capital"] = float(payload.get("initial_capital", state["initial_capital"]))
    except Exception:
        pass
    try:
        state["balance"] = float(payload.get("balance", state["balance"]))
    except Exception:
        pass
    try:
        state["last_pnl"] = float(payload.get("last_pnl", 0.0))
    except Exception:
        pass
    state["last_update"] = payload.get("last_update")

    positions = {}
    for ticker, data in payload.get("positions", {}).items():
        if not isinstance(data, dict):
            continue
        try:
            positions[str(ticker)] = {
                "weight": float(data.get("weight", 0.0)),
                "price": float(data.get("price", 0.0)),
                "notional": float(data.get("notional", 0.0)),
            }
        except Exception:
            continue
    state["positions"] = positions

    history = []
    for entry in payload.get("equity_history", []):
        timestamp = None
        balance = None
        if isinstance(entry, dict):
            timestamp = entry.get("timestamp")
            balance = entry.get("balance")
        elif isinstance(entry, list) and len(entry) >= 2:
            timestamp, balance = entry[0], entry[1]
        if not timestamp:
            continue
        try:
            dt = _normalize_timestamp(str(timestamp))
            bal = float(balance)
        except Exception:
            continue
        history.append((dt, bal))
    history.sort(key=lambda pair: pair[0])
    state["equity_history"] = history
    LOGGER.info("Loaded session state from %s (balance %.2f)", path, state["balance"])
    return state


def spawn_background_process(argv: List[str]) -> None:
    script_path = Path(__file__).resolve()
    args = [sys.executable, str(script_path)]
    args.extend(argv)
    args.append("--background-child")
    if "--no-gui" not in args:
        args.append("--no-gui")
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = os.setsid  # type: ignore[attr-defined]
    try:
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
            close_fds=True,
        )
        print("Live runner started in background.")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to launch background process: {exc}")

@dataclass
class HistoryEntry:
    closes: pd.Series
    fetched_at: datetime


class EquityDashboard:
    def __init__(self, title: str = "Live Portfolio Balance") -> None:
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI backend unavailable; cannot start live dashboard.")
        import tkinter as tk  # type: ignore[import]
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._tk = tk
        self._queue: "queue.Queue[tuple[List[datetime], List[float]]]" = queue.Queue()
        self._closed = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.ax.set_title(title)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Balance")
        self.ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        self.line, = self.ax.plot([], [], color="#1f77b4", linewidth=2)
        self.window_points = 200

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        self.root.after(250, self._poll_queue)

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def _poll_queue(self) -> None:
        if self._closed:
            return
        try:
            while True:
                timestamps, balances = self._queue.get_nowait()
                self._refresh_plot(timestamps, balances)
        except queue.Empty:
            pass
        self.root.after(250, self._poll_queue)

    def _refresh_plot(self, timestamps: List[datetime], balances: List[float]) -> None:
        if not timestamps or not balances:
            return
        if len(timestamps) > self.window_points:
            timestamps = timestamps[-self.window_points :]
            balances = balances[-self.window_points :]
        normalized = [
            ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts
            for ts in timestamps
        ]
        xs = mdates.date2num(normalized)
        self.line.set_data(xs, balances)
        if len(xs) == 1:
            span = 1 / (24 * 60)  # one minute window
            self.ax.set_xlim(xs[0] - span, xs[0] + span)
            value = balances[0]
            pad = max(abs(value) * 0.05, 1.0)
            self.ax.set_ylim(value - pad, value + pad)
        else:
            self.ax.set_xlim(xs[0], xs[-1])
            ymin = min(balances)
            ymax = max(balances)
            pad = max((ymax - ymin) * 0.1, max(abs(ymin), abs(ymax)) * 0.01, 0.5)
            if ymin == ymax:
                pad = max(abs(ymin) * 0.05, 0.5)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()

    def update(self, history: List[tuple]) -> None:
        if self._closed:
            return
        if not history:
            return
        timestamps, balances = zip(*history)
        ts_list = list(timestamps)
        bal_list = list(balances)
        if len(ts_list) > self.window_points:
            ts_list = ts_list[-self.window_points :]
            bal_list = bal_list[-self.window_points :]
        self._queue.put((ts_list, bal_list))

    def close(self) -> None:
        self._on_close()

    def process_events(self) -> bool:
        if self._closed:
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._closed = True
            return False
        return True

    @property
    def closed(self) -> bool:
        return self._closed


def configure_logging(log_path: Optional[str], verbose: bool) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


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
        data = ticker.history(period="1d", interval=price_interval, prepost=True)
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
    portfolio_balance: float,
    cycle_pnl: float,
    total_pnl: float,
    ticker_pnls: Dict[str, float],
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
    profile_balance = portfolio_balance if np.isfinite(portfolio_balance) else np.nan

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
                "profile_balance": profile_balance,
                "cycle_pnl": cycle_pnl,
                "total_pnl": total_pnl,
                "ticker_pnl": float(ticker_pnls.get(ticker, 0.0)),
                "sentiment": info["sentiment"],
                "headline": info["headline"],
            }
        )

    results.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return results


def compute_cycle_pnl(
    state: Dict[str, object],
    frames: Dict[str, pd.DataFrame],
) -> tuple[float, Dict[str, float]]:
    positions: Dict[str, Dict[str, float]] = state.get("positions", {})  # type: ignore[assignment]
    if not positions:
        return 0.0, {}

    current_prices: Dict[str, float] = {}
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            continue
        price = float(frame.iloc[-1].get("close", np.nan))
        if np.isfinite(price) and price > 0:
            current_prices[ticker] = price

    pnl = 0.0
    breakdown: Dict[str, float] = {}
    for ticker, pos in positions.items():
        price_now = current_prices.get(ticker)
        if price_now is None:
            continue
        entry_price = float(pos.get("price", np.nan))
        notional = float(pos.get("notional", 0.0))
        weight = float(pos.get("weight", 0.0))
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        if not np.isfinite(notional) or notional == 0.0:
            continue
        direction = 1.0 if weight >= 0 else -1.0
        ret = (price_now / entry_price - 1.0) * direction
        contribution = notional * ret
        pnl += contribution
        breakdown[ticker] = contribution
    return pnl, breakdown


def update_positions(state: Dict[str, object], records: List[Dict[str, object]]) -> None:
    balance = float(state.get("balance", 0.0))
    positions: Dict[str, Dict[str, float]] = {}
    for record in records:
        ticker = record.get("ticker")
        if not ticker:
            continue
        weight = float(record.get("weight", 0.0))
        price = float(record.get("price", np.nan))
        if not np.isfinite(weight) or not np.isfinite(price) or price <= 0:
            continue
        notional = balance * abs(weight)
        positions[str(ticker)] = {
            "weight": weight,
            "price": price,
            "notional": notional,
        }
    state["positions"] = positions


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
        "profile_balance",
        "cycle_pnl",
        "ticker_pnl",
        "total_pnl",
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
    df["profile_balance"] = df["profile_balance"].map(
        lambda val: "n/a" if not np.isfinite(val) else f"{val:,.2f}"
    )
    df["cycle_pnl"] = df["cycle_pnl"].map(
        lambda val: "n/a" if not np.isfinite(val) else f"{val:,.2f}"
    )
    df["ticker_pnl"] = df["ticker_pnl"].map(
        lambda val: "n/a" if not np.isfinite(val) else f"{val:,.2f}"
    )
    df["total_pnl"] = df["total_pnl"].map(
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


def append_equity_csv(path: Path, timestamp: datetime, balance: float, cycle_pnl: float, total_pnl: float) -> None:
    df = pd.DataFrame(
        [
            {
                "timestamp": timestamp.isoformat(),
                "balance": balance,
                "cycle_pnl": cycle_pnl,
                "total_pnl": total_pnl,
            }
        ]
    )
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def run_once(
    genome: np.ndarray,
    history: Dict[str, HistoryEntry],
    news_cache: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
    state: Dict[str, object],
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

    initial_capital = float(state.setdefault("initial_capital", args.capital))
    balance = float(state.setdefault("balance", initial_capital))
    state.setdefault("positions", {})

    cycle_pnl, ticker_pnls = compute_cycle_pnl(state, frames)
    if not np.isfinite(cycle_pnl):
        cycle_pnl = 0.0

    state["balance"] = balance + cycle_pnl
    state["last_pnl"] = cycle_pnl
    portfolio_balance = float(state["balance"])
    total_pnl = portfolio_balance - initial_capital

    capital_for_signals = max(portfolio_balance, 0.0)
    records = evaluate_signals(
        genome,
        frames,
        news_map,
        capital_for_signals,
        portfolio_balance,
        cycle_pnl,
        total_pnl,
        ticker_pnls,
    )
    LOGGER.info("Generated %d signals.", len(records))
    if records:
        LOGGER.info("\n%s", render_table(records))
    else:
        LOGGER.info("No qualifying signals this cycle.")

    LOGGER.info(
        "Cycle PnL: %s | Portfolio balance: %s | Total PnL: %s",
        f"{cycle_pnl:,.2f}",
        f"{portfolio_balance:,.2f}",
        f"{total_pnl:,.2f}",
    )

    update_positions(state, records)
    state["last_update"] = loop_start.isoformat()
    history: List[tuple] = state.setdefault("equity_history", [])  # type: ignore[assignment]
    history.append((loop_start, portfolio_balance))
    if len(history) > 10_000:
        del history[: len(history) - 10_000]
    dashboard = state.get("dashboard")
    if isinstance(dashboard, EquityDashboard):
        dashboard.update(history)

    if args.output_csv:
        append_to_csv(Path(args.output_csv), records)
    if args.equity_csv:
        append_equity_csv(Path(args.equity_csv), loop_start, portfolio_balance, cycle_pnl, total_pnl)

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
    parser.add_argument("--equity-csv", type=str, default="live_equity.csv", help="Append equity curve history here.")
    parser.add_argument(
        "--finnhub-key",
        type=str,
        default=None,
        help="Finnhub API key (overrides FINNHUB_API_KEY env).",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable live dashboard window.")
    parser.add_argument("--state-file", type=str, default=DEFAULT_STATE_FILE, help="Path to persist session state.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with a clean session state (ignores any saved state file).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Force resuming from the saved state file if available.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Launch the runner in a detached background process (implies --no-gui).",
    )
    parser.add_argument(
        "--background-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--log-file", type=str, default="live_runner.log", help="Optional log file path.")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.background_child:
        args.background = False
        args.no_gui = True

    if args.background and not args.background_child:
        source_args = list(argv) if argv is not None else sys.argv[1:]
        forward_args = [token for token in source_args if token != "--background"]
        spawn_background_process(forward_args)
        return 0

    configure_logging(args.log_file, args.verbose)

    if args.fresh and args.resume:
        LOGGER.error("Cannot use --fresh and --resume together.")
        return 1

    if args.finnhub_key:
        key = args.finnhub_key.strip()
        os.environ["FINNHUB_API_KEY"] = key
        backtest.FINNHUB_API_KEY = key

    state_path = Path(args.state_file).expanduser().resolve()

    if args.fresh and state_path.exists():
        try:
            state_path.unlink()
            LOGGER.info("Removed existing session state at %s", state_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to delete %s: %s", state_path, exc)

    resume_requested = args.resume or (not args.fresh and state_path.exists())

    if resume_requested:
        loaded_state = load_state(state_path)
        if loaded_state is None:
            LOGGER.info("No previous session to resume. Starting fresh.")
            state = default_state(args.capital)
        else:
            state = loaded_state
            if abs(state.get("initial_capital", args.capital) - args.capital) > 1e-6:
                LOGGER.info(
                    "Ignoring --capital %.2f in favour of saved initial capital %.2f.",
                    args.capital,
                    state.get("initial_capital", args.capital),
                )
            args.capital = float(state.get("initial_capital", args.capital))
    else:
        state = default_state(args.capital)

    genome = load_genome(args.genome)
    history: Dict[str, HistoryEntry] = {}
    news_cache: Dict[str, Dict[str, object]] = {}
    dashboard: EquityDashboard | None = None
    if not args.no_gui:
        if not GUI_AVAILABLE:
            LOGGER.warning("GUI backend unavailable; continuing without live dashboard.")
        else:
            try:
                dashboard = EquityDashboard()
                dashboard.process_events()
                LOGGER.info("Live dashboard started.")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to launch live dashboard: %s", exc)
                dashboard = None
    state["dashboard"] = dashboard
    if dashboard and state.get("equity_history"):
        dashboard.update(state["equity_history"])

    stop_requested = False

    def _signal_handler(signum, _frame):
        nonlocal stop_requested
        LOGGER.info("Received signal %s, preparing to shut down.", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while True:
            if stop_requested:
                break
            if dashboard and dashboard.closed:
                stop_requested = True
                break
            if dashboard and not dashboard.process_events():
                stop_requested = True
                break

            start_ts = time.time()
            run_once(genome, history, news_cache, args, state)
            if dashboard and not dashboard.closed:
                dashboard.process_events()
            if args.once or stop_requested:
                break
            if dashboard and dashboard.closed:
                stop_requested = True
                break
            elapsed = time.time() - start_ts
            sleep_for = max(1.0, args.interval - elapsed)
            LOGGER.debug("Sleeping for %.2f seconds.", sleep_for)
            end_time = time.monotonic() + sleep_for
            while time.monotonic() < end_time:
                remaining = end_time - time.monotonic()
                chunk = min(0.2, max(0.05, remaining))
                time.sleep(chunk)
                if stop_requested:
                    break
                if dashboard:
                    if dashboard.closed:
                        stop_requested = True
                        break
                    if not dashboard.process_events():
                        stop_requested = True
                        break
            if dashboard and dashboard.closed:
                stop_requested = True
                break
    finally:
        LOGGER.info("Live runner stopped.")
        dashboard = state.get("dashboard")  # type: ignore[assignment]
        if isinstance(dashboard, EquityDashboard):
            dashboard.close()
        state.pop("dashboard", None)
        try:
            save_state(state_path, state)
        except Exception:
            LOGGER.warning("Unable to persist session state to %s", state_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
