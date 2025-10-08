import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS: List[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "NVDA",
    "AMD",
    "INTC",
    "XOM",
    "CVX",
    "BP",
    "JPM",
    "BAC",
    "WMT",
    "PG",
    "UNH",
    "DIS",
    "V",
    "KO",
    "JNJ",
]

FEATURE_NAMES = ["short_diff", "long_diff", "vol5_z", "vol_spike_z"]
FEATURES_PER_TICKER = len(FEATURE_NAMES)

ENTRY_THRESHOLD = 0.25
SIGNAL_BIAS = 0.20
WEIGHT_CLIP = 1.0
LEVERAGE = 5.0

LOOKBACK_DAYS = 400  # candles to pull for feature computation
BEST_GENOME_PATH = "best_genome.npy"
FINNHUB_API_KEY = (os.environ.get("FINNHUB_API_KEY") or "").strip()
NEWS_LOOKBACK_DAYS = 5
FINNHUB_API_KEY = 'd0tp9v9r01qlvahdj2lgd0tp9v9r01qlvahdj2m0'

INITIAL_BALANCE = 1_000.0
RISK_FREE_RATE = 0.0  # for daily Sharpe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Dict[str, object]]

    @property
    def final_balance(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        start = float(self.equity_curve.iloc[0])
        return self.final_balance / start - 1.0

    @property
    def annualized_return(self) -> float:
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            return 0.0
        return (self.final_balance / float(self.equity_curve.iloc[0])) ** (365.0 / days) - 1.0

    @property
    def max_drawdown(self) -> float:
        curve = self.equity_curve.values
        running_max = np.maximum.accumulate(curve)
        drawdowns = 1.0 - curve / running_max
        return float(np.nanmax(drawdowns))

    @property
    def sharpe(self) -> float:
        returns = self.equity_curve.pct_change().dropna()
        if returns.empty:
            return 0.0
        excess = returns - (RISK_FREE_RATE / 365.0)
        std = excess.std()
        if std == 0.0 or not np.isfinite(std):
            return 0.0
        return float(np.sqrt(252.0) * excess.mean() / std)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_genome(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best genome file '{path}' not found. Run dev_trainer.py first to generate it."
        )
    genome = np.load(path)
    expected_length = len(TICKERS) * FEATURES_PER_TICKER
    if genome.ndim != 1 or genome.size != expected_length:
        raise ValueError(f"Genome shape mismatch. Expected {expected_length}, got {genome.size}.")
    return genome.astype(float)


def load_company_news(ticker: str, start: date, end: date) -> Dict[date, List[str]]:
    if not FINNHUB_API_KEY:
        return {}
    params = {
        "symbol": ticker,
        "from": (start - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }
    try:
        resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
        resp.raise_for_status()
        stories = resp.json()
    except Exception as exc:
        print(f"[warn] News fetch failed for {ticker}: {exc}")
        return {}
    news_map: Dict[date, List[str]] = {}
    if isinstance(stories, list):
        for story in stories:
            ts = story.get("datetime")
            if ts is None:
                continue
            try:
                story_date = datetime.utcfromtimestamp(float(ts)).date()
            except Exception:
                continue
            if story_date < start - timedelta(days=NEWS_LOOKBACK_DAYS) or story_date > end:
                continue
            headline = story.get("headline") or story.get("summary") or "N/A"
            news_map.setdefault(story_date, []).append(headline)
    return news_map


def select_headline(news_map: Dict[date, List[str]], target_date: date) -> str:
    if not news_map:
        return "N/A"
    for offset in range(NEWS_LOOKBACK_DAYS + 1):
        query_date = target_date - timedelta(days=offset)
        headlines = news_map.get(query_date)
        if headlines:
            return headlines[0]
    return "N/A"

def download_history(symbol: str, start: datetime, end: datetime) -> pd.Series:
    df = yf.download(
        symbol,
        start=start - timedelta(days=60),  # small buffer for indicators
        end=end + timedelta(days=5),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Could not download data for {symbol}")
    closes = df["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes.name = symbol
    return closes.astype(float)


def compute_feature_frame(close: pd.Series) -> pd.DataFrame:
    close = close.sort_index()
    short_ma = close.rolling(window=10, min_periods=5).mean()
    long_ma = close.rolling(window=50, min_periods=10).mean()
    returns = close.pct_change()
    vol5 = returns.rolling(window=5, min_periods=3).std()
    vol_spike = vol5 / vol5.shift(1)

    def _std_z(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        return ((series - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df = pd.DataFrame(
        {
            "close": close,
            "ret": returns.fillna(0.0),
            "short_diff": ((close - short_ma) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0),
            "long_diff": ((close - long_ma) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0),
            "vol5_z": _std_z(vol5.fillna(0.0)),
            "vol_spike_z": _std_z(vol_spike.fillna(0.0)),
        }
    )
    return df


def compute_signals_for_day(
    genome: np.ndarray,
    feature_frames: Dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
) -> Dict[str, float]:
    signals: Dict[str, float] = {}
    for idx, ticker in enumerate(TICKERS):
        frame = feature_frames.get(ticker)
        if frame is None or current_date not in frame.index:
            continue
        row = frame.loc[current_date]
        if row.isnull().any():
            continue

        base = idx * FEATURES_PER_TICKER
        signal = (
            genome[base + 0] * row["short_diff"]
            + genome[base + 1] * row["long_diff"]
            + genome[base + 2] * row["vol5_z"]
            + genome[base + 3] * row["vol_spike_z"]
            + SIGNAL_BIAS
        )
        signal = float(np.clip(signal, -WEIGHT_CLIP, WEIGHT_CLIP))
        if abs(signal) < ENTRY_THRESHOLD:
            continue
        signals[ticker] = signal
    return signals


def normalize_weights(signals: Dict[str, float]) -> Dict[str, float]:
    if not signals:
        return {}
    long_sum = sum(sig for sig in signals.values() if sig > 0)
    short_sum = sum(-sig for sig in signals.values() if sig < 0)

    weights: Dict[str, float] = {}
    for ticker, signal in signals.items():
        if signal > 0 and long_sum > 0:
            weights[ticker] = signal / long_sum
        elif signal < 0 and short_sum > 0:
            weights[ticker] = -signal / short_sum * -1.0
    return weights


def run_backtest(
    genome: np.ndarray,
    start_date: datetime,
    end_date: datetime,
) -> BacktestResult:
    feature_frames: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        try:
            closes = download_history(ticker, start_date - timedelta(days=LOOKBACK_DAYS), end_date)
            feature_frames[ticker] = compute_feature_frame(closes)
        except Exception as exc:
            print(f"[warn] Skipping {ticker}: {exc}")

    news_cache: Dict[str, Dict[date, List[str]]] = {}
    news_start = start_date.date()
    news_end = end_date.date()
    if FINNHUB_API_KEY:
        for ticker in feature_frames.keys():
            news_cache[ticker] = load_company_news(ticker, news_start, news_end)

    if not feature_frames:
        raise RuntimeError("No data available for any ticker; cannot run backtest.")

    all_dates = sorted(
        set().union(*[frame.loc[start_date:end_date].index for frame in feature_frames.values()])
    )
    if len(all_dates) < 2:
        raise RuntimeError("Not enough overlapping dates across tickers.")

    balance = INITIAL_BALANCE
    equity = []
    trades: List[Dict[str, object]] = []

    for idx in range(len(all_dates) - 1):
        current_date = all_dates[idx]
        next_date = all_dates[idx + 1]

        signals = compute_signals_for_day(genome, feature_frames, current_date)
        weights = normalize_weights(signals)

        if weights:
            positive_returns: List[Tuple[str, float, float]] = []
            daily_trades: List[Dict[str, object]] = []
            for ticker, weight in weights.items():
                frame = feature_frames.get(ticker)
                if frame is None:
                    continue
                if current_date not in frame.index or next_date not in frame.index:
                    continue
                cur_close = frame.at[current_date, "close"]
                next_close = frame.at[next_date, "close"]
                if cur_close <= 0 or next_close <= 0:
                    continue
                ret = (next_close / cur_close) - 1.0
                trade_record = {
                    "date": next_date.date().isoformat(),
                    "ticker": ticker,
                    "weight": weight,
                    "entry": float(cur_close),
                    "exit": float(next_close),
                    "return": float(ret),
                    "article": select_headline(news_cache.get(ticker, {}), next_date.date()),
                }
                if ret > 0:
                    positive_returns.append((ticker, weight, ret))
                daily_trades.append(trade_record)
            if daily_trades:
                trades.extend(daily_trades)
            if positive_returns:
                norm = sum(abs(weight) for _, weight, _ in positive_returns)
                if norm <= 0:
                    norm = len(positive_returns)
                portfolio_ret = sum((weight / norm) * ret for _, weight, ret in positive_returns)
                balance *= max(0.01, 1.0 + LEVERAGE * portfolio_ret)
        equity.append((next_date, balance))

    equity_curve = pd.Series([bal for _, bal in equity], index=[dt for dt, _ in equity])
    equity_curve.iloc[0] = INITIAL_BALANCE
    return BacktestResult(equity_curve=equality_align(equity_curve), trades=trades)


def equality_align(series: pd.Series) -> pd.Series:
    series.iloc[0] = INITIAL_BALANCE
    return series.astype(float)


def summarize(result: BacktestResult) -> None:
    print("\n=== Historical Backtest Summary ===")
    print(f"Start: {result.equity_curve.index[0].date()} | End: {result.equity_curve.index[-1].date()}")
    print(f"Final Balance: ${result.final_balance:,.2f}")
    print(f"Total Return: {result.total_return * 100:.2f}%")
    print(f"Annualized Return: {result.annualized_return * 100:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genome-based historical backtest using live APIs.")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD). Defaults to 2015-01-01.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD). Defaults to 2018-12-31.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="historical_equity.csv",
        help="Path to save the equity curve CSV.",
    )
    parser.add_argument(
        "--finnhub-key",
        type=str,
        default=None,
        help="Finnhub API key (overrides FINNHUB_API_KEY env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global FINNHUB_API_KEY
    if args.finnhub_key:
        FINNHUB_API_KEY = args.finnhub_key.strip()

    if not FINNHUB_API_KEY:
        print("[warn] FINNHUB_API_KEY not set. News headlines will be 'N/A'.")

    default_start = datetime(2015, 1, 1).date()
    default_end = datetime(2018, 12, 31).date()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else default_start
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else default_end

    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    genome = load_best_genome(BEST_GENOME_PATH)
    result = run_backtest(genome, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time()))

    summarize(result)
    result.equity_curve.to_csv(args.output, header=["equity"])
    print(f"[info] Saved equity curve to {args.output}")

    # Save trades
    trades_df = pd.DataFrame(result.trades)
    if not trades_df.empty:
        trades_df.sort_values(by="return", ascending=False, inplace=True)
        trades_df.to_csv("historical_trades.csv", index=False)
        print(f"[info] Saved trades to historical_trades.csv ({len(trades_df)} rows)")
    else:
        print("[warn] No trades were recorded.")

    # Generate equity curve plot
    plt.figure(figsize=(10, 5))
    result.equity_curve.plot(title="Equity Curve", ylabel="Balance (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("historical_equity.png")
    plt.close()
    print("[info] Saved equity curve plot to historical_equity.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)
