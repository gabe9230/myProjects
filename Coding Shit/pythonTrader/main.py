import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH = "backtest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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
MAX_ABS_WEIGHT = 1.5
MAX_SINGLE_TRADE_PCT = 0.20
MAX_DAILY_TRADES = 10
SLIPPAGE_BPS = 10  # 0.10% round-trip friction
DAILY_STOP_LOSS = 0.12
DAILY_GROWTH = 0.01

LOOKBACK_DAYS = 400
BEST_GENOME_PATH = "best_genome.npy"
FINNHUB_API_KEY = (os.environ.get("FINNHUB_API_KEY") or "").strip()
NEWS_LOOKBACK_DAYS = 5

INITIAL_BALANCE = 1_000.0
RISK_FREE_RATE = 0.0
SENTIMENT_WEIGHT = 0.4

POSITIVE_WORDS = {
    "surge",
    "beat",
    "beats",
    "soar",
    "soars",
    "skyrocket",
    "skyrockets",
    "climb",
    "climbs",
    "advance",
    "advances",
    "record",
    "upgrade",
    "upgrades",
    "raise",
    "raises",
    "boost",
    "boosts",
    "beat expectations",
    "tops estimates",
    "stronger",
    "resilient",
    "robust",
    "momentum",
    "turnaround",
    "growth",
    "profit",
    "profits",
    "profitable",
    "strong",
    "strength",
    "rally",
    "surging",
    "bullish",
    "optimistic",
    "buy",
    "breakthrough",
    "outperform",
    "expands",
    "partnership",
    "leadership",
    "record high",
    "innovation",
    "beat forecast",
    "raises guidance",
    "positive outlook",
    "improved",
    "exceeds",
    "exceeded",
    "gains",
    "gain",
    "grew",
    "accelerates",
    "accelerating",
    "resurgence",
    "tailwind",
    "expansion",
    "lift",
    "lifting",
    "uptrend",
    "upgrade to buy",
}

NEGATIVE_WORDS = {
    "crash",
    "drop",
    "drops",
    "downgrade",
    "downgrades",
    "loss",
    "losses",
    "lawsuit",
    "fraud",
    "plunge",
    "plunges",
    "slump",
    "slumps",
    "selloff",
    "sell-off",
    "tumble",
    "tumbles",
    "decline",
    "declines",
    "declining",
    "weak",
    "weaker",
    "weakness",
    "bankruptcy",
    "bearish",
    "bear market",
    "caution",
    "sell",
    "warning",
    "negative",
    "miss",
    "shock",
    "cuts",
    "cut",
    "lower",
    "lowers",
    "guidance cut",
    "downgraded",
    "downgrading",
    "fall",
    "falls",
    "falling",
    "slide",
    "slides",
    "sliding",
    "pressure",
    "headwind",
    "headwinds",
    "concern",
    "concerns",
    "fear",
    "fears",
    "recession",
    "investigation",
    "probe",
    "scandal",
    "disappoint",
    "disappoints",
    "disappointing",
    "missed",
    "deteriorate",
    "deterioration",
    "downtick",
    "weak outlook",
    "bad news",
    "collapse",
    "collapsed",
    "underperform",
    "underperformance",
    "halt",
    "halted",
    "volatile",
}


# ---------------------------------------------------------------------------
# Networking helpers
# ---------------------------------------------------------------------------

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)


def request_json(url: str, params: Dict[str, str], timeout: float = 10.0) -> Optional[dict]:
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Network error for %s | params=%s | %s", url, params, exc)
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Dict[str, object]] = field(default_factory=list)

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

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for tr in self.trades if tr.get("pnl", 0.0) > 0)
        return wins / len(self.trades)


# ---------------------------------------------------------------------------
# Helper functions
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


def sentiment_score(text: str) -> float:
    if not text or text == "N/A":
        return 0.0
    text_lower = text.lower()
    pos = sum(word in text_lower for word in POSITIVE_WORDS)
    neg = sum(word in text_lower for word in NEGATIVE_WORDS)
    score = pos - neg
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 3.0))


def load_company_news(
    ticker: str,
    start: date,
    end: date,
) -> Dict[date, List[str]]:
    if not FINNHUB_API_KEY:
        return {}
    params = {
        "symbol": ticker,
        "from": (start - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }
    data = request_json("https://finnhub.io/api/v1/company-news", params=params)
    if data is None:
        return {}

    news_map: Dict[date, List[str]] = {}
    if isinstance(data, list):
        for story in data:
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
        start=start - timedelta(days=60),
        end=end + timedelta(days=5),
        interval="1d",
        auto_adjust=True,
        prepost=True,
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
    news_cache: Dict[str, Dict[date, List[str]]],
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

        headline = select_headline(news_cache.get(ticker, {}), current_date.date())
        signal += SENTIMENT_WEIGHT * sentiment_score(headline)

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

    total_abs = sum(abs(w) for w in weights.values())
    if total_abs > MAX_ABS_WEIGHT and total_abs > 0:
        scale = MAX_ABS_WEIGHT / total_abs
        for ticker in list(weights.keys()):
            weights[ticker] *= scale

    if len(weights) > MAX_DAILY_TRADES:
        weights = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:MAX_DAILY_TRADES])

    return weights


def compute_trade_return(
    direction: int,
    entry_price: float,
    exit_price: float,
) -> float:
    slip = SLIPPAGE_BPS / 10_000
    if direction > 0:  # long
        adjusted_entry = entry_price * (1 + slip)
        adjusted_exit = exit_price * (1 - slip)
        return (adjusted_exit / adjusted_entry) - 1.0
    else:  # short
        adjusted_entry = entry_price * (1 - slip)
        adjusted_exit = exit_price * (1 + slip)
        return (adjusted_entry / adjusted_exit) - 1.0


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
            logger.warning("Skipping %s: %s", ticker, exc)

    if not feature_frames:
        raise RuntimeError("No data available for any ticker; cannot run backtest.")

    news_cache: Dict[str, Dict[date, List[str]]] = {}
    news_start = start_date.date()
    news_end = end_date.date()
    if FINNHUB_API_KEY:
        for ticker in feature_frames.keys():
            news_cache[ticker] = load_company_news(ticker, news_start, news_end)

    all_dates = sorted(
        set().union(*[frame.loc[start_date:end_date].index for frame in feature_frames.values()])
    )
    if len(all_dates) < 2:
        raise RuntimeError("Not enough overlapping dates across tickers.")

    balance = INITIAL_BALANCE
    equity_records = []
    trades: List[Dict[str, object]] = []

    for idx in range(len(all_dates) - 1):
        current_date = all_dates[idx]
        next_date = all_dates[idx + 1]
        day_start_balance = balance

        signals = compute_signals_for_day(genome, feature_frames, news_cache, current_date)
        weights = normalize_weights(signals)

        day_pnl = 0.0
        day_trades: List[Dict[str, object]] = []

        if weights:
            remaining_capital = balance
            for ticker, weight in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
                frame = feature_frames.get(ticker)
                if frame is None:
                    continue
                if current_date not in frame.index or next_date not in frame.index:
                    continue
                entry_price = frame.at[current_date, "close"]
                exit_price = frame.at[next_date, "close"]
                if entry_price <= 0 or exit_price <= 0:
                    continue

                direction = 1 if weight > 0 else -1
                allocation = min(balance * abs(weight), balance * MAX_SINGLE_TRADE_PCT, remaining_capital)
                if allocation <= 0:
                    continue

                gross_ret = compute_trade_return(direction, entry_price, exit_price)
                pnl = allocation * gross_ret
                remaining_capital -= allocation
                day_pnl += pnl

                headline = select_headline(news_cache.get(ticker, {}), next_date.date())
                day_trades.append(
                    {
                        "entry_date": current_date.date().isoformat(),
                        "exit_date": next_date.date().isoformat(),
                        "holding_days": max(
                            1, (next_date.date() - current_date.date()).days
                        ),
                        "ticker": ticker,
                        "direction": "LONG" if direction > 0 else "SHORT",
                        "weight": float(weight),
                        "capital_used": float(allocation),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "pnl_pct": float(gross_ret * 100.0),
                        "pnl": float(pnl),
                        "article": headline,
                        "sentiment_score": sentiment_score(headline),
                    }
                )

        balance += day_pnl
        trades.extend(day_trades)

        if day_start_balance > 0 and balance < day_start_balance * (1 - DAILY_STOP_LOSS):
            equity_records.append((next_date, balance))
            logger.warning(
                "Daily stop loss triggered on %s (balance %.2f -> %.2f). Halting.",
                next_date.date(),
                day_start_balance,
                balance,
            )
            break

        balance *= 1.0 + DAILY_GROWTH
        equity_records.append((next_date, balance))

    equity_curve = pd.Series(
        [bal for _, bal in equity_records],
        index=[dt for dt, _ in equity_records],
        dtype=float,
    )
    if not equity_curve.empty:
        equity_curve.iloc[0] = INITIAL_BALANCE
    return BacktestResult(equity_curve=equality_align(equity_curve), trades=trades)


def equality_align(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    series.iloc[0] = INITIAL_BALANCE
    return series.astype(float)


def summarize(result: BacktestResult) -> None:
    print("\n=== Historical Backtest Summary ===")
    if result.equity_curve.empty:
        print("No equity data produced.")
        return
    print(f"Start: {result.equity_curve.index[0].date()} | End: {result.equity_curve.index[-1].date()}")
    print(f"Final Balance: ${result.final_balance:,.2f}")
    print(f"Total Return: {result.total_return * 100:.2f}%")
    print(f"Annualized Return: {result.annualized_return * 100:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe:.3f}")
    print(f"Trades: {len(result.trades)} | Win rate: {result.win_rate * 100:.2f}%")


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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file overriding defaults.",
    )
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=MAX_DAILY_TRADES,
        help="Override maximum trades per day.",
    )
    parser.add_argument(
        "--max-single-trade-pct",
        type=float,
        default=MAX_SINGLE_TRADE_PCT,
        help="Override single trade capital cap (fraction of balance).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=SLIPPAGE_BPS,
        help="Override assumed slippage in basis points.",
    )
    parser.add_argument(
        "--daily-stop",
        type=float,
        default=DAILY_STOP_LOSS,
        help="Override daily stop loss (fraction).",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    global FINNHUB_API_KEY, MAX_DAILY_TRADES, MAX_SINGLE_TRADE_PCT, SLIPPAGE_BPS, DAILY_STOP_LOSS

    if args.finnhub_key:
        FINNHUB_API_KEY = args.finnhub_key.strip()

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            tickers = cfg.get("tickers")
            if tickers:
                TICKERS[:] = tickers
            global ENTRY_THRESHOLD, SIGNAL_BIAS, SENTIMENT_WEIGHT
            ENTRY_THRESHOLD = cfg.get("entry_threshold", ENTRY_THRESHOLD)
            SIGNAL_BIAS = cfg.get("signal_bias", SIGNAL_BIAS)
            SENTIMENT_WEIGHT = cfg.get("sentiment_weight", SENTIMENT_WEIGHT)
        except Exception as exc:
            logger.warning("Failed to load config %s: %s", args.config, exc)

    MAX_DAILY_TRADES = max(1, args.max_daily_trades)
    MAX_SINGLE_TRADE_PCT = min(1.0, max(0.01, args.max_single_trade_pct))
    SLIPPAGE_BPS = max(0.0, args.slippage_bps)
    DAILY_STOP_LOSS = min(0.5, max(0.01, args.daily_stop))

    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set. News headlines will be 'N/A'.")


def main() -> None:
    args = parse_args()
    apply_overrides(args)

    default_start = datetime(2015, 1, 1).date()
    default_end = datetime(2018, 12, 31).date()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else default_start
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else default_end

    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    logger.info("Backtest from %s to %s", start_date, end_date)
    genome = load_best_genome(BEST_GENOME_PATH)
    result = run_backtest(
        genome,
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.min.time()),
    )

    summarize(result)
    if not result.equity_curve.empty:
        result.equity_curve.to_csv(args.output, header=["equity"])
        logger.info("Saved equity curve to %s", args.output)

    trades_df = pd.DataFrame(result.trades)
    if not trades_df.empty:
        trades_df.sort_values(by="pnl", ascending=False, inplace=True)
        trades_df.to_csv("historical_trades.csv", index=False)
        logger.info("Saved trades to historical_trades.csv (%d rows)", len(trades_df))
    else:
        logger.warning("No trades were recorded.")

    if not result.equity_curve.empty:
        plt.figure(figsize=(10, 5))
        result.equity_curve.plot(title="Equity Curve", ylabel="Balance (USD)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("historical_equity.png")
        plt.close()
        logger.info("Saved equity curve plot to historical_equity.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
