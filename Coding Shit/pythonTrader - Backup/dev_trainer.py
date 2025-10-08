from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_START = date(2010, 1, 1)
TRAIN_END = date(2014, 12, 31)
HOLDOUT_START = date(2015, 1, 1)
HOLDOUT_END = date(2016, 12, 31)
# Final evaluation window (entire out-of-sample period)
TEST_START = HOLDOUT_START
TEST_END = date(2018, 12, 31)

TRAIN_DAYS, TEST_DAYS = 180, 60
ENTRY_THRESHOLD = 0.25  # require stronger signal before opening a trade
MAX_CONCURRENT_POS = 6

TICKERS = [
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

GENOME_LENGTH = len(TICKERS) * FEATURES_PER_TICKER
POP_SIZE, GENS = 30, 18
ELITE_COUNT = 5
MUTATIONS_PER_ELITE = 6
TOURNAMENT, CROSSOVER, MUTATE, SIGMA = 4, 0.0, 0.12, 0.08
TP, SL, MAX_HOLD = 0.03, 0.02, 3

MIN_TRADES = 60
MAX_DRAWDOWN_LIMIT = 0.4
RETURN_WEIGHT = 100.0
DD_PENALTY = 180.0
TRADE_BONUS = 0.06
CV_FOLDS = 5
EVAL_SAMPLE_FRACTION = 1.0
IMMIGRANTS_PER_GEN = 3
TRANSACTION_COST = 0.0005  # retained for compatibility (unused)
HOLDOUT_DD_PENALTY = 70000.0
HOLDOUT_RETURN_PENALTY = 40.0
SIGNAL_BIAS = 0.2
HOLDOUT_PENALTY_START_GEN = 12
DRAW_HARD_BASE = 70000.0  # training hard penalty base
DRAW_HARD_MULT = 30000.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_file = os.path.join(os.getcwd(), "trainer_debug.log")
logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")


# ---------------------------------------------------------------------------
# Global data holders (populated by load_data)
# ---------------------------------------------------------------------------

BEST_GENOME_PATH = Path("best_genome.npy")
BEST_METRICS_PATH = Path("best_genome_metrics.json")

price_data: Dict[str, pd.Series] = {}
feature_data: Dict[str, Dict[str, pd.Series]] = {name: {} for name in FEATURE_NAMES}
returns_data: Dict[str, pd.Series] = {}
feature_stats: Dict[str, Dict[str, Dict[str, float]]] = {name: {} for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _timestamp(value: date) -> pd.Timestamp:
    return pd.Timestamp(value)


def _ensure_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        raise ValueError("Expected single-column DataFrame for price data")
    raise TypeError(f"Unsupported object type for series conversion: {type(obj)!r}")


def _standardize(series: pd.Series, mask: pd.Series) -> Tuple[pd.Series, float, float]:
    base = series.loc[mask]
    mean = float(base.mean()) if not base.empty else 0.0
    std = float(base.std()) if not base.empty else 0.0
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    standardized = (series - mean) / std
    standardized = standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return standardized, mean, std


def generate_windows(start: date, end: date, train: int, test: int) -> List[Tuple[date, date, date, date]]:
    windows: List[Tuple[date, date, date, date]] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(days=train - 1)
        test_start = train_end + timedelta(days=1)
        test_end = train_end + timedelta(days=test)
        if test_end > end:
            break
        windows.append((cursor, train_end, test_start, test_end))
        cursor += timedelta(days=test)
    return windows


ALL_SPLITS = generate_windows(TRAIN_START, TEST_END, TRAIN_DAYS, TEST_DAYS)
TRAIN_SPLITS = [s for s in ALL_SPLITS if s[3] <= TRAIN_END]
HOLDOUT_SPLITS = [s for s in ALL_SPLITS if HOLDOUT_START <= s[2] and s[3] <= HOLDOUT_END]
TEST_SPLITS = [s for s in ALL_SPLITS if s[2] >= TEST_START and s[3] <= TEST_END]


def _split_into_folds(splits: Sequence[Tuple[date, date, date, date]], folds: int) -> List[List[Tuple[date, date, date, date]]]:
    if folds <= 1 or len(splits) <= 1:
        return [list(splits)]
    buckets: List[List[Tuple[date, date, date, date]]] = [[] for _ in range(folds)]
    for idx, window in enumerate(splits):
        buckets[idx % folds].append(window)
    # Remove any empty folds
    return [bucket for bucket in buckets if bucket]


TRAIN_FOLDS = _split_into_folds(TRAIN_SPLITS, CV_FOLDS)


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data() -> None:
    """Download price history and build normalized features."""

    train_start_ts = _timestamp(TRAIN_START)
    train_end_ts = _timestamp(TRAIN_END)
    print("Downloading historical data...")

    for ticker in TICKERS:
        df = yf.download(
            ticker,
            start=TRAIN_START - timedelta(days=120),
            end=TEST_END + timedelta(days=TEST_DAYS),
            progress=False,
        )
        if df.empty:
            logging.warning("No data downloaded for %s", ticker)
            continue

        close = _ensure_series(df["Close"]).astype(float)
        close.index = pd.to_datetime(close.index)

        price_data[ticker] = close
        returns_data[ticker] = close.pct_change().fillna(0.0)

        short_ma = close.rolling(window=10, min_periods=5).mean()
        long_ma = close.rolling(window=50, min_periods=10).mean()
        returns = close.pct_change()
        vol5 = returns.rolling(window=5, min_periods=3).std()
        vol_spike = vol5 / vol5.shift(1)

        train_mask = (close.index >= train_start_ts) & (close.index <= train_end_ts)
        if not train_mask.any():
            logging.warning("Ticker %s has no training-period data; skipping", ticker)
            del price_data[ticker]
            continue

        features_raw = {
            "short_diff": (close - short_ma) / close,
            "long_diff": (close - long_ma) / close,
            "vol5_z": vol5,
            "vol_spike_z": vol_spike.replace([np.inf, -np.inf], np.nan),
        }

        for name, series in features_raw.items():
            series = series.fillna(0.0)
            standardized, mean, std = _standardize(series, train_mask)
            feature_data[name][ticker] = standardized
            feature_stats[name][ticker] = {"mean": mean, "std": std}

    loaded = len(price_data)
    print(f"Loaded features for {loaded}/{len(TICKERS)} tickers.")


# ---------------------------------------------------------------------------
# Simulation and evaluation helpers
# ---------------------------------------------------------------------------

def _init_worker(context: Tuple[Dict[str, pd.Series], Dict[str, Dict[str, pd.Series]], Dict[str, pd.Series]]):
    global price_data, feature_data, returns_data
    price_ctx, feature_ctx, returns_ctx = context
    price_data = price_ctx
    feature_data = feature_ctx
    returns_data = returns_ctx


def _compute_returns(equity: np.ndarray) -> np.ndarray:
    if equity.size < 2:
        return np.array([])
    return np.diff(equity) / equity[:-1]


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    drawdowns = drawdowns[np.isfinite(drawdowns)]
    return float(drawdowns.max()) if drawdowns.size else 0.0


def _sample_splits(splits: Sequence[Tuple[date, date, date, date]], fraction: float) -> List[Tuple[date, date, date, date]]:
    if fraction >= 1.0 or len(splits) <= 1:
        return list(splits)
    target = max(1, int(len(splits) * fraction))
    return random.sample(list(splits), target)


def simulate_equity(genome: np.ndarray, start: date, end: date) -> Tuple[np.ndarray, Dict[str, int]]:
    balance = 1000.0
    equity: List[float] = []
    total_activity = 0
    dates = pd.date_range(start=start, end=end, freq="D")

    short_w = genome[0::FEATURES_PER_TICKER]
    long_w = genome[1::FEATURES_PER_TICKER]
    vol5_w = genome[2::FEATURES_PER_TICKER]
    vol_spike_w = genome[3::FEATURES_PER_TICKER]

    for ts in dates:
        active_weights: List[Tuple[float, float]] = []
        total_weight = 0.0

        for i, ticker in enumerate(TICKERS):
            if ticker not in price_data or ticker not in returns_data:
                continue
            if ts not in returns_data[ticker].index:
                continue

            ret = float(returns_data[ticker].loc[ts])
            if not np.isfinite(ret):
                continue

            features_for_ticker = []
            for name in FEATURE_NAMES:
                series = feature_data[name].get(ticker)
                features_for_ticker.append(series.get(ts, 0.0) if series is not None else 0.0)

            signal = (
                short_w[i] * features_for_ticker[0]
                + long_w[i] * features_for_ticker[1]
                + vol5_w[i] * features_for_ticker[2]
                + vol_spike_w[i] * features_for_ticker[3]
                + SIGNAL_BIAS
            )

            weight = max(0.0, signal)
            if weight <= 0.0:
                continue

            active_weights.append((weight, ret))
            total_weight += weight

        if total_weight <= 0.0:
            equity.append(balance)
            continue

        portfolio_return = 0.0
        for weight, ret in active_weights:
            portfolio_return += (weight / total_weight) * ret
        portfolio_return = float(np.clip(portfolio_return, -0.4, 0.4))

        balance *= max(0.01, 1.0 + portfolio_return)
        equity.append(balance)
        total_activity += len(active_weights)

    if not equity:
        equity = [1000.0, 1000.0]

    return np.asarray(equity, dtype=float), {"closed": total_activity, "opened": total_activity}


def _compute_metrics(
    genome: np.ndarray,
    splits: Sequence[Tuple[date, date, date, date]],
    *,
    min_trades_override: int | None = None,
    drawdown_limit_override: float | None = None,
    dd_penalty_override: float | None = None,
    hard_penalty_scale: float = 1.0,
) -> Dict[str, float]:
    sharpe_scores: List[float] = []
    returns: List[float] = []
    drawdowns: List[float] = []
    trades_total = 0

    for _, _, test_start, test_end in splits:
        equity, stats = simulate_equity(genome, test_start, test_end)
        if equity.size < 2:
            continue
        rets = _compute_returns(equity)
        if rets.size < 2:
            continue

        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-6))
        sharpe_scores.append(sharpe)
        returns.append(float(equity[-1] / equity[0] - 1.0))
        drawdowns.append(_max_drawdown(equity))
        trades_total += stats["closed"]

    metrics = {
        "windows": len(sharpe_scores),
        "avg_sharpe": float(np.mean(sharpe_scores)) if sharpe_scores else float("nan"),
        "avg_return": float(np.mean(returns)) if returns else float("nan"),
        "worst_drawdown": float(max(drawdowns)) if drawdowns else 0.0,
        "trades": trades_total,
        "valid": False,
        "score": float("-inf"),
    }

    if not sharpe_scores:
        metrics["score"] = -1e6
        return metrics

    required_trades = min_trades_override if min_trades_override is not None else MIN_TRADES
    if trades_total < required_trades:
        deficit = required_trades - trades_total
        metrics["score"] = -5_000 - deficit * 200.0
        return metrics

    worst_drawdown = metrics["worst_drawdown"]
    dd_limit = drawdown_limit_override if drawdown_limit_override is not None else MAX_DRAWDOWN_LIMIT
    if worst_drawdown > dd_limit:
        hard_penalty_base = DRAW_HARD_BASE * hard_penalty_scale
        metrics["score"] = -hard_penalty_base - worst_drawdown * DRAW_HARD_MULT * hard_penalty_scale
        return metrics

    avg_sharpe = metrics["avg_sharpe"]
    avg_return = metrics["avg_return"] if np.isfinite(metrics["avg_return"]) else 0.0

    dd_penalty = dd_penalty_override if dd_penalty_override is not None else DD_PENALTY
    score = avg_sharpe * 100.0 + avg_return * RETURN_WEIGHT - worst_drawdown * dd_penalty
    score += min(trades_total, 400) * TRADE_BONUS

    metrics["valid"] = True
    metrics["score"] = float(score)
    return metrics


def _aggregate_metrics(metrics_list: Sequence[Dict[str, float]]):
    valid_metrics = [m for m in metrics_list if m.get("valid")]
    if not valid_metrics:
        return {
            "windows": 0,
            "avg_sharpe": float("nan"),
            "avg_return": float("nan"),
            "worst_drawdown": 0.0,
            "trades": 0,
            "valid": False,
            "score": -1e6,
        }

    windows_total = sum(m["windows"] for m in valid_metrics)

    def weighted_average(field: str) -> float:
        numerator = 0.0
        denom = 0.0
        for m in valid_metrics:
            value = m.get(field)
            if value is None or not np.isfinite(value):
                continue
            weight = max(1, m["windows"])
            numerator += value * weight
            denom += weight
        return numerator / denom if denom else float("nan")

    worst_drawdown = max(
        (m["worst_drawdown"] for m in valid_metrics if np.isfinite(m["worst_drawdown"])),
        default=0.0,
    )

    aggregated = {
        "windows": windows_total,
        "avg_sharpe": weighted_average("avg_sharpe"),
        "avg_return": weighted_average("avg_return"),
        "worst_drawdown": worst_drawdown,
        "trades": sum(m.get("trades", 0) for m in valid_metrics),
        "valid": len(valid_metrics) == len(metrics_list),
        "score": float(np.mean([m["score"] for m in valid_metrics])),
    }
    return aggregated


def evaluate(
    genome: np.ndarray,
    eval_splits: Sequence[Tuple[date, date, date, date]] | None = None,
    generation: int = 0,
) -> float:
    attempts = 0
    fraction = EVAL_SAMPLE_FRACTION
    while attempts < 3:
        sampled = list(eval_splits) if eval_splits is not None else _sample_splits(TRAIN_SPLITS, fraction)
        required = max(15, int(MIN_TRADES * len(sampled) / max(1, len(TRAIN_SPLITS))))

        if generation >= HOLDOUT_PENALTY_START_GEN:
            dd_limit = MAX_DRAWDOWN_LIMIT
            dd_penalty_override = None
            hard_scale = 1.0
        else:
            dd_limit = MAX_DRAWDOWN_LIMIT + 0.15
            dd_penalty_override = DD_PENALTY * 0.5
            hard_scale = 0.5

        metrics = _compute_metrics(
            genome,
            sampled,
            min_trades_override=required,
            drawdown_limit_override=dd_limit,
            dd_penalty_override=dd_penalty_override,
            hard_penalty_scale=hard_scale,
        )
        score = metrics["score"]
        penalty = 0.0

        if generation >= HOLDOUT_PENALTY_START_GEN:
            holdout_metrics = _compute_metrics(genome, HOLDOUT_SPLITS)
            holdout_dd = holdout_metrics.get("worst_drawdown", float("nan"))
            if not np.isfinite(holdout_dd):
                holdout_dd = 1.0
            holdout_return = holdout_metrics.get("avg_return", float("nan"))
            if not np.isfinite(holdout_return):
                holdout_return = -1.0
            penalty = max(0.0, holdout_dd - MAX_DRAWDOWN_LIMIT) * HOLDOUT_DD_PENALTY
            penalty += max(0.0, -holdout_return) * HOLDOUT_RETURN_PENALTY
            score -= penalty

        if metrics["valid"] and penalty == 0.0 and np.isfinite(score):
            return score
        if np.isfinite(score):
            return score
        fraction = min(1.0, fraction + 0.15)
        attempts += 1

    metrics = compute_training_metrics(genome)
    score = metrics["score"]
    if generation >= HOLDOUT_PENALTY_START_GEN:
        holdout_metrics = _compute_metrics(genome, HOLDOUT_SPLITS)
        holdout_dd = holdout_metrics.get("worst_drawdown", float("nan"))
        if not np.isfinite(holdout_dd):
            holdout_dd = 1.0
        holdout_return = holdout_metrics.get("avg_return", float("nan"))
        if not np.isfinite(holdout_return):
            holdout_return = -1.0
        penalty = max(0.0, holdout_dd - MAX_DRAWDOWN_LIMIT) * HOLDOUT_DD_PENALTY
        penalty += max(0.0, -holdout_return) * HOLDOUT_RETURN_PENALTY
        score -= penalty
    if np.isfinite(score):
        return score
    return -1e6


def compute_metrics(genome: np.ndarray, splits: Sequence[Tuple[date, date, date, date]] | None = None):
    selected = splits if splits is not None else TRAIN_SPLITS
    return _compute_metrics(genome, selected)


def compute_training_metrics(genome: np.ndarray):
    total_splits = max(1, len(TRAIN_SPLITS))
    fold_metrics = []
    for fold in TRAIN_FOLDS:
        required = max(15, int(MIN_TRADES * len(fold) / total_splits))
        fold_metrics.append(_compute_metrics(genome, fold, min_trades_override=required))
    return _aggregate_metrics(fold_metrics)


def evaluate_period(
    genome: np.ndarray,
    splits: Sequence[Tuple[date, date, date, date]],
    start: date,
    end: date,
) -> Dict[str, float]:
    metrics = compute_metrics(genome, splits)

    equity_full, stats_full = simulate_equity(genome, start, end)
    if equity_full.size >= 2:
        metrics["full_period_return"] = float(equity_full[-1] / equity_full[0] - 1.0)
        metrics["full_period_drawdown"] = _max_drawdown(equity_full)
    else:
        metrics["full_period_return"] = float("nan")
        metrics["full_period_drawdown"] = float("nan")
    metrics["full_period_trades"] = stats_full["closed"]

    return metrics


def evaluate_holdout(genome: np.ndarray):
    return evaluate_period(genome, HOLDOUT_SPLITS, HOLDOUT_START, HOLDOUT_END)


def evaluate_test(genome: np.ndarray):
    return evaluate_period(genome, TEST_SPLITS, TEST_START, TEST_END)


def tournament(population):
    best_genome = max(random.sample(population, TOURNAMENT), key=lambda x: x[1])[0]
    return np.array(best_genome, dtype=np.float32)


def mutate(genome: np.ndarray) -> None:
    for i in range(GENOME_LENGTH):
        if random.random() < MUTATE:
            genome[i] = float(np.clip(genome[i] + random.gauss(0, SIGMA), -1.0, 1.0))


def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pt = random.randint(1, GENOME_LENGTH - 1)
    child1 = np.concatenate([parent_a[:pt], parent_b[pt:]])
    child2 = np.concatenate([parent_b[:pt], parent_a[pt:]])
    return child1, child2


MAP_CHUNK_SIZE = 4


def _evaluate_parallel(args):
    genome, eval_split_hint, generation = args
    return genome, evaluate(genome, eval_split_hint, generation=generation)


def parallel_eval(genomes: Sequence[np.ndarray], generation: int = 0):
    context = (price_data, feature_data, returns_data)
    sampled_splits = _sample_splits(TRAIN_SPLITS, EVAL_SAMPLE_FRACTION)
    tasks = [(genome, sampled_splits, generation) for genome in genomes]
    with ProcessPoolExecutor(
        max_workers=max(1, multiprocessing.cpu_count()),
        initializer=_init_worker,
        initargs=(context,),
    ) as executor:
        results = list(
            tqdm(
                executor.map(_evaluate_parallel, tasks, chunksize=MAP_CHUNK_SIZE),
                total=len(tasks),
                desc="Evaluating",
                ncols=80,
            )
        )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genetic trainer for walk-forward trading strategy")
    parser.add_argument(
        "--seed-best",
        action="store_true",
        help="Include the saved best genome in the initial population if available.",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="Skip the interactive prompt for seeding when a saved genome exists.",
    )
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="Remove any saved best genome/metrics before training.",
    )
    return parser


def run(seed_best: np.ndarray | None = None):
    population: List[np.ndarray] = []
    if seed_best is not None:
        print("Seeding initial population with saved best genome.")
        population.append(np.copy(seed_best))

    while len(population) < POP_SIZE:
        population.append(np.random.uniform(-1, 1, GENOME_LENGTH))

    population = parallel_eval(population, generation=0)
    population.sort(key=lambda item: item[1], reverse=True)
    print(f"Initial best: {population[0][1]:.2f}")

    for generation in range(1, GENS + 1):
        elite_count = min(ELITE_COUNT, len(population))
        elites = [np.copy(population[i][0]) for i in range(elite_count)]

        next_gen: List[np.ndarray] = [np.copy(genome) for genome in elites]

        for elite in elites:
            for _ in range(MUTATIONS_PER_ELITE):
                if len(next_gen) >= max(ELITE_COUNT, POP_SIZE - IMMIGRANTS_PER_GEN):
                    break
                clone = np.copy(elite)
                mutate(clone)
                next_gen.append(clone)

        while len(next_gen) < max(ELITE_COUNT, POP_SIZE - IMMIGRANTS_PER_GEN):
            elite = elites[len(next_gen) % elite_count]
            clone = np.copy(elite)
            mutate(clone)
            next_gen.append(clone)

        while len(next_gen) < POP_SIZE - IMMIGRANTS_PER_GEN:
            elite = elites[random.randrange(elite_count)]
            clone = np.copy(elite)
            mutate(clone)
            next_gen.append(clone)

        while len(next_gen) < POP_SIZE:
            next_gen.append(np.random.uniform(-1, 1, GENOME_LENGTH))

        population = parallel_eval(next_gen, generation=generation)
        population.sort(key=lambda item: item[1], reverse=True)
        best_score = population[0][1]
        avg_score = float(np.mean([score for _, score in population]))
        if generation % 2 == 0 or generation == GENS:
            print(f"Gen {generation:02d} best: {best_score:.2f}, avg: {avg_score:.2f}")

    logging.shutdown()
    return population[0]


def _to_serializable(value):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _write_metrics(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({k: _to_serializable(v) for k, v in payload.items()}, handle, indent=2)


def _load_saved_genome(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        genome = np.load(path)
    except Exception as exc:
        print(f"Failed to load saved genome from {path}: {exc}")
        return None
    if genome.shape[0] != GENOME_LENGTH:
        print(
            f"Saved genome length mismatch (expected {GENOME_LENGTH}, got {genome.shape[0]}). Ignoring seed."
        )
        return None
    return genome.astype(float)


def _pause_before_exit() -> None:
    if os.name == "nt":
        os.system("pause")
    else:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.reset_best:
        if BEST_GENOME_PATH.exists():
            BEST_GENOME_PATH.unlink()
            print("Deleted existing best genome file.")
        if BEST_METRICS_PATH.exists():
            BEST_METRICS_PATH.unlink()
            print("Deleted existing metrics file.")

    print("Starting trainer...")
    load_data()

    if not price_data:
        print("No price data was loaded. Check network connectivity or ticker list.")
        return

    seed_genome: np.ndarray | None = None
    saved_genome = _load_saved_genome(BEST_GENOME_PATH)
    if saved_genome is not None:
        if args.seed_best:
            seed_genome = saved_genome
        elif not args.skip_prompt:
            choice = input(
                "A saved best genome is available. Seed the initial population with it? [Y/N]: "
            ).strip().lower()
            if choice in {"y", "yes"}:
                seed_genome = saved_genome

    best_genome, best_score = run(seed_best=seed_genome)

    training_metrics = compute_training_metrics(best_genome)
    holdout_metrics = evaluate_holdout(best_genome)
    test_metrics = evaluate_test(best_genome)

    print("\n=== Training Metrics ===")
    print(f"Score: {training_metrics['score']:.2f}")
    print(
        f"Avg Sharpe: {training_metrics['avg_sharpe']:.3f} | "
        f"Avg Return: {training_metrics['avg_return'] * 100:.2f}% | "
        f"Worst DD: {training_metrics['worst_drawdown'] * 100:.2f}% | "
        f"Trades: {training_metrics['trades']}"
    )

    print(
        f"\n=== Holdout Metrics ({HOLDOUT_START:%Y}-{HOLDOUT_END:%Y}) ==="
    )
    print(
        f"Avg Sharpe: {holdout_metrics['avg_sharpe']:.3f} | "
        f"Avg Return: {holdout_metrics['avg_return'] * 100:.2f}% | "
        f"Worst DD: {holdout_metrics['worst_drawdown'] * 100:.2f}% | "
        f"Trades: {holdout_metrics['trades']}"
    )
    print(
        f"Full-period Return: {holdout_metrics['full_period_return'] * 100:.2f}% | "
        f"Full-period DD: {holdout_metrics['full_period_drawdown'] * 100:.2f}% | "
        f"Full-period Trades: {holdout_metrics['full_period_trades']}"
    )

    print(
        f"\n=== Test Metrics ({TEST_START:%Y}-{TEST_END:%Y}) ==="
    )
    print(
        f"Avg Sharpe: {test_metrics['avg_sharpe']:.3f} | "
        f"Avg Return: {test_metrics['avg_return'] * 100:.2f}% | "
        f"Worst DD: {test_metrics['worst_drawdown'] * 100:.2f}% | "
        f"Trades: {test_metrics['trades']}"
    )
    print(
        f"Full-period Return: {test_metrics['full_period_return'] * 100:.2f}% | "
        f"Full-period DD: {test_metrics['full_period_drawdown'] * 100:.2f}% | "
        f"Full-period Trades: {test_metrics['full_period_trades']}"
    )

    np.save(BEST_GENOME_PATH, best_genome)
    print("Saved best genome to best_genome.npy")

    metrics_payload = {
        "training_score": training_metrics["score"],
        "training_avg_sharpe": training_metrics["avg_sharpe"],
        "training_avg_return": training_metrics["avg_return"],
        "training_worst_drawdown": training_metrics["worst_drawdown"],
        "training_trades": training_metrics["trades"],
        "training_valid": training_metrics["valid"],
        "holdout_avg_sharpe": holdout_metrics["avg_sharpe"],
        "holdout_avg_return": holdout_metrics["avg_return"],
        "holdout_worst_drawdown": holdout_metrics["worst_drawdown"],
        "holdout_trades": holdout_metrics["trades"],
        "holdout_full_period_return": holdout_metrics["full_period_return"],
        "holdout_full_period_drawdown": holdout_metrics["full_period_drawdown"],
        "holdout_full_period_trades": holdout_metrics["full_period_trades"],
        "holdout_valid": holdout_metrics["valid"],
        "test_avg_sharpe": test_metrics["avg_sharpe"],
        "test_avg_return": test_metrics["avg_return"],
        "test_worst_drawdown": test_metrics["worst_drawdown"],
        "test_trades": test_metrics["trades"],
        "test_full_period_return": test_metrics["full_period_return"],
        "test_full_period_drawdown": test_metrics["full_period_drawdown"],
        "test_full_period_trades": test_metrics["full_period_trades"],
        "test_valid": test_metrics["valid"],
    }
    _write_metrics(str(BEST_METRICS_PATH), metrics_payload)
    print("Saved metrics to best_genome_metrics.json")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        main()
    except Exception as exc:
        import traceback

        print("\nFatal error during training:")
        traceback.print_exc()
    finally:
        _pause_before_exit()
