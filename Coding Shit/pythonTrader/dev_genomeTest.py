from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dev_trainer as trainer


EQUITY_PLOT_PATH = Path("genome_equity.png")


def load_best_genome(path: Path = Path("best_genome.npy")) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Saved genome file '{path}' not found. Run dev_trainer.py to train and persist a genome."
        )
    genome = np.load(path)
    expected = trainer.GENOME_LENGTH
    if genome.shape[0] != expected:
        raise ValueError(f"Genome length mismatch: expected {expected}, found {genome.shape[0]} in '{path}'")
    return genome


def equity_with_dates(genome: np.ndarray, start, end):
    equity, stats = trainer.simulate_equity(genome, start, end)
    dates = pd.date_range(start=start, end=end, freq="D")
    length = min(len(equity), len(dates))
    return equity[:length], dates[:length], stats


def format_ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def describe_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    print(
        f"Avg Sharpe: {format_ratio(metrics.get('avg_sharpe'))} | "
        f"Avg Return: {format_pct(metrics.get('avg_return'))} | "
        f"Worst DD: {format_pct(metrics.get('worst_drawdown'))} | "
        f"Trades: {metrics.get('trades', 'n/a')} | "
        f"Windows: {metrics.get('windows', 'n/a')}"
    )
    if "full_period_return" in metrics:
        print(
            f"Full-period Return: {format_pct(metrics.get('full_period_return'))} | "
            f"Full-period Drawdown: {format_pct(metrics.get('full_period_drawdown'))} | "
            f"Full-period Trades: {metrics.get('full_period_trades', 'n/a')}"
        )


def plot_equity(dates: pd.DatetimeIndex, equity: np.ndarray) -> None:
    if len(equity) == 0:
        print("No equity data to plot.")
        return

    plt.figure(figsize=(11, 4))
    plt.plot(dates, equity, label="Equity")

    holdout_start = pd.Timestamp(trainer.HOLDOUT_START)
    holdout_end = pd.Timestamp(trainer.HOLDOUT_END)
    test_start = pd.Timestamp(trainer.TEST_START)
    test_end = pd.Timestamp(trainer.TEST_END)

    # Test span first so holdout overlay remains visible
    plt.axvspan(
        test_start,
        test_end,
        alpha=0.08,
        color="#003f5c",
        label=f"Test ({trainer.TEST_START:%Y}-{trainer.TEST_END:%Y})",
    )
    plt.axvspan(
        holdout_start,
        holdout_end,
        alpha=0.18,
        color="#ffa600",
        label=f"Holdout ({trainer.HOLDOUT_START:%Y}-{trainer.HOLDOUT_END:%Y})",
    )

    plt.title("Equity Curve (Holdout + Test)")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EQUITY_PLOT_PATH)
    plt.close()
    print(f"Saved equity curve plot to {EQUITY_PLOT_PATH}")


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    drawdowns = drawdowns[np.isfinite(drawdowns)]
    return float(drawdowns.max()) if drawdowns.size else 0.0


def main() -> None:
    trainer.load_data()
    genome = load_best_genome()

    training_metrics = trainer.compute_training_metrics(genome)
    holdout_metrics = trainer.evaluate_holdout(genome)
    test_metrics = trainer.evaluate_test(genome)

    describe_metrics(
        f"Training Metrics ({trainer.TRAIN_START:%Y}-{trainer.TRAIN_END:%Y})",
        training_metrics,
    )
    describe_metrics(
        f"Holdout Metrics ({trainer.HOLDOUT_START:%Y}-{trainer.HOLDOUT_END:%Y})",
        holdout_metrics,
    )
    describe_metrics(
        f"Test Metrics ({trainer.TEST_START:%Y}-{trainer.TEST_END:%Y})",
        test_metrics,
    )

    equity_full, dates_full, _ = equity_with_dates(genome, trainer.TEST_START, trainer.TEST_END)

    if equity_full.size:
        final_balance = float(equity_full[-1])
        total_return = float(equity_full[-1] / equity_full[0] - 1.0)
        max_dd = max_drawdown(equity_full)
        print(
            f"\nOut-of-sample ending balance ({trainer.TEST_START:%Y}-{trainer.TEST_END:%Y}): {final_balance:.2f} | "
            f"Return: {format_pct(total_return)} | Max Drawdown: {format_pct(max_dd)}"
        )
        plot_equity(dates_full, equity_full)
    else:
        print("No equity curve generated for combined period.")

    if os.name == "nt":
        os.system("pause")
    else:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
