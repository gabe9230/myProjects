# walkforward_trainer.py
# Full trainer with walk-forward backtesting using actual signal features

from datetime import date, timedelta, datetime
import os
import numpy as np
import pandas as pd
import random
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
TRAIN_DAYS = 180
TEST_DAYS = 60
START_DATE = date(2010, 1, 1)
END_DATE = date(2015, 6, 30)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "INTC", "XOM", "CVX",
    "BP", "JPM", "BAC", "WMT", "PG", "UNH", "DIS", "V", "KO", "JNJ",
]
GENOME_LENGTH = len(TICKERS) * 6
POPULATION_SIZE = 50
GENERATIONS = 10
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.1
TP_PERCENT = 0.05
SL_PERCENT = 0.03
MAX_HOLD_DAYS = 3

# Global data cache
price_data = {}
short_ma_data = {}
long_ma_data = {}
vol5_data = {}
vol_spike_data = {}


def generate_walk_forward_windows(start_date, end_date, train_days, test_days):
    windows = []
    current_start = start_date
    while True:
        train_start = current_start
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > end_date:
            break
        windows.append((train_start, train_end, test_start, test_end))
        current_start += timedelta(days=test_days)
    return windows


walk_forward_splits = generate_walk_forward_windows(START_DATE, END_DATE, TRAIN_DAYS, TEST_DAYS)

# Load price data
print("Downloading historical data...")
for tkr in TICKERS:
    df = yf.download(
        tkr,
        start=START_DATE - timedelta(days=100),
        end=END_DATE + timedelta(days=TEST_DAYS),
        progress=False,
    )
    if df.empty:
        continue
    price_data[tkr] = df["Close"].copy()
    short_ma_data[tkr] = df["Close"].rolling(window=10).mean()
    long_ma_data[tkr] = df["Close"].rolling(window=50).mean()
    returns = df["Close"].pct_change()
    vol5_data[tkr] = returns.rolling(5).std()
    vol_spike_data[tkr] = vol5_data[tkr] / vol5_data[tkr].shift(1)


def simulate_equity_curve(genome, start, end, verbose=False, return_dates=False):
    balance = 1000.0
    equity_curve = []
    date_points = [] if return_dates else None
    open_positions = []
    dates = pd.date_range(start=start, end=end)
    EPS = 1e-4
    BIAS = 0.001

    sent_w = genome[0::6]
    geo_w = genome[1::6]
    short_w = genome[2::6]
    long_w = genome[3::6]
    vol5_w = genome[4::6]
    vol_spike_w = genome[5::6]

    for current_date in dates:
        # Close positions
        still_open = []
        for pos in open_positions:
            tkr = TICKERS[pos["i"]]
            if current_date not in price_data[tkr].index:
                if verbose:
                    print(f"Missing price for {tkr} on {current_date}")
                continue
            price = price_data[tkr].get(current_date, np.nan)
            if np.isnan(price):
                continue
            pnl = (price - pos["entry"]) / pos["entry"]
            if pos["short"]:
                pnl *= -1
            days_held = (current_date.date() - pos["date"]).days
            if pnl >= TP_PERCENT or pnl <= -SL_PERCENT or days_held >= MAX_HOLD_DAYS:
                if verbose:
                    print(
                        f"Closing position in {tkr} on {current_date}: pnl={pnl:.4f}, held={days_held}d"
                    )
                balance *= (1 + pnl)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open new positions
        for i, tkr in enumerate(TICKERS):
            if current_date not in price_data[tkr].index:
                continue
            price = price_data[tkr][current_date]
            short_ma = short_ma_data[tkr].get(current_date, price)
            long_ma = long_ma_data[tkr].get(current_date, price)
            vol5 = vol5_data[tkr].get(current_date, 0.0)
            vol_spike = vol_spike_data[tkr].get(current_date, 0.0)

            if verbose:
                print(
                    f"{tkr} signal weights: short={short_w[i]:.3f}, long={long_w[i]:.3f}, vol5={vol5_w[i]:.3f}, vol_spike={vol_spike_w[i]:.3f}"
                )
            signal = (
                short_w[i] * (price - short_ma) * 100
                + long_w[i] * (price - long_ma) * 100
                + vol5_w[i] * vol5 * 10
                + vol_spike_w[i] * vol_spike * 10
                + BIAS
            )
            if verbose:
                print(f"{current_date} {tkr} signal: {signal:.6f}")
            # if abs(signal) < EPS: continue  # allow weak signals to trade
            if any(p["i"] == i for p in open_positions):
                continue
            open_positions.append({"i": i, "entry": price, "date": current_date, "short": signal < 0})

        equity_curve.append(balance)
        if return_dates:
            date_points.append(current_date)

    if len(equity_curve) == 0:
        return np.array([1000.0, 1000.0])
    if len(equity_curve) <= 2 or balance == 1000.0:
        curve = np.array([1000.0, 1000.0])
    else:
        curve = np.array(equity_curve)

    if return_dates:
        return curve, pd.DatetimeIndex(date_points)
    return curve


def evaluate_individual(genome):
    all_returns = []
    for _, _, test_start, test_end in walk_forward_splits:
        eq = simulate_equity_curve(genome, test_start, test_end)
        ret = np.diff(eq) / eq[:-1]
        if len(ret) < 2:
            continue
        all_returns.extend(ret)
    if not all_returns:
        return -np.inf
    mean = np.mean(all_returns)
    std = np.std(all_returns) + 1e-6
    return mean / std * 100


def make_random_individual():
    return np.random.uniform(-1, 1, GENOME_LENGTH)


def tournament_selection(pop):
    return max(random.sample(pop, TOURNAMENT_SIZE), key=lambda x: x[1])[0].copy()


def single_point_crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    pt = random.randint(1, GENOME_LENGTH - 1)
    return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])


def mutate(genome):
    for i in range(GENOME_LENGTH):
        if random.random() < MUTATION_RATE:
            genome[i] += random.gauss(0, MUTATION_SIGMA)
            genome[i] = float(np.clip(genome[i], -1.0, 1.0))


def run_evolution():
    population = [(make_random_individual(), 0.0) for _ in range(POPULATION_SIZE)]
    population = [(g, evaluate_individual(g)) for g, _ in population]
    population.sort(key=lambda x: x[1], reverse=True)
    print(f"Initial best: {population[0][1]:.2f}")

    for gen in range(1, GENERATIONS + 1):
        new_pop = [population[0], population[1]]
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = single_point_crossover(p1, p2)
            mutate(c1)
            mutate(c2)
            new_pop.append((c1, evaluate_individual(c1)))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append((c2, evaluate_individual(c2)))
        population = sorted(new_pop, key=lambda x: x[1], reverse=True)
        print(
            f"Gen {gen:02d} -> best: {population[0][1]:.2f}, avg: {np.mean([f for _, f in population]):.2f}"
        )
    return population[0]


def load_saved_genome(path="best_genome.npy"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved genome file '{path}' not found. Run dev_trainer.py first.")
    genome = np.load(path)
    if genome.shape[0] != GENOME_LENGTH:
        raise ValueError(
            f"Genome length mismatch: expected {GENOME_LENGTH}, found {genome.shape[0]} in '{path}'"
        )
    return genome


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    saved_path = "best_genome.npy"
    if os.path.exists(saved_path):
        best = load_saved_genome(saved_path)
        fitness = evaluate_individual(best)
        print("Loaded saved genome from best_genome.npy")
    else:
        best, fitness = run_evolution()
        np.save(saved_path, best)
        print("Saved newly evolved genome to best_genome.npy")

    print("\nBest genome fitness:", fitness)
    print("Slice:", best[:10])

    equity_curve, equity_dates = simulate_equity_curve(
        best, START_DATE, END_DATE, verbose=False, return_dates=True
    )
    if len(equity_curve) > 1:
        final_balance = float(equity_curve[-1])
        total_return = (final_balance / float(equity_curve[0]) - 1.0) * 100
        print(f"Full-period ending balance: {final_balance:.2f} ({total_return:.2f}% return)")

        plt.figure(figsize=(10, 4))
        plt.plot(equity_dates, equity_curve, label="Equity")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Balance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = "genome_equity.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved equity curve plot to {output_path}")
    else:
        print("Equity curve too short to plot.")

    try:
        input("Press Enter to exit...")
    except EOFError:
        pass
