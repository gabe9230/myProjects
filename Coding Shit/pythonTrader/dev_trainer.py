from datetime import date, timedelta
import numpy as np
import pandas as pd
import random
import yfinance as yf
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
from textblob import TextBlob

log_file = os.path.join(os.getcwd(), "trainer_debug.log")
logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")

# Constants
TRAIN_DAYS, TEST_DAYS = 180, 60
START_DATE, END_DATE = date(2010, 1, 1), date(2015, 6, 30)
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "INTC", "XOM", "CVX",
           "BP", "JPM", "BAC", "WMT", "PG", "UNH", "DIS", "V", "KO", "JNJ"]
GENOME_LENGTH = len(TICKERS) * 6
POP_SIZE, GENS = 50, 10
TOURNAMENT, CROSSOVER, MUTATE, SIGMA = 3, 0.8, 0.1, 0.1
TP, SL, MAX_HOLD = 0.05, 0.03, 3

price_data, short_ma, long_ma, vol5, vol_spike = {}, {}, {}, {}, {}
sentiment_data, geo_data = {}, {}

# Walkforward splits
def generate_windows(start, end, train, test):
	windows, now = [], start
	while True:
		train_end = now + timedelta(days=train - 1)
		test_start, test_end = train_end + timedelta(days=1), train_end + timedelta(days=test)
		if test_end > end: break
		windows.append((now, train_end, test_start, test_end))
		now += timedelta(days=test)
	return windows
splits = generate_windows(START_DATE, END_DATE, TRAIN_DAYS, TEST_DAYS)

# Preload price and news data
def load_data():
	print("Downloading historical data...")
	for tkr in TICKERS:
		df = yf.download(tkr, start=START_DATE - timedelta(days=100), end=END_DATE + timedelta(days=TEST_DAYS), progress=False)
		if df.empty: continue
		price_data[tkr] = df['Close']
		short_ma[tkr] = df['Close'].rolling(10).mean()
		long_ma[tkr] = df['Close'].rolling(50).mean()
		ret = df['Close'].pct_change()
		vol5[tkr] = ret.rolling(5).std()
		vol_spike[tkr] = vol5[tkr] / vol5[tkr].shift(1)

	df = pd.read_csv("stocknews.csv")
	df['Date'] = pd.to_datetime(df['Date'])
	df['Combined'] = df[[f'Top{i}' for i in range(1, 26)]].astype(str).agg(' '.join, axis=1)
	df['Sentiment'] = df['Combined'].apply(lambda x: TextBlob(x).sentiment.polarity)
	df['Geo'] = df['Combined'].str.lower().apply(lambda x: sum(1 for kw in ["war", "conflict", "treaty", "europe", "asia", "iran", "china", "russia", "sanction", "embassy"] if kw in x))
	sentiment = df.groupby('Date')['Sentiment'].mean()
	geo = df.groupby('Date')['Geo'].mean()
	for tkr in TICKERS:
		sentiment_data[tkr] = sentiment
		geo_data[tkr] = geo
	print("✅ Data loaded.")

def simulate_equity(genome, start, end):
	balance = 1000.0
	eq, open_pos = [], []
	dates = pd.date_range(start, end)
	B = 0.001
	sent_w = genome[0::6]; geo_w = genome[1::6]
	short_w = genome[2::6]; long_w = genome[3::6]
	vol5_w = genome[4::6]; vol_spike_w = genome[5::6]

	for date in dates:
		still_open, closed, opened = [], 0, 0
		for pos in open_pos:
			tkr, ts = TICKERS[pos['i']], pd.Timestamp(date)
			if ts not in price_data[tkr].index: continue
			price = price_data[tkr].loc[ts]
			pnl = (price - pos['entry']) / pos['entry']
			if bool(pos['short']): pnl *= -1
			days = (ts - pos['date']).days
			if pnl >= TP or pnl <= -SL or days >= MAX_HOLD:
				logging.info(f"✔️ Closed {tkr} @ {ts.date()}: PnL={pnl:.4f}, held={days}d")
				balance *= (1 + pnl); closed += 1
			else: still_open.append(pos)
		open_pos = still_open

		for i, tkr in enumerate(TICKERS):
			ts = pd.Timestamp(date)
			if ts not in price_data[tkr].index: continue
			price = price_data[tkr].loc[ts]
			signal = float(
				short_w[i] * (price - short_ma[tkr].get(ts, price)) * 100 +
				long_w[i] * (price - long_ma[tkr].get(ts, price)) * 100 +
				vol5_w[i] * vol5[tkr].get(ts, 0.0) * 10 +
				vol_spike_w[i] * vol_spike[tkr].get(ts, 0.0) * 10 +
				sent_w[i] * sentiment_data[tkr].get(ts, 0.0) +
				geo_w[i] * geo_data[tkr].get(ts, 0.0) + B)
			if any(p['i'] == i for p in open_pos): continue
			open_pos.append({"i": i, "entry": price, "date": ts, "short": signal < 0})
			opened += 1
			logging.info(f"📥 Opened {tkr} @ {ts.date()} | Signal: {signal:.4f}")
		eq.append(balance)

	if not eq or balance == 1000.0: return np.array([1000.0, 1000.0])
	return np.array(eq)

def evaluate(genome):
	all_returns = []
	for _, _, test_start, test_end in splits:
		eq = simulate_equity(genome, test_start, test_end)
		ret = np.diff(eq) / eq[:-1]
		if len(ret) < 2: continue
		all_returns.extend(ret)
	if not all_returns: return -np.inf
	return np.mean(all_returns) / (np.std(all_returns) + 1e-6) * 100

def tournament(pop):
	best_genome = max(random.sample(pop, TOURNAMENT), key=lambda x: x[1])[0]
	return np.array(best_genome, dtype=np.float32)

def mutate(g): [g.__setitem__(i, float(np.clip(g[i] + random.gauss(0, SIGMA), -1, 1))) for i in range(GENOME_LENGTH) if random.random() < MUTATE]
def crossover(p1, p2): pt = random.randint(1, GENOME_LENGTH - 1); return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])

def parallel_eval(genomes):
	with ThreadPoolExecutor(max_workers=int(multiprocessing.cpu_count() * 0.8)) as ex:
		futures = [ex.submit(evaluate, g) for g in genomes]
		return [(g, f.result()) for g, f in zip(genomes, tqdm(futures, desc="Evaluating", ncols=80))]

def run():
	pop = [np.random.uniform(-1, 1, GENOME_LENGTH) for _ in range(POP_SIZE)]
	pop = parallel_eval(pop); pop.sort(key=lambda x: x[1], reverse=True)
	print(f"Initial best: {pop[0][1]:.2f}")

	for gen in range(1, GENS + 1):
		new = [pop[0], pop[1]]
		while len(new) < POP_SIZE:
			p1, p2 = tournament(pop), tournament(pop)
			c1, c2 = crossover(p1, p2) if random.random() < CROSSOVER else (np.copy(p1), np.copy(p2))
			mutate(c1); mutate(c2); new.append(c1)
			if len(new) < POP_SIZE: new.append(c2)
		pop = parallel_eval(new); pop.sort(key=lambda x: x[1], reverse=True)
		best, avg = pop[0][1], np.mean([f for _, f in pop])
		print(f"Gen {gen:02d} → best: {best:.2f}, avg: {avg:.2f}")
		logging.info(f"Gen {gen:02d} → best: {best:.2f}, avg: {avg:.2f}")
	logging.shutdown(); return pop[0]

if __name__ == "__main__":
	random.seed(42); np.random.seed(42)
	load_data()
	best, fitness = run()
	print("\n🏆 Best genome fitness:", fitness)
	print("Slice:", best[:10])
