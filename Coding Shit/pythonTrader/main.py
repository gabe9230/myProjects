import time
import requests
from datetime import datetime

# === API KEYS ===
NEWS_API_KEY = "6c7f6c17d43c4b0394396cc4f3deb96a"
FINNHUB_API_KEY = "d0tp9v9r01qlvahdj2lgd0tp9v9r01qlvahdj2m0"

# === CONFIG ===
TRADE_DURATION = 10 * 60  # 10 minutes
TP = 0.03  # 3% take profit
SL = 0.02  # 2% stop loss

POSITIVE_WORDS = ["record", "growth", "beats", "profit", "surge", "upgrade", "acquisition", "strong", "rebound"]
NEGATIVE_WORDS = ["crash", "bankruptcy", "plunge", "lawsuit", "default", "layoffs", "scandal", "fraud", "miss"]

# === Runtime State ===
position = None
entry_price = None
entry_time = None
active_ticker = None
is_short = False
balance = 1000
log = []

# === Sentiment classification ===
def get_sentiment(text):
	text = text.lower()
	score = sum(w in text for w in POSITIVE_WORDS) - sum(w in text for w in NEGATIVE_WORDS)
	if score > 0:
		return "positive"
	elif score < 0:
		return "negative"
	return "neutral"

# === Ticker extraction from news ===
def lookup_ticker_from_text(text):
	words = text.lower().split()
	tried = set()

	for word in words:
		if len(word) < 3 or word in tried:
			continue
		tried.add(word)
		try:
			url = f"https://finnhub.io/api/v1/search?q={word}&token={FINNHUB_API_KEY}"
			resp = requests.get(url)
			if resp.status_code == 200:
				results = resp.json().get("result", [])
				for r in results:
					if "Common Stock" in r.get("type", "") and r.get("symbol"):
						return r["symbol"]
		except:
			pass
	return None

# === News and sentiment check ===
def check_news():
	url = f"https://newsapi.org/v2/everything?q=stocks&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
	resp = requests.get(url)
	if resp.status_code != 200:
		print("News API error:", resp.status_code)
		return None, None

	articles = resp.json().get("articles", [])
	for article in articles:
		content = (article.get("title") or "") + " " + (article.get("description") or "")
		sentiment = get_sentiment(content)
		if sentiment in ["positive", "negative"]:
			ticker = lookup_ticker_from_text(content)
			if ticker:
				print(f"📰 {sentiment.upper()} news → {ticker}: {content[:100]}...")
				return sentiment, ticker
	return None, None

# === Live Price from Finnhub ===
def get_price(ticker):
	try:
		url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
		resp = requests.get(url)
		if resp.status_code == 200:
			return float(resp.json().get("c", 0))
	except:
		pass
	return None

# === Trade Simulation ===
def simulate_trade():
	global position, entry_price, entry_time, balance, active_ticker, is_short

	if position:
		price = get_price(active_ticker)
		if not price or price <= 1:
			print(f"⚠️ Price fetch failed for {active_ticker}")
			return

		pnl = (price - entry_price) / entry_price
		if is_short:
			pnl *= -1

		duration = (datetime.now() - entry_time).total_seconds()
		if pnl >= TP or pnl <= -SL or duration >= TRADE_DURATION:
			balance *= (1 + pnl)
			log.append({
				"time": datetime.now().isoformat(),
				"ticker": active_ticker,
				"position": "short" if is_short else "long",
				"entry_price": entry_price,
				"exit_price": price,
				"pnl_pct": round(pnl * 100, 2),
				"balance": round(balance, 2)
			})
			print(f"💰 Exited {active_ticker} ({'SHORT' if is_short else 'LONG'}): PnL={pnl*100:.2f}%, Balance=${balance:.2f}")
			position = None

	else:
		sentiment, ticker = check_news()
		if sentiment and ticker:
			price = get_price(ticker)
			if price and price > 1:
				position = True
				entry_price = price
				entry_time = datetime.now()
				active_ticker = ticker
				is_short = sentiment == "negative"
				print(f"📈 Entered {'SHORT' if is_short else 'LONG'} on {ticker} at ${price:.2f}")
			else:
				print(f"⚠️ Skipped {ticker}, invalid or unavailable price")
		else:
			print("🔍 No actionable news found")

# === Main Loop ===
print("🚀 Starting dynamic news-sentiment trading bot...")
try:
	while True:
		simulate_trade()
		time.sleep(60)
except KeyboardInterrupt:
	print("\n🛑 Bot stopped by user.")
	print("\n📊 Trade log:")
	for l in log:
		print(l)
