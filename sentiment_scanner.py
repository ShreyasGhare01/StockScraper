import requests
from bs4 import BeautifulSoup
import re
import yfinance as yf
from pyfinviz.quote import Quote

class SentimentScanner:
    def __init__(self):
        self.pump_keywords = [
            'pump', 'moon', 'breakout', 'short squeeze', 'gamma squeeze',
            'buy now', 'rocket', 'to the moon', 'explosive', 'undervalued',
            'soar', 'surge', 'rally'
        ]
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scan_news(self, ticker):
        """
        Scans both Google News RSS and yfinance for the latest ticker news.
        """
        all_headlines = []
        mentions = 0

        # 1. Google News RSS (Recent Focus)
        google_url = f"https://news.google.com/rss/search?q={ticker}+stock+outlook+analysis+when:2d"
        try:
            response = requests.get(google_url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all('item')
            for item in items:
                title = item.title.text
                all_headlines.append({"source": "Google", "title": title})
                if any(kw in title.lower() for kw in self.pump_keywords):
                    mentions += 1
        except:
            pass

        # 2. yfinance News (Real-time Focus)
        try:
            yf_ticker = yf.Ticker(ticker)
            yf_news = yf_ticker.news
            for n in yf_news[:10]:
                title = n['content']['title']
                all_headlines.append({"source": "yfinance", "title": title})
                if any(kw in title.lower() for kw in self.pump_keywords):
                    mentions += 1
        except:
            pass

        # 3. pyfinviz News (Market Professional Focus)
        try:
            quote = Quote(ticker)
            news_df = quote.outer_news_df
            if not news_df.empty:
                for _, row in news_df.head(10).iterrows():
                    title = row['News']
                    all_headlines.append({"source": "Finviz", "title": title})
                    if any(kw in title.lower() for kw in self.pump_keywords):
                        mentions += 1
        except:
            pass

        sentiment_score = mentions / len(all_headlines) if all_headlines else 0

        return {
            "ticker": ticker,
            "pump_mentions": mentions,
            "sentiment_score": sentiment_score,
            "latest_headlines": [h['title'] for h in all_headlines[:10]]
        }

    def get_pump_score(self, ticker):
        results = self.scan_news(ticker)
        if "error" in results or not results.get("latest_headlines"):
            return 0.0

        score = min(results["sentiment_score"] * 5, 1.0)
        return score

if __name__ == "__main__":
    scanner = SentimentScanner()
    print(f"Scanning for NVDA: {scanner.scan_news('NVDA')}")
