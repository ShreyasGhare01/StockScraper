import requests
from bs4 import BeautifulSoup
import re

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
        # Using a more standard Google News RSS
        url = f"https://news.google.com/rss/search?q={ticker}+stock+pump+research"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all('item')

            mentions = 0
            titles = []
            for item in items:
                title = item.title.text.lower()
                titles.append(title)
                for kw in self.pump_keywords:
                    if kw in title:
                        mentions += 1
                        break

            sentiment_score = mentions / len(items) if items else 0

            return {
                "ticker": ticker,
                "pump_mentions": mentions,
                "sentiment_score": sentiment_score,
                "latest_headlines": [i.title.text for i in items[:5]]
            }
        except Exception as e:
            return {"error": str(e)}

    def get_pump_score(self, ticker):
        results = self.scan_news(ticker)
        if "error" in results or not results.get("latest_headlines"):
            return 0.0

        score = min(results["sentiment_score"] * 5, 1.0)
        return score

if __name__ == "__main__":
    scanner = SentimentScanner()
    print(f"Scanning for NVDA: {scanner.scan_news('NVDA')}")
