import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import requests
from typing import Optional
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def fetch_history(self, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical data for the ticker.
        Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        df = self.stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        return df

    def get_info(self) -> dict:
        """Returns general stock information."""
        return self.stock.info

    @staticmethod
    def fetch_macro_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches macroeconomic data from FRED.
        Example: FEDFUNDS (Interest Rate), DGS10 (10-Year Treasury Yield), VIXCLS (VIX Sentiment)
        """
        try:
            # Shift start date slightly back to ensure we have enough data to forward fill
            shifted_start = start_date - timedelta(days=10)
            # FEDFUNDS: Interest Rate, DGS10: 10-Yr Yield, VIXCLS: Market Volatility Index (Sentiment)
            macro_data = web.DataReader(['FEDFUNDS', 'DGS10', 'VIXCLS'], 'fred', shifted_start, end_date)
            # Daily stock data needs daily macro data.
            # We forward fill to align with stock trading days.
            macro_data = macro_data.resample('D').ffill()
            return macro_data
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return pd.DataFrame()

    def get_fundamental_data(self) -> dict:
        """
        Retrieves key fundamental metrics for the stock.
        """
        info = self.get_info()
        return {
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE") or 0,
            "eps": info.get("trailingEps") or 0,
            "debt_to_equity": info.get("debtToEquity") or 0,
            "price_to_book": info.get("priceToBook") or 0,
            "profit_margins": info.get("profitMargins") or 0
        }

    def fetch_alpha_vantage_sentiment(self, api_key: str = "demo") -> float:
        """
        Fetches average news sentiment score from AlphaVantage.
        Returns a score between -1 and 1.
        """
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&apikey={api_key}"
        try:
            r = requests.get(url)
            data = r.json()
            feed = data.get("feed", [])
            if not feed:
                return 0.0

            # Average the sentiment scores
            scores = [float(item.get("overall_sentiment_score", 0)) for item in feed]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception as e:
            print(f"AlphaVantage Sentiment Error: {e}")
            return 0.0

if __name__ == "__main__":
    # Test fetcher
    fetcher = StockDataFetcher("AAPL")
    try:
        data = fetcher.fetch_history(period="1mo")
        print(f"Successfully fetched {len(data)} rows for AAPL")
        print(data.tail())
    except Exception as e:
        print(f"Error: {e}")
