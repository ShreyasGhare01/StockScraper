import yfinance as yf
import pandas as pd
from typing import Optional

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

if __name__ == "__main__":
    # Test fetcher
    fetcher = StockDataFetcher("AAPL")
    try:
        data = fetcher.fetch_history(period="1mo")
        print(f"Successfully fetched {len(data)} rows for AAPL")
        print(data.tail())
    except Exception as e:
        print(f"Error: {e}")
