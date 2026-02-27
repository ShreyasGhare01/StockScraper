import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
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
        Example: FEDFUNDS (Interest Rate), DGS10 (10-Year Treasury Yield)
        """
        try:
            # Shift start date slightly back to ensure we have enough data to forward fill
            shifted_start = start_date - timedelta(days=10)
            macro_data = web.DataReader(['FEDFUNDS', 'DGS10'], 'fred', shifted_start, end_date)
            # Daily stock data needs daily macro data. FRED data might be monthly (FEDFUNDS) or daily (DGS10).
            # We forward fill to align with stock trading days.
            macro_data = macro_data.resample('D').ffill()
            return macro_data
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test fetcher
    fetcher = StockDataFetcher("AAPL")
    try:
        data = fetcher.fetch_history(period="1mo")
        print(f"Successfully fetched {len(data)} rows for AAPL")
        print(data.tail())
    except Exception as e:
        print(f"Error: {e}")
