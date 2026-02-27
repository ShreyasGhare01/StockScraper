import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.dropna()
        # Using XGBoost for potentially higher accuracy
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = []

    def _prepare_data(self):
        # Select features: RSI, MACD, SMA, EMA, BB, ATR, Macro, Fundamentals, etc.
        # Exclude basic OHLCV and targets from features to avoid data leakage.
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target_7d']
        self.feature_cols = [c for c in self.data.columns if c not in exclude]

        X = self.data[self.feature_cols]
        # Target is the next day's Close price
        y = self.data['Close'].shift(-1)

        # Remove the last row as it won't have a target
        X = X[:-1]
        y = y[:-1]

        return X, y

    def train(self):
        X, y = self._prepare_data()
        if len(X) < 50:
            raise ValueError("Not enough data to train the model")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_next_day(self):
        # Use the very last available data point to predict tomorrow
        last_row = self.data[self.feature_cols].tail(1)
        last_row_scaled = self.scaler.transform(last_row)
        prediction = self.model.predict(last_row_scaled)
        return prediction[0]

if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    from technical_analysis import TechnicalAnalyzer

    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_history(period="2y")
    analyzer = TechnicalAnalyzer(data)
    data_with_indicators = analyzer.add_all_indicators()

    predictor = StockPredictor(data_with_indicators)
    predictor.train()
    prediction = predictor.predict_next_day()

    current_price = data['Close'].iloc[-1]
    print(f"Current Price: {current_price:.2f}")
    print(f"Predicted Next Day Price: {prediction:.2f}")
    print(f"Expected Movement: {((prediction - current_price) / current_price) * 100:.2f}%")
