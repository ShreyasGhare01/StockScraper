import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class PumpDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = []

    def _prepare_features(self):
        df = self.data.copy()

        # Calculate Z-scores for Volume and Price changes
        df['Vol_Mean'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Std'] = df['Volume'].rolling(window=20).std()
        df['Vol_Z'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']

        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Mean'] = df['Price_Change'].rolling(window=20).mean()
        df['Price_Change_Std'] = df['Price_Change'].rolling(window=20).std()
        df['Price_Z'] = (df['Price_Change'] - df['Price_Change_Mean']) / df['Price_Change_Std']

        # Volatility
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low']

        self.features = ['Vol_Z', 'Price_Z', 'High_Low_Pct']
        return df.dropna()

    def _label_data(self, df):
        # Label as 'Pump' (1) if Volume Z-score > 2 and Price Change > 5%
        # These thresholds can be adjusted
        df['Is_Pump'] = 0
        df.loc[(df['Vol_Z'] > 2) & (df['Price_Change'] > 0.05), 'Is_Pump'] = 1
        return df

    def train(self):
        df = self._prepare_features()
        df = self._label_data(df)

        X = df[self.features]
        y = df['Is_Pump']

        if len(X[y == 1]) < 2:
            # Fallback if not enough pump examples: use lower threshold for training data
             df.loc[(df['Vol_Z'] > 1.5) & (df['Price_Change'] > 0.03), 'Is_Pump'] = 1
             y = df['Is_Pump']

        if len(X) < 10:
            return False

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return True

    def detect_current(self):
        df = self._prepare_features()
        last_row = df[self.features].tail(1)

        if last_row.empty:
            return {"pump_probability": 0.0, "is_pump": False, "reason": "Insufficient data"}

        last_row_scaled = self.scaler.transform(last_row)
        prob = self.model.predict_proba(last_row_scaled)[0][1]

        is_pump = prob > 0.5

        # Descriptive reason
        vol_z = df['Vol_Z'].iloc[-1]
        price_z = df['Price_Z'].iloc[-1]

        reasons = []
        if vol_z > 2: reasons.append(f"Unusual volume spike (Z={vol_z:.2f})")
        if price_z > 2: reasons.append(f"Sharp price movement (Z={price_z:.2f})")

        return {
            "pump_probability": prob,
            "is_pump": is_pump,
            "reasons": reasons if reasons else ["Normal market activity"],
            "vol_z": vol_z,
            "price_z": price_z
        }

if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    # Use a volatile stock for testing
    fetcher = StockDataFetcher("TSLA")
    data = fetcher.fetch_history(period="2y")
    detector = PumpDetector(data)
    if detector.train():
        result = detector.detect_current()
        print(f"Pump Detection for TSLA: {result}")
    else:
        print("Failed to train detector.")
