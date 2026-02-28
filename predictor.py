import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    def __init__(self, data: pd.DataFrame, target_type="overnight", mode="edge"):
        self.data = data.dropna()
        self.target_type = target_type
        self.mode = mode

        if self.mode == "edge":
            # Using XGBClassifier for binary edge detection
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,
                reg_alpha=0.2,
                reg_lambda=1.5,
                min_child_weight=2,
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42
            )
        else:
            # Using XGBRegressor for price forecasting
            self.model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=1,
                n_jobs=-1,
                random_state=42
            )
        self.scaler = StandardScaler()
        self.feature_cols = []

    def _prepare_data(self):
        # Select features excluding raw price and future targets
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Target_7d', 'Target_Overnight', 'Target_5Day']
        self.feature_cols = [c for c in self.data.columns if c not in exclude]

        X = self.data[self.feature_cols]

        if self.mode == "edge":
            if self.target_type == "overnight":
                # Target 1: Next Day Return > +2.5%
                returns = (self.data['Close'].shift(-1) - self.data['Close']) / self.data['Close']
                y = (returns > 0.025).astype(int)
                lookahead = 1
            else:
                # Target 2: Max Return within next 5 days > +5.0%
                y_series = []
                for i in range(len(self.data)):
                    subset = self.data['Close'].iloc[i+1 : i+6]
                    if len(subset) < 5:
                        y_series.append(np.nan)
                    else:
                        max_ret = (subset.max() - self.data['Close'].iloc[i]) / self.data['Close'].iloc[i]
                        y_series.append(1 if max_ret > 0.05 else 0)
                y = pd.Series(y_series, index=self.data.index)
                lookahead = 5
        else:
            # Regression Mode
            if self.target_type == "overnight":
                y = self.data['Close'].shift(-1)
                lookahead = 1
            else:
                y = self.data['Close'].shift(-5)
                lookahead = 5

        # Remove trailing rows that don't have enough lookahead data
        X = X[:-lookahead]
        y = y[:-lookahead]

        # Drop any remaining NaNs in y if they exist
        mask = y.notna()
        X = X[mask]
        y = y[mask]

        # Only cast to int for classification mode
        if self.mode == "edge":
            y = y.astype(int)

        return X, y

    def tune_hyperparameters(self, X, y):
        """
        Runs RandomizedSearchCV to find the best XGBoost parameters for the specific data.
        """
        param_dist = {
            'n_estimators': [300, 500, 800],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        scoring = 'roc_auc' if self.mode == "edge" else 'neg_mean_squared_error'

        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        return search.best_params_

    def train(self, optimize=False):
        X, y = self._prepare_data()
        if len(X) < 50:
            raise ValueError("Not enough data to train the model")

        X_scaled = self.scaler.fit_transform(X)

        if optimize:
            try:
                self.tune_hyperparameters(X_scaled, y)
            except Exception as e:
                print(f"Hyperparameter tuning failed, using defaults: {e}")
                self.model.fit(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

    def calculate_expected_value(self, probability):
        """
        EV = P(up) * Avg Gain - P(down) * Avg Loss
        """
        if self.target_type == "overnight":
            returns = (self.data['Close'].shift(-1) - self.data['Close']) / self.data['Close']
            threshold = 0.025
        else:
            # Approx for 5d
            returns = (self.data['Close'].shift(-5) - self.data['Close']) / self.data['Close']
            threshold = 0.05

        wins = returns[returns > threshold]
        losses = returns[returns < -threshold]

        avg_gain = wins.mean() if not wins.empty else threshold
        avg_loss = abs(losses.mean()) if not losses.empty else threshold

        p_up = probability
        p_down = 1 - p_up

        ev = (p_up * avg_gain) - (p_down * avg_loss)
        return ev, avg_gain, avg_loss

    def predict_probability(self):
        # Use the very last available data point to predict probability of target
        last_row = self.data[self.feature_cols].tail(1)
        last_row_scaled = self.scaler.transform(last_row)
        if self.mode == "edge":
            # Returns probability of class 1
            probs = self.model.predict_proba(last_row_scaled)[0]
            prob = float(probs[1])
            ev, _, _ = self.calculate_expected_value(prob)
            return prob, ev
        else:
            # For regression, we return the raw predicted value
            prediction = self.model.predict(last_row_scaled)
            return float(prediction[0]), 0.0

if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    from technical_analysis import TechnicalAnalyzer

    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_history(period="2y")
    analyzer = TechnicalAnalyzer(data)
    data_with_indicators = analyzer.add_all_indicators()

    predictor = StockPredictor(data_with_indicators, target_type="overnight")
    predictor.train()
    prob, ev = predictor.predict_probability()

    print(f"Ticker: AAPL")
    print(f"Overnight Edge Probability (>2.5%): {prob*100:.1f}%")
    print(f"Expected Value: {ev*100:.2f}%")
