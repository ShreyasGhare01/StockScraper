import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def _manual_rsi(self, series, length=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _manual_macd(self, series, fast=12, slow=26, signal=9):
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_hist, macd_signal

    def _manual_sma(self, series, length):
        return series.rolling(window=length).mean()

    def _manual_ema(self, series, length):
        return series.ewm(span=length, adjust=False).mean()

    def _manual_bbands(self, series, length=20, std=2):
        sma = series.rolling(window=length).mean()
        std_dev = series.rolling(window=length).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return lower, sma, upper

    def _manual_atr(self, high, low, close, length=14):
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    def _manual_obv(self, close, volume):
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()

    def _manual_roc(self, close, length=12):
        return ((close - close.shift(length)) / close.shift(length)) * 100

    def _manual_zscore(self, series, window=20):
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std

    def add_all_indicators(self) -> pd.DataFrame:
        """Calculates and adds common technical indicators to the dataframe."""
        # Manual RSI
        self.data['RSI_14'] = self._manual_rsi(self.data['Close'], length=14)

        # Manual MACD
        macd, macd_hist, macd_signal = self._manual_macd(self.data['Close'])
        self.data['MACD_12_26_9'] = macd
        self.data['MACDh_12_26_9'] = macd_hist
        self.data['MACDs_12_26_9'] = macd_signal

        # Manual Moving Averages
        self.data['SMA_20'] = self._manual_sma(self.data['Close'], 20)
        self.data['SMA_50'] = self._manual_sma(self.data['Close'], 50)
        self.data['EMA_200'] = self._manual_ema(self.data['Close'], 200)

        # Manual Bollinger Bands
        lower, mid, upper = self._manual_bbands(self.data['Close'], length=20, std=2)
        self.data['BBL_20_2.0'] = lower
        self.data['BBM_20_2.0'] = mid
        self.data['BBU_20_2.0'] = upper

        # Manual ATR
        self.data['ATRr_14'] = self._manual_atr(self.data['High'], self.data['Low'], self.data['Close'], length=14)

        # New Predictive Indicators for ML
        self.data['OBV'] = self._manual_obv(self.data['Close'], self.data['Volume'])
        self.data['ROC_12'] = self._manual_roc(self.data['Close'], length=12)
        self.data['Volatility_20'] = self.data['Close'].rolling(window=20).std()

        # --- Restructuring Step 2 Features ---

        # 1. End-of-Day Momentum (Daily equivalents)
        # Close position in daily range: (Close - Low) / (High - Low)
        self.data['Close_Pos_Range'] = (self.data['Close'] - self.data['Low']) / (self.data['High'] - self.data['Low'] + 1e-9)
        # Gap % from previous open (as requested)
        self.data['Gap_Prev_Open'] = (self.data['Open'] - self.data['Open'].shift(1)) / (self.data['Open'].shift(1) + 1e-9)
        # RSI Slope (today - yesterday)
        self.data['RSI_Slope'] = self.data['RSI_14'].diff()
        # Distance from 20 EMA (normalized)
        ema20 = self._manual_ema(self.data['Close'], 20)
        self.data['EMA20_Dist'] = (self.data['Close'] - ema20) / (ema20 + 1e-9)

        # 2. Volume Acceleration
        # Volume / 20-day average
        vol_avg20 = self.data['Volume'].rolling(window=20).mean()
        self.data['Vol_Rel_Avg'] = self.data['Volume'] / (vol_avg20 + 1e-9)
        # 3-day volume growth rate
        self.data['Vol_Growth_3d'] = self.data['Volume'].pct_change(3)
        # Volume Z-score
        self.data['Vol_ZScore'] = self._manual_zscore(self.data['Volume'], window=20)
        # Volume Acceleration (today - yesterday)
        self.data['Vol_Acceleration'] = self.data['Volume'].diff()
        # Interaction term: Volume Z-score * Price Momentum (ROC_12)
        self.data['Vol_Price_Interaction'] = self.data['Vol_ZScore'] * self.data['ROC_12']

        return self.data

    def get_signals(self) -> dict:
        """Determines simple buy/sell signals based on indicators."""
        if len(self.data) < 200:
            return {"error": "Not enough data for full analysis (minimum 200 days recommended)"}

        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]

        signals = []
        score = 0

        rsi_col = 'RSI_14'
        macd_col = 'MACD_12_26_9'
        macds_col = 'MACDs_12_26_9'
        ema200_col = 'EMA_200'
        bbl_col = 'BBL_20_2.0'
        bbu_col = 'BBU_20_2.0'

        # RSI analysis
        if latest[rsi_col] < 30:
            signals.append("Oversold (RSI < 30)")
            score += 2
        elif latest[rsi_col] > 70:
            signals.append("Overbought (RSI > 70)")
            score -= 2

        # MACD Crossover
        if prev[macd_col] < prev[macds_col] and latest[macd_col] > latest[macds_col]:
            signals.append("MACD Bullish Crossover")
            score += 2
        elif prev[macd_col] > prev[macds_col] and latest[macd_col] < latest[macds_col]:
            signals.append("MACD Bearish Crossover")
            score -= 2

        # Price vs EMA200
        if latest['Close'] > latest[ema200_col]:
            signals.append("Price above EMA 200 (Long-term Bullish)")
            score += 1
        else:
            signals.append("Price below EMA 200 (Long-term Bearish)")
            score -= 1

        # Bollinger Bands
        if latest['Close'] < latest[bbl_col]:
            signals.append("Price below Lower Bollinger Band")
            score += 1
        elif latest['Close'] > latest[bbu_col]:
            signals.append("Price above Upper Bollinger Band")
            score -= 1

        recommendation = "Neutral"
        if score >= 3:
            recommendation = "Strong Buy"
        elif score > 0:
            recommendation = "Buy"
        elif score <= -3:
            recommendation = "Strong Sell"
        elif score < 0:
            recommendation = "Sell"

        return {
            "score": score,
            "recommendation": recommendation,
            "signals": signals,
            "latest_values": {
                "RSI": latest[rsi_col],
                "MACD": latest[macd_col],
                "EMA200": latest[ema200_col],
                "Close": latest['Close']
            }
        }

if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_history(period="2y")
    analyzer = TechnicalAnalyzer(data)
    analyzed_data = analyzer.add_all_indicators()
    print("Indicators added.")
    signals = analyzer.get_signals()
    print("Signals:", signals)
