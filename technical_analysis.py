import pandas as pd
import pandas_ta as ta

class TechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def add_all_indicators(self) -> pd.DataFrame:
        """Calculates and adds common technical indicators to the dataframe."""
        # RSI
        self.data.ta.rsi(length=14, append=True)

        # MACD
        self.data.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Moving Averages
        self.data.ta.sma(length=20, append=True)
        self.data.ta.sma(length=50, append=True)
        self.data.ta.ema(length=200, append=True)

        # Bollinger Bands
        self.data.ta.bbands(length=20, std=2, append=True)

        # Average True Range (Volatility)
        self.data.ta.atr(length=14, append=True)

        return self.data

    def get_signals(self) -> dict:
        """Determines simple buy/sell signals based on indicators."""
        if len(self.data) < 200:
            return {"error": "Not enough data for full analysis (minimum 200 days recommended)"}

        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]

        signals = []
        score = 0

        # Column names can vary slightly by pandas_ta version
        rsi_col = [c for c in self.data.columns if c.startswith('RSI')][0]
        macd_col = [c for c in self.data.columns if c.startswith('MACD_') and 's' not in c and 'h' not in c][0]
        macds_col = [c for c in self.data.columns if c.startswith('MACDs_')][0]
        ema200_col = [c for c in self.data.columns if c.startswith('EMA_200')][0]
        bbl_col = [c for c in self.data.columns if c.startswith('BBL_')][0]
        bbu_col = [c for c in self.data.columns if c.startswith('BBU_')][0]

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
