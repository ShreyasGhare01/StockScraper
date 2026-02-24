import pandas as pd
import numpy as np

class OptionStrategist:
    def __init__(self, ticker: str, current_price: float, prediction: float, analysis: dict, data: pd.DataFrame):
        self.ticker = ticker
        self.current_price = current_price
        self.prediction = prediction
        self.analysis = analysis
        self.data = data

    def suggest_trades(self):
        recommendation = self.analysis.get("recommendation", "Neutral")
        predicted_change_pct = ((self.prediction - self.current_price) / self.current_price) * 100

        atr_col = [c for c in self.data.columns if c.startswith('ATRr_')][0]
        latest_atr = self.data[atr_col].iloc[-1]

        trades = []

        # Bullish Strategy
        if recommendation in ["Buy", "Strong Buy"] and predicted_change_pct > 0.5:
            # Check if predicted move is significant relative to volatility (ATR)
            is_significant = (self.prediction - self.current_price) > (latest_atr * 0.5)

            strike_price = round(self.current_price)
            if recommendation == "Strong Buy" and is_significant:
                strategy_type = "Aggressive Call"
                strike_price = round(self.current_price + latest_atr) # OTM by 1 ATR
            else:
                strategy_type = "Conservative Call"
                strike_price = round(self.current_price) # ATM

            trades.append({
                "type": "CALL",
                "strategy": strategy_type,
                "strike": strike_price,
                "expiry": "14-30 days",
                "reason": f"Bullish indicators and predicted upward movement of {predicted_change_pct:.2f}% (Volatility ATR: {latest_atr:.2f})"
            })

        # Bearish Strategy
        elif recommendation in ["Sell", "Strong Sell"] and predicted_change_pct < -0.5:
            is_significant = (self.current_price - self.prediction) > (latest_atr * 0.5)

            strike_price = round(self.current_price)
            if recommendation == "Strong Sell" and is_significant:
                strategy_type = "Aggressive Put"
                strike_price = round(self.current_price - latest_atr) # OTM by 1 ATR
            else:
                strategy_type = "Conservative Put"
                strike_price = round(self.current_price) # ATM

            trades.append({
                "type": "PUT",
                "strategy": strategy_type,
                "strike": strike_price,
                "expiry": "14-30 days",
                "reason": f"Bearish indicators and predicted downward movement of {predicted_change_pct:.2f}% (Volatility ATR: {latest_atr:.2f})"
            })

        # Volatility Strategy (Straddle/Strangle if high ATR but uncertain direction)
        # For simplicity, let's just do Calls/Puts for now as per "Strong Buy" requirement.

        if not trades:
            return "No clear option trade suggested based on current analysis."

        return trades

if __name__ == "__main__":
    # Test with dummy data
    dummy_analysis = {"recommendation": "Strong Buy", "score": 4}
    strategist = OptionStrategist("AAPL", 150.0, 155.0, dummy_analysis, pd.DataFrame({'ATRr_14': [2.0]}))
    print(strategist.suggest_trades())
