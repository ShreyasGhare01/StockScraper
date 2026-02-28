import pandas as pd
import numpy as np

class OptionStrategist:
    def __init__(self, ticker: str, current_price: float, analysis: dict, data: pd.DataFrame):
        self.ticker = ticker
        self.current_price = current_price
        self.analysis = analysis
        self.data = data

    def suggest_trades(self):
        recommendation = self.analysis.get("recommendation", "Neutral")
        # Ensure we use correct keys from unified results
        prob_overnight = self.analysis.get("pred_overnight", 0.5)
        prob_5day = self.analysis.get("pred_5day", 0.5)

        # Guard against missing ATR
        atr_cols = [c for c in self.data.columns if c.startswith('ATRr_')]
        if not atr_cols:
            return "No clear option trade suggested: Technical data (ATR) missing."
        atr_col = atr_cols[0]
        latest_atr = self.data[atr_col].iloc[-1]

        trades = []

        # Bullish Strategy based on Edge Probabilities
        if recommendation in ["Buy", "Strong Buy"] and (prob_overnight > 0.55 or prob_5day > 0.55):
            strike_price = round(self.current_price)
            # High confidence = Aggressive
            if (prob_overnight > 0.7 or prob_5day > 0.7):
                strategy_type = "Aggressive Call"
                strike_price = round(self.current_price + latest_atr) # OTM
                reason = f"High confidence ({max(prob_overnight, prob_5day)*100:.1f}%) in edge move."
            else:
                strategy_type = "Conservative Call"
                strike_price = round(self.current_price) # ATM
                reason = f"Moderate confidence ({max(prob_overnight, prob_5day)*100:.1f}%) in edge move."

            trades.append({
                "type": "CALL",
                "strategy": strategy_type,
                "strike": strike_price,
                "expiry": "14-30 days",
                "reason": reason + f" (ATR: {latest_atr:.2f})"
            })

        # Bearish Strategy (If edge probs are low, could imply bearish or neutral)
        # Note: Our current model only predicts UP edges. In a real system, we'd have DOWN edges too.
        # For now, let's stick to Buy/Strong Buy based on the provided request to show "Strong Buys".

        elif recommendation in ["Sell", "Strong Sell"]:
            trades.append({
                "type": "PUT",
                "strategy": "Protective Put / Sell",
                "strike": round(self.current_price),
                "expiry": "14-30 days",
                "reason": "Bearish technical indicators suggest hedging or exiting."
            })

        # Volatility Strategy (Straddle/Strangle if high ATR but uncertain direction)
        # For simplicity, let's just do Calls/Puts for now as per "Strong Buy" requirement.

        if not trades:
            return "No clear option trade suggested based on current analysis."

        return trades

if __name__ == "__main__":
    # Test with dummy data
    dummy_analysis = {"recommendation": "Strong Buy", "score": 4, "prob_overnight": 0.75, "prob_5day": 0.6}
    strategist = OptionStrategist("AAPL", 150.0, dummy_analysis, pd.DataFrame({'ATRr_14': [2.0]}))
    print(strategist.suggest_trades())
