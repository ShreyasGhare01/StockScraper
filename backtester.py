import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data, predictions, initial_capital=10000.0, transaction_cost=0.002, stop_loss=0.04):
        """
        data: DataFrame with technical indicators and price
        predictions: Series or array of predicted probabilities/classes aligned with data
        """
        self.data = data
        self.predictions = predictions
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss

    def run(self):
        """
        Simulates a rolling walk-forward backtest.
        For simplicity in this prototype, we assume we enter at the Close of day T
        if prediction for T+1 is positive (Edge > threshold) and exit at Close of T+1.
        """
        df = self.data.copy()
        # Align predictions (assuming they are for the 'next day')
        df['pred_prob'] = self.predictions

        # Strategy: Buy if prob > 0.55
        df['signal'] = (df['pred_prob'] > 0.55).astype(int)

        # We trade 'tomorrow' based on 'today' signal
        df['next_day_ret'] = df['Close'].pct_change().shift(-1)

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []

        for i in range(len(df) - 1):
            if df['signal'].iloc[i] == 1:
                # Enter Trade
                entry_price = df['Close'].iloc[i]
                # Apply slippage/cost on entry
                entry_cost = entry_price * self.transaction_cost

                raw_ret = df['next_day_ret'].iloc[i]

                # Check Stop Loss (Intraday simulation proxy)
                # If Low of next day hits stop loss relative to entry
                next_low = df['Low'].iloc[i+1]
                if (next_low - entry_price) / entry_price < -self.stop_loss:
                    final_ret = -self.stop_loss
                else:
                    final_ret = raw_ret

                # Apply transaction cost on exit
                exit_cost_pct = self.transaction_cost
                net_ret = final_ret - self.transaction_cost - (entry_cost / entry_price)

                capital *= (1 + net_ret)
                trades.append(net_ret)

            equity_curve.append(capital)

        # Metrics
        returns = pd.Series(trades)
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        total_return = (capital - self.initial_capital) / self.initial_capital

        # Annualized Sharpe (Assuming 252 trading days)
        if len(returns) > 5:
            avg_ret = returns.mean()
            std_ret = returns.std()
            sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
        else:
            sharpe = 0

        # Drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Profit Factor
        pos_rets = returns[returns > 0].sum()
        neg_rets = abs(returns[returns < 0].sum())
        profit_factor = pos_rets / neg_rets if neg_rets != 0 else pos_rets

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "equity_curve": equity_curve,
            "num_trades": len(trades)
        }
