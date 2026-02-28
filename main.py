import sys
import argparse
from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from option_strategy import OptionStrategist
from analysis_engine import AnalysisEngine

def main():
    parser = argparse.ArgumentParser(description="Stock Analyzer and Predictor")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g., 2y, 5y)")
    parser.add_argument("--mode", type=str, default="edge", choices=["edge", "regression"], help="Analysis mode")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}...")
    print(f"{'='*50}\n")

    try:
        # 1. Fetch Data
        fetcher = StockDataFetcher(ticker)
        data = fetcher.fetch_history(period=args.period)
        current_price = data['Close'].iloc[-1]

        # 2. Technical Analysis
        analyzer = TechnicalAnalyzer(data)
        data_with_indicators = analyzer.add_all_indicators()
        analysis = analyzer.get_signals()

        # 3. ML Prediction (Edge or Regression)
        res = AnalysisEngine.analyze_ticker(ticker, period=args.period, mode=args.mode)
        if "error" in res:
            raise Exception(res["error"])

        # 4. Option Strategy
        strategist = OptionStrategist(ticker, current_price, res, data_with_indicators)
        trades = strategist.suggest_trades()

        # Display Results
        print(f"Current Price: ${current_price:.2f}")
        if args.mode == "edge":
            print(f"Overnight Edge Prob (>2.5%): {res['pred_overnight']*100:.1f}%")
            print(f"5-Day Edge Prob (>5.0%): {res['pred_5day']*100:.1f}%")
        else:
            print(f"Predicted Tomorrow: ${res['pred_overnight']:.2f}")
            print(f"Predicted 1-Week: ${res['pred_5day']:.2f}")

        print(f"\nTechnical Analysis Recommendation: {analysis['recommendation']}")
        print(f"Score: {analysis['score']}")
        print("Signals:")
        for signal in analysis['signals']:
            print(f"  - {signal}")

        print(f"\n{'='*20} SUGGESTED TRADES {'='*20}")
        if isinstance(trades, list):
            for trade in trades:
                print(f"Type: {trade['type']}")
                print(f"Strategy: {trade['strategy']}")
                print(f"Strike Price: ${trade['strike']}")
                print(f"Recommended Expiry: {trade['expiry']}")
                print(f"Reason: {trade['reason']}")
                print("-" * 30)
        else:
            print(trades)
        print(f"{'='*58}\n")

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
