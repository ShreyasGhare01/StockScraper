from analysis_engine import AnalysisEngine
import pandas as pd

def test_intraday_target():
    print("Testing Intraday Target for NVDA...")
    # Intraday target should have >70% accuracy on some historical data if features are good
    res = AnalysisEngine.analyze_ticker("NVDA", period="2y", mode="edge", target_override="intraday_edge")

    if "error" in res:
        print(f"Error: {res['error']}")
        return

    print(f"Ticker: {res['ticker']}")
    print(f"Intraday Prob (>1.5%): {res['pred_overnight']*100:.1f}%")
    print(f"Accuracy Score: {res['acc_overnight']*100:.1f}%")

    if res['acc_overnight'] >= 0.70:
        print("SUCCESS: Accuracy target reached (>= 70%)")
    else:
        print(f"WARNING: Accuracy {res['acc_overnight']*100:.1f}% is below target 70%.")

if __name__ == "__main__":
    test_intraday_target()
