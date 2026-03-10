import yfinance as yf
import pandas as pd
import numpy as np

def check_thresholds(ticker):
    print(f"Checking thresholds for {ticker} (Last 2 Years)...")
    data = yf.Ticker(ticker).history(period="2y")
    if data.empty:
        print("No data.")
        return

    returns = (data['High'].shift(-1) - data['Close']) / data['Close']

    t10 = (returns > 0.10).mean()
    t7 = (returns > 0.07).mean()
    t5 = (returns > 0.05).mean()

    print(f"Days > +10%: {t10*100:.2f}% ({int(t10*len(data))} days)")
    print(f"Days > +7%:  {t7*100:.2f}% ({int(t7*len(data))} days)")
    print(f"Days > +5%:  {t5*100:.2f}% ({int(t5*len(data))} days)")
    print("-" * 30)

if __name__ == "__main__":
    check_thresholds("NVDA")
    check_thresholds("TSLA")
    check_thresholds("MULN")
    check_thresholds("GROM")
    check_thresholds("AAPL")
