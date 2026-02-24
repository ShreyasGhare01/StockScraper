import subprocess

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    result = subprocess.run(["python3", "main.py", ticker], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {ticker} analyzed successfully.")
        # print(result.stdout)
        return True
    else:
        print(f"FAILED: {ticker} analysis failed.")
        print(result.stderr)
        return False

if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "AMZN", "NFLX"]
    results = [test_ticker(t) for t in tickers]

    if all(results):
        print("\nAll integration tests passed!")
    else:
        print("\nSome tests failed.")
        exit(1)
