import subprocess

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    result = subprocess.run(["python3", "main.py", ticker], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {ticker} analyzed successfully.")
        return True
    else:
        print(f"FAILED: {ticker} analysis failed.")
        print(result.stderr)
        return False

if __name__ == "__main__":
    tickers = ["AAPL"] # Reduced for speed
    results = [test_ticker(t) for t in tickers]

    if all(results):
        print("\nIntegration test passed!")
    else:
        print("\nTest failed.")
        exit(1)
