from analysis_engine import AnalysisEngine
import concurrent.futures

class MarketScanner:
    def __init__(self, tickers=None):
        if tickers is None:
            # Default top stocks
            self.tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "TSLA", "META", "BRK-B", "V", "UNH",
                "JNJ", "WMT", "JPM", "PG", "MA",
                "HD", "CVX", "LLY", "ABBV", "KO"
            ]
        else:
            self.tickers = tickers

    def scan_top_buys(self, limit=10):
        results = []

        # Use concurrent.futures to speed up scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(AnalysisEngine.analyze_ticker, ticker): ticker for ticker in self.tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if "error" not in res:
                    results.append(res)

        # Sort by score (descending) and then by predicted change pct (descending)
        sorted_results = sorted(results, key=lambda x: (x["score"], x["predicted_change_pct"]), reverse=True)

        return sorted_results[:limit]

if __name__ == "__main__":
    scanner = MarketScanner(tickers=["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"])
    top_10 = scanner.scan_top_buys(limit=10)
    print(f"{'Ticker':<10} | {'Score':<5} | {'Rec':<15} | {'Prediction':<10} | {'Change %':<10}")
    print("-" * 65)
    for res in top_10:
        print(f"{res['ticker']:<10} | {res['score']:<5} | {res['recommendation']:<15} | {res['predicted_price']:<10.2f} | {res['predicted_change_pct']:<10.2f}%")
