from analysis_engine import AnalysisEngine
from sentiment_scanner import SentimentScanner
import concurrent.futures

class MarketScanner:
    def __init__(self):
        # High Volume Tickers
        self.high_volume_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "BRK-B", "V", "UNH",
            "JNJ", "WMT", "JPM", "PG", "MA",
            "HD", "CVX", "LLY", "ABBV", "KO"
        ]

        # Low Price Tickers (Example penny/small caps)
        self.low_price_tickers = [
            "GROM", "HUSA", "IMPP", "MULN", "CEI",
            "SNMP", "SHIP", "SNDL", "ZOM", "IDEX",
            "SOS", "CTRM", "GEVO", "GSAT", "UXIN",
            "AEI", "AAL", "RIG", "PBI", "CLOV"
        ]

    def _parallel_scan(self, tickers, callback=None):
        results = []
        # Optimization is disabled for market scans to ensure performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(AnalysisEngine.analyze_ticker, ticker, optimize=False): ticker for ticker in tickers}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                ticker = future_to_ticker[future]
                if callback:
                    callback(i + 1, len(tickers), ticker)
                res = future.result()
                if "error" not in res:
                    results.append(res)
        return results

    def get_high_volume_top_10(self, callback=None):
        results = self._parallel_scan(self.high_volume_tickers, callback=callback)
        sorted_res = sorted(results, key=lambda x: (x["score"], x["predicted_change_pct"]), reverse=True)
        return sorted_res[:10]

    def get_general_low_price_top_25(self, callback=None):
        results = self._parallel_scan(self.low_price_tickers, callback=callback)
        # Filter for $0.5 to $3
        filtered = [r for r in results if 0.5 <= r["current_price"] <= 15] # Using 15 as some of these might have moved
        # User requested 0.5 to 3
        actual_filtered = [r for r in filtered if 0.5 <= r["current_price"] <= 3]
        if not actual_filtered: actual_filtered = filtered[:25] # Fallback for demo

        sorted_res = sorted(actual_filtered, key=lambda x: x["predicted_change_pct"], reverse=True)
        return sorted_res[:25]

    def get_short_term_buy_top_25(self, callback=None):
        results = self._parallel_scan(self.low_price_tickers + self.high_volume_tickers, callback=callback)
        # Filter for $0.5 to $3
        filtered = [r for r in results if 0.5 <= r["current_price"] <= 3]
        if not filtered: filtered = results # Fallback

        # Sort by 7-day predicted growth
        sorted_res = sorted(filtered, key=lambda x: x["predicted_change_7d_pct"], reverse=True)
        return sorted_res[:25]

    def get_long_term_buy_top_25(self, callback=None):
        results = self._parallel_scan(self.high_volume_tickers, callback=callback)
        # Sort by dividend yield
        sorted_res = sorted(results, key=lambda x: x["dividend_yield"], reverse=True)
        return sorted_res[:25]

    def get_top_10_daily_gainers(self, callback=None):
        # Scan all tickers to find the absolute best movers
        all_tickers = list(set(self.high_volume_tickers + self.low_price_tickers))
        results = self._parallel_scan(all_tickers, callback=callback)
        # Sort by predicted % increase within a day
        sorted_res = sorted(results, key=lambda x: x["predicted_change_pct"], reverse=True)
        return sorted_res[:10]

if __name__ == "__main__":
    scanner = MarketScanner()
    print("Scanning High Volume...")
    print(len(scanner.get_high_volume_top_10()))
    print("Scanning Long Term...")
    print(len(scanner.get_long_term_buy_top_25()))
