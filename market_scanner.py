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

    def _parallel_scan(self, tickers, callback=None, mode="edge"):
        results = []
        # Optimization is disabled for market scans to ensure performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(AnalysisEngine.analyze_ticker, ticker, optimize=False, mode=mode): ticker for ticker in tickers}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
                ticker = future_to_ticker[future]
                if callback:
                    callback(i + 1, len(tickers), ticker)
                res = future.result()
                if "error" not in res:
                    results.append(res)
        return results

    def get_high_volume_top_10(self, callback=None, mode="edge"):
        results = self._parallel_scan(self.high_volume_tickers, callback=callback, mode=mode)
        if mode == "edge":
            # Sort by Overnight Expected Value
            sorted_res = sorted(results, key=lambda x: (x["ev_overnight"], x["score"]), reverse=True)
        else:
            sorted_res = sorted(results, key=lambda x: (x["pred_overnight"] - x["current_price"]), reverse=True)
        return sorted_res[:10]

    def get_general_low_price_top_25(self, callback=None, mode="edge"):
        results = self._parallel_scan(self.low_price_tickers, callback=callback, mode=mode)
        filtered = [r for r in results if 0.5 <= r["current_price"] <= 3]
        if not filtered: filtered = sorted(results, key=lambda x: x["current_price"])[:25]

        if mode == "edge":
            # Sort by 5-day Expected Value
            sorted_res = sorted(filtered, key=lambda x: x["ev_5day"], reverse=True)
        else:
            sorted_res = sorted(filtered, key=lambda x: (x["pred_overnight"] / x["current_price"]), reverse=True)
        return sorted_res[:25]

    def get_short_term_buy_top_25(self, callback=None, mode="edge"):
        results = self._parallel_scan(self.low_price_tickers + self.high_volume_tickers, callback=callback, mode=mode)
        filtered = [r for r in results if 0.5 <= r["current_price"] <= 3]
        if not filtered: filtered = results

        if mode == "edge":
            # Sort by 5-day Expected Value
            sorted_res = sorted(filtered, key=lambda x: x["ev_5day"], reverse=True)
        else:
            sorted_res = sorted(filtered, key=lambda x: (x["pred_5day"] / x["current_price"]), reverse=True)
        return sorted_res[:25]

    def get_long_term_buy_top_25(self, callback=None, mode="edge"):
        results = self._parallel_scan(self.high_volume_tickers, callback=callback, mode=mode)
        # Sort by dividend yield
        sorted_res = sorted(results, key=lambda x: x["dividend_yield"], reverse=True)
        return sorted_res[:25]

    def get_top_10_daily_gainers(self, callback=None, mode="edge"):
        # Scan all tickers to find the absolute best movers
        all_tickers = list(set(self.high_volume_tickers + self.low_price_tickers))
        results = self._parallel_scan(all_tickers, callback=callback, mode=mode)

        if mode == "edge":
            # Sort by highest Overnight Expected Value
            sorted_res = sorted(results, key=lambda x: x["ev_overnight"], reverse=True)
        else:
            sorted_res = sorted(results, key=lambda x: (x["pred_overnight"] / x["current_price"]), reverse=True)
        return sorted_res[:10]

if __name__ == "__main__":
    scanner = MarketScanner()
    print("Scanning High Volume...")
    print(len(scanner.get_high_volume_top_10()))
    print("Scanning Long Term...")
    print(len(scanner.get_long_term_buy_top_25()))
