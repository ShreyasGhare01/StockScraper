from pump_detector import PumpDetector
from sentiment_scanner import SentimentScanner
from data_fetcher import StockDataFetcher

def test_pump_detector():
    print("Testing PumpDetector...")
    fetcher = StockDataFetcher("AAPL")
    data = fetcher.fetch_history(period="1y")
    detector = PumpDetector(data)
    if detector.train():
        res = detector.detect_current()
        assert "pump_probability" in res
        print("PumpDetector test passed.")
    else:
        print("PumpDetector training failed (not enough data?)")

def test_sentiment_scanner():
    print("Testing SentimentScanner...")
    scanner = SentimentScanner()
    res = scanner.scan_news("AAPL")
    assert "ticker" in res
    assert "latest_headlines" in res
    print("SentimentScanner test passed.")

if __name__ == "__main__":
    test_pump_detector()
    test_sentiment_scanner()
