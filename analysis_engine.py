from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor

class AnalysisEngine:
    @staticmethod
    def analyze_ticker(ticker, period="2y"):
        try:
            # 1. Fetch Data
            fetcher = StockDataFetcher(ticker)
            data = fetcher.fetch_history(period=period)
            if data.empty:
                return {"error": f"No data for {ticker}"}

            current_price = data['Close'].iloc[-1]

            # 2. Technical Analysis
            analyzer = TechnicalAnalyzer(data)
            data_with_indicators = analyzer.add_all_indicators()
            analysis = analyzer.get_signals()

            if "error" in analysis:
                return {"ticker": ticker, "error": analysis["error"]}

            # 3. Price Prediction
            predictor = StockPredictor(data_with_indicators)
            predictor.train()
            predicted_price = predictor.predict_next_day()

            predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "predicted_change_pct": predicted_change_pct,
                "score": analysis["score"],
                "recommendation": analysis["recommendation"],
                "signals": analysis["signals"],
                "data": data_with_indicators
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

if __name__ == "__main__":
    # Quick verification
    engine = AnalysisEngine()
    res = engine.analyze_ticker("AAPL")
    print(res["ticker"], res["score"], res["recommendation"], res["predicted_price"])
