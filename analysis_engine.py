from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from sentiment_scanner import SentimentScanner
import pandas as pd
import numpy as np

class AnalysisEngine:
    @staticmethod
    def analyze_ticker(ticker, period="2y", optimize=True, mode="edge"):
        try:
            # 1. Fetch Data
            fetcher = StockDataFetcher(ticker)
            data = fetcher.fetch_history(period=period)
            if data.empty:
                return {"error": f"No data for {ticker}"}

            current_price = data['Close'].iloc[-1]

            # Fetch dividend yield
            try:
                info = fetcher.get_info()
                dividend_yield = info.get("dividendYield", 0)
            except:
                dividend_yield = 0

            if dividend_yield is None:
                dividend_yield = 0

            # 2. Technical Analysis
            analyzer = TechnicalAnalyzer(data)
            data_with_indicators = analyzer.add_all_indicators()

            # 2b. Add Macro Data
            try:
                # Get start and end dates from data
                # We need to handle timezone-aware vs naive dates
                start_date = data.index[0].to_pydatetime()
                end_date = data.index[-1].to_pydatetime()

                # Strip timezone if it exists (FRED expects naive)
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)

                macro_df = fetcher.fetch_macro_data(start_date, end_date)
                regime_df = fetcher.fetch_market_regime(start_date, end_date)

                # Make data index naive if it's tz-aware to match external sources
                if data_with_indicators.index.tz is not None:
                    data_with_indicators.index = data_with_indicators.index.tz_localize(None)

                if not macro_df.empty:
                    data_with_indicators = data_with_indicators.join(macro_df, how='left').ffill()
                if not regime_df.empty:
                    # Strip regime timezone just in case
                    if regime_df.index.tz is not None:
                        regime_df.index = regime_df.index.tz_localize(None)
                    data_with_indicators = data_with_indicators.join(regime_df, how='left').ffill()
            except Exception as e:
                print(f"Warning: Failed to merge macro/regime data: {e}")

            # 2c. Add Fundamental & AlphaVantage Sentiment
            try:
                fundamentals = fetcher.get_fundamental_data()
                for key, value in fundamentals.items():
                    data_with_indicators[key] = value

                # Fetch AlphaVantage sentiment (using demo key)
                av_sentiment = fetcher.fetch_alpha_vantage_sentiment()
                data_with_indicators['av_news_sentiment'] = av_sentiment
            except Exception as e:
                print(f"Warning: Failed to merge fundamental/AV data: {e}")

            analysis = analyzer.get_signals()

            if "error" in analysis:
                return {"ticker": ticker, "error": analysis["error"]}

            # 3. Overnight Prediction
            predictor_overnight = StockPredictor(data_with_indicators, target_type="overnight", mode=mode)
            predictor_overnight.train(optimize=optimize)
            pred_overnight, ev_overnight = predictor_overnight.predict_probability()

            # 4. 5-Day Prediction
            predictor_5day = StockPredictor(data_with_indicators, target_type="5day", mode=mode)
            predictor_5day.train(optimize=optimize)
            pred_5day, ev_5day = predictor_5day.predict_probability()

            # 5. Sentiment & Velocity metrics
            sentiment_scanner = SentimentScanner()
            news_metrics = sentiment_scanner.get_news_metrics(ticker)

            # Add news features to data for XGBoost (Velocity focus)
            data_with_indicators['news_count'] = news_metrics['news_count']
            data_with_indicators['news_velocity'] = news_metrics['news_velocity']
            data_with_indicators['sentiment_velocity'] = news_metrics['sentiment_velocity']
            data_with_indicators['recent_sentiment'] = news_metrics['sentiment']

            return {
                "ticker": ticker,
                "current_price": float(current_price),
                "pred_overnight": float(pred_overnight),
                "pred_5day": float(pred_5day),
                "ev_overnight": float(ev_overnight),
                "ev_5day": float(ev_5day),
                "mode": mode,
                "score": int(analysis["score"]),
                "recommendation": analysis["recommendation"],
                "signals": analysis["signals"],
                "dividend_yield": float(dividend_yield),
                "sentiment_score": float(news_metrics['sentiment']),
                "data": data_with_indicators
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

if __name__ == "__main__":
    # Quick verification
    engine = AnalysisEngine()
    res = engine.analyze_ticker("AAPL")
    if "error" in res:
        print(f"Error: {res['error']}")
    else:
        print(f"Ticker: {res['ticker']}, Score: {res['score']}, Mode: {res['mode']}")
        if res['mode'] == "edge":
            print(f"Overnight Prob (>2.5%): {res['pred_overnight']*100:.1f}% (EV: {res['ev_overnight']*100:.2f}%)")
            print(f"5-Day Prob (>5.0%): {res['pred_5day']*100:.1f}% (EV: {res['ev_5day']*100:.2f}%)")
        else:
            print(f"Predicted Tomorrow: ${res['pred_overnight']:.2f}")
            print(f"Predicted 1-Week: ${res['pred_5day']:.2f}")
