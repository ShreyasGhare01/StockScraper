from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from sentiment_scanner import SentimentScanner
import pandas as pd
import numpy as np

class AnalysisEngine:
    @staticmethod
    def analyze_ticker(ticker, period="2y", optimize=True):
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
                if not macro_df.empty:
                    # Merge macro data into stock data.
                    # First make data index naive if it's tz-aware to match macro_df
                    if data_with_indicators.index.tz is not None:
                        data_with_indicators.index = data_with_indicators.index.tz_localize(None)

                    data_with_indicators = data_with_indicators.join(macro_df, how='left').ffill()
            except Exception as e:
                print(f"Warning: Failed to merge macro data: {e}")

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

            # 3. Price Prediction (Next Day)
            predictor = StockPredictor(data_with_indicators)
            predictor.train(optimize=optimize)
            predicted_price = predictor.predict_next_day()
            predicted_change_pct = ((predicted_price - current_price) / current_price) * 100

            # 4. Short-term Prediction (7-Day)
            # Add target for 7-day prediction
            df_7d = data_with_indicators.copy()
            df_7d['Target_7d'] = df_7d['Close'].shift(-5) # 5 trading days approx 1 week
            df_7d_train = df_7d.dropna()

            if len(df_7d_train) > 50:
                predictor_7d = StockPredictor(data_with_indicators)
                predictor_7d.data = df_7d_train # Train on shifted data
                # Re-run preparation logic locally or use existing feature_cols
                X_train = df_7d_train[[c for c in df_7d_train.columns if c in predictor.feature_cols]]
                y_train = df_7d_train['Target_7d']

                predictor_7d.scaler.fit(X_train)
                X_scaled = predictor_7d.scaler.transform(X_train)
                predictor_7d.model.fit(X_scaled, y_train)

                # Predict 7 days out using latest available features
                last_row = data_with_indicators[[c for c in data_with_indicators.columns if c in predictor.feature_cols]].tail(1)
                last_row_scaled = predictor_7d.scaler.transform(last_row)
                predicted_price_7d = predictor_7d.model.predict(last_row_scaled)[0]
                predicted_change_7d_pct = ((predicted_price_7d - current_price) / current_price) * 100
            else:
                predicted_price_7d = predicted_price
                predicted_change_7d_pct = predicted_change_pct

            # 5. Sentiment Score
            sentiment_scanner = SentimentScanner()
            sentiment_res = sentiment_scanner.scan_news(ticker)
            sentiment_score = sentiment_scanner.get_pump_score(ticker)

            # Add news features to data for XGBoost (Latest context)
            data_with_indicators['recent_pump_mentions'] = sentiment_res['pump_mentions']
            data_with_indicators['recent_sentiment_score'] = sentiment_res['sentiment_score']

            return {
                "ticker": ticker,
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "predicted_change_pct": float(predicted_change_pct),
                "predicted_price_7d": float(predicted_price_7d),
                "predicted_change_7d_pct": float(predicted_change_7d_pct),
                "score": int(analysis["score"]),
                "recommendation": analysis["recommendation"],
                "signals": analysis["signals"],
                "dividend_yield": float(dividend_yield),
                "sentiment_score": float(sentiment_score),
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
        print(f"Ticker: {res['ticker']}, Score: {res['score']}, Div: {res['dividend_yield']}, 7d Change: {res['predicted_change_7d_pct']:.2f}%")
