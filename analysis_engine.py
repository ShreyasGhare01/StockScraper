from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from sentiment_scanner import SentimentScanner
from backtester import Backtester
from vision_engine import VisionEngine
import pandas as pd
import numpy as np

class AnalysisEngine:
    @staticmethod
    def backtest_ticker(ticker, data_with_indicators):
        """
        Runs a historical backtest for the last 100 days of data.
        """
        # We need historical predictions. For backtest, we iterate through last 100 days.
        # This is expensive, so we'll simulate it by training on a subset.
        try:
            # Simple walk-forward:
            # We'll take the last 150 days. For the last 100 days, we'll "predict"
            # using a model trained on everything before that point.
            bt_data = data_with_indicators.tail(150)
            if len(bt_data) < 100:
                return {"error": "Insufficient data for backtest"}

            # For prototype speed, we'll train one model on the first 50 and test on last 100.
            train_data = data_with_indicators.iloc[:-100]
            test_data = data_with_indicators.tail(100)

            predictor = StockPredictor(train_data, target_type="overnight", mode="edge")
            predictor.train(optimize=False) # Fast for backtest

            # Predict for all 100 test days
            # We scale the test data using the scaler from train
            X_test = test_data[predictor.feature_cols]
            X_test_scaled = predictor.scaler.transform(X_test)
            # predict_proba returns [prob_0, prob_1]
            probs = predictor.model.predict_proba(X_test_scaled)[:, 1]

            backtester = Backtester(test_data, probs)
            return backtester.run()
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def analyze_visual(image_path, manual_ticker=None):
        """
        Integrates vision analysis with quantitative metrics.
        """
        vision = VisionEngine()
        res = vision.process_screenshot(image_path)

        if "error" in res:
            return res

        ticker = manual_ticker if manual_ticker and manual_ticker.strip() else res["ticker"]

        if not ticker or ticker == "UNKNOWN":
            return {"error": "Could not detect ticker from screenshot. Please enter it manually."}

        # Overwrite ticker if manual override was used
        res["ticker"] = ticker.upper()

        # Enrich vision data with live quantitative data
        # For Visual Trader, we focus on INTRADAY edge
        quant_res = AnalysisEngine.analyze_ticker(ticker, period="1y", mode="edge", target_override="intraday_edge")

        if "error" in quant_res:
            return {"error": f"Detected ticker {ticker} but could not fetch data. Check if the ticker is correct."}

        # Merge results
        res["quant_analysis"] = quant_res
        res["direction"] = quant_res["recommendation"]

        return res

    @staticmethod
    def analyze_ticker(ticker, period="2y", optimize=True, mode="edge", target_override=None):
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

            # 3. Prediction Mode Setup
            # If target_override is provided (e.g., intraday_edge), focus on that instead of default
            target_1 = target_override if target_override else "overnight"
            target_2 = "5day" if not target_override else None

            # 3. Target 1 Prediction
            predictor_overnight = StockPredictor(data_with_indicators, target_type=target_1, mode=mode)
            predictor_overnight.train(optimize=optimize)
            pred_overnight, ev_overnight = predictor_overnight.predict_probability()

            # 4. Target 2 Prediction (Optional)
            if target_2:
                predictor_5day = StockPredictor(data_with_indicators, target_type=target_2, mode=mode)
                predictor_5day.train(optimize=optimize)
                pred_5day, ev_5day = predictor_5day.predict_probability()
            else:
                pred_5day, ev_5day = 0.0, 0.0
                predictor_5day = predictor_overnight # Fallback for reliability metrics

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
                "acc_overnight": float(predictor_overnight.accuracy_score),
                "acc_5day": float(predictor_5day.accuracy_score),
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
