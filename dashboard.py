import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from option_strategy import OptionStrategist
from pump_detector import PumpDetector
from sentiment_scanner import SentimentScanner
import datetime

st.set_page_config(page_title="Stock AI Analyzer", layout="wide")

st.title("🚀 Stock AI Analyzer & Pump Detector")
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "max"], index=1)

if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # 1. Fetch Data
            fetcher = StockDataFetcher(ticker)
            data = fetcher.fetch_history(period=period)
            current_price = data['Close'].iloc[-1]

            # 2. Technical Analysis
            analyzer = TechnicalAnalyzer(data)
            data_with_indicators = analyzer.add_all_indicators()
            analysis = analyzer.get_signals()

            # 3. Price Prediction
            predictor = StockPredictor(data_with_indicators)
            predictor.train()
            predicted_price = predictor.predict_next_day()

            # 4. Pump Detection
            detector = PumpDetector(data)
            detector.train()
            pump_results = detector.detect_current()

            # 5. Sentiment Scan
            scanner = SentimentScanner()
            sentiment_results = scanner.scan_news(ticker)

            # 6. Option Strategy
            strategist = OptionStrategist(ticker, current_price, predicted_price, analysis, data_with_indicators)
            trades = strategist.suggest_trades()

            # --- Layout ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted (Next Day)", f"${predicted_price:.2f}", f"{((predicted_price-current_price)/current_price)*100:.2f}%")
            col3.metric("Rec", analysis['recommendation'], f"Score: {analysis['score']}")

            # --- Charts ---
            st.subheader("Price Action & Indicators")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name="Price"))

            # Add EMA 200
            ema200_col = [c for c in data_with_indicators.columns if c.startswith('EMA_200')][0]
            fig.add_trace(go.Scatter(x=data.index, y=data_with_indicators[ema200_col], name="EMA 200", line=dict(color='orange')))

            st.plotly_chart(fig, use_container_width=True)

            # --- Pump Alert ---
            st.subheader("🚨 Pump Detector AI")
            if pump_results['is_pump'] or pump_results['pump_probability'] > 0.3:
                st.warning(f"Potential Pump Alert! Probability: {pump_results['pump_probability']*100:.1f}%")
            else:
                st.success(f"No pump detected. Probability: {pump_results['pump_probability']*100:.1f}%")

            st.write("**Signals:**")
            for r in pump_results['reasons']:
                st.write(f"- {r}")

            # --- Sentiment & Research ---
            st.subheader("🔍 AI Research & Sentiment")
            st.write(f"Found **{sentiment_results['pump_mentions']}** potential 'pump' related keywords in recent headlines.")
            st.progress(scanner.get_pump_score(ticker), text=f"Pump Sentiment Score: {scanner.get_pump_score(ticker)*100:.1f}%")

            with st.expander("Latest Research Headlines"):
                for head in sentiment_results['latest_headlines']:
                    st.write(f"• {head}")

            # --- Option Trades ---
            st.subheader("💡 Suggested Option Trades")
            if isinstance(trades, list):
                for t in trades:
                    with st.container():
                        st.markdown(f"**{t['type']} - {t['strategy']}**")
                        st.write(f"Strike: ${t['strike']} | Expiry: {t['expiry']}")
                        st.info(f"Reason: {t['reason']}")
            else:
                st.write(trades)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Enter a ticker and click 'Run Analysis' to start.")
