import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from predictor import StockPredictor
from option_strategy import OptionStrategist
from pump_detector import PumpDetector
from sentiment_scanner import SentimentScanner
from analysis_engine import AnalysisEngine
from market_scanner import MarketScanner
import datetime

st.set_page_config(page_title="Stock AI Analyzer", layout="wide")

st.sidebar.title("🚀 AI Stock Suite")
page = st.sidebar.radio("Navigate", ["Single Ticker Analysis", "Top 10 Strong Buys"])

if page == "Single Ticker Analysis":
    st.title("📊 Individual Stock Analysis")
    st.sidebar.header("Settings")

    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "max"], index=1)

    if st.sidebar.button("Run Analysis"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Use AnalysisEngine for core logic
                res = AnalysisEngine.analyze_ticker(ticker, period=period)

                if "error" in res:
                    st.error(res["error"])
                else:
                    data = res["data"]
                    current_price = res["current_price"]
                    predicted_price = res["predicted_price"]
                    predicted_change = res["predicted_change_pct"]
                    recommendation = res["recommendation"]
                    score = res["score"]
                    signals = res["signals"]

                    # 4. Pump Detection
                    detector = PumpDetector(data)
                    detector.train()
                    pump_results = detector.detect_current()

                    # 5. Sentiment Scan
                    scanner = SentimentScanner()
                    sentiment_results = scanner.scan_news(ticker)

                    # 6. Option Strategy
                    strategist = OptionStrategist(ticker, current_price, predicted_price, {"recommendation": recommendation, "score": score}, data)
                    trades = strategist.suggest_trades()

                    # --- Layout ---
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("Predicted (Next Day)", f"${predicted_price:.2f}", f"{predicted_change:+.2f}%")
                    col3.metric("Rec", recommendation, f"Score: {score}")

                    # --- Charts ---
                    st.subheader("Price Action & Indicators")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'], name="Price"))

                    # Add EMA 200
                    if 'EMA_200' in data.columns:
                        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], name="EMA 200", line=dict(color='orange')))

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

elif page == "Top 10 Strong Buys":
    st.title("🏆 Market Scanner: Top 10 Strong Buys")
    st.write("Scan the market for the best AI-rated opportunities based on technical indicators and price predictions.")

    if st.button("Start Market Scan"):
        with st.spinner("Scanning market tickers... This may take a minute."):
            scanner = MarketScanner()
            top_buys = scanner.scan_top_buys(limit=10)

            if top_buys:
                # Convert to DataFrame for display
                df_results = []
                for res in top_buys:
                    df_results.append({
                        "Ticker": res["ticker"],
                        "Score": res["score"],
                        "Recommendation": res["recommendation"],
                        "Current Price": f"${res['current_price']:.2f}",
                        "Predicted Price": f"${res['predicted_price']:.2f}",
                        "Expected Change": f"{res['predicted_change_pct']:+.2f}%"
                    })

                df = pd.DataFrame(df_results)
                st.table(df)
            else:
                st.warning("No high-score stocks found in the current scan.")
