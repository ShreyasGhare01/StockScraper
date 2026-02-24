import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis_engine import AnalysisEngine
from market_scanner import MarketScanner
from pump_detector import PumpDetector
from sentiment_scanner import SentimentScanner
from option_strategy import OptionStrategist
import trading_patterns_info as info

st.set_page_config(page_title="Stock AI Analyzer Pro", layout="wide")

st.sidebar.title("🚀 AI Stock Suite")
page = st.sidebar.radio("Navigate", [
    "Individual Stock Analysis",
    "Market Scanners",
    "Trading Education"
])

if page == "Individual Stock Analysis":
    st.title("📊 Individual Stock Analysis")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "max"], index=1)

    if st.sidebar.button("Run Analysis"):
        with st.spinner(f"Analyzing {ticker}..."):
            res = AnalysisEngine.analyze_ticker(ticker, period=period)
            if "error" in res:
                st.error(res["error"])
            else:
                data = res["data"]
                # Display Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${res['current_price']:.2f}")
                col2.metric("Next Day Pred", f"${res['predicted_price']:.2f}", f"{res['predicted_change_pct']:+.2f}%")
                col3.metric("7-Day Pred", f"${res['predicted_price_7d']:.2f}", f"{res['predicted_change_7d_pct']:+.2f}%")
                col4.metric("Score", res["score"], res["recommendation"])

                # Chart
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
                if 'EMA_200' in data.columns:
                    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], name="EMA 200", line=dict(color='orange')))
                st.plotly_chart(fig, use_container_width=True)

                # Pump Detection & Sentiment
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("🚨 Pump Detector")
                    detector = PumpDetector(data)
                    detector.train()
                    p = detector.detect_current()
                    if p['is_pump']: st.warning(f"Pump Probability: {p['pump_probability']*100:.1f}%")
                    else: st.success(f"Pump Probability: {p['pump_probability']*100:.1f}%")
                    for r in p['reasons']: st.write(f"- {r}")
                with c2:
                    st.subheader("🔍 AI Sentiment")
                    scanner = SentimentScanner()
                    s = scanner.scan_news(ticker)
                    st.progress(scanner.get_pump_score(ticker), text=f"Sentiment Score: {scanner.get_pump_score(ticker)*100:.1f}%")
                    with st.expander("Headlines"):
                        for h in s['latest_headlines']: st.write(f"• {h}")

                # Options
                st.subheader("💡 Suggested Options")
                strategist = OptionStrategist(ticker, res['current_price'], res['predicted_price'], {"recommendation": res['recommendation'], "score": res['score']}, data)
                trades = strategist.suggest_trades()
                st.write(trades)

elif page == "Market Scanners":
    st.title("🏆 Market Scanning Opportunity Finder")
    scan_type = st.selectbox("Select List Type", [
        "High-Volume Top 10",
        "General Low Price Top 25 ($0.5 - $3)",
        "Short Term Buy Top 25 (1-Week Rocket)",
        "Long Term Buy Top 25 (High Dividends)"
    ])

    if st.button("Start Market Scan"):
        with st.spinner("Scanning markets..."):
            scanner = MarketScanner()
            if scan_type == "High-Volume Top 10":
                results = scanner.get_high_volume_top_10()
            elif scan_type == "General Low Price Top 25 ($0.5 - $3)":
                results = scanner.get_general_low_price_top_25()
            elif scan_type == "Short Term Buy Top 25 (1-Week Rocket)":
                results = scanner.get_short_term_buy_top_25()
            else:
                results = scanner.get_long_term_buy_top_25()

            if results:
                display_data = []
                for r in results:
                    row = {
                        "Ticker": r["ticker"],
                        "Price": f"${r['current_price']:.2f}",
                        "Score": r["score"],
                        "Next Day %": f"{r['predicted_change_pct']:+.2f}%",
                        "7-Day %": f"{r['predicted_change_7d_pct']:+.2f}%"
                    }
                    if scan_type == "Long Term Buy Top 25 (High Dividends)":
                        row["Div Yield"] = f"{r['dividend_yield']*100:.2f}%"
                    if scan_type == "Short Term Buy Top 25 (1-Week Rocket)":
                        row["Sentiment"] = f"{r['sentiment_score']*100:.0f}%"
                    display_data.append(row)
                st.table(pd.DataFrame(display_data))
            else:
                st.warning("No stocks found matching criteria.")

elif page == "Trading Education":
    st.title("📚 Trading Education & Pattern Guide")
    st.write("Master the art of reading charts and identifying profitable patterns.")

    st.markdown(info.GENERAL_TRENDS)

    st.subheader("🕯️ Essential Candlestick Patterns")
    for p in info.PATTERNS:
        with st.container():
            col_text, col_img = st.columns([2, 1])
            with col_text:
                st.markdown(f"### {p['name']} ({p['type']})")
                st.write(p['description'])
                st.info(f"**What it means:** {p['indication']}")
            with col_img:
                st.image(p['image'], width=200)
            st.divider()
