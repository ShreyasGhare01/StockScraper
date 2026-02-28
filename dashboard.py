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

# Analysis Mode Selection
analysis_mode = st.sidebar.radio("Select Analysis Mode", ["Edge Detection (Probabilistic)", "Price Prediction (Regression)"])
mode_code = "edge" if "Edge" in analysis_mode else "regression"

page = st.sidebar.radio("Navigate", [
    "Individual Stock Analysis",
    "Market Scanners",
    "Top Daily Gainers",
    "Trading Education"
])

if page == "Individual Stock Analysis":
    st.title("📊 Individual Stock Analysis")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    period = st.sidebar.selectbox("Select Period", ["5d", "1mo", "6mo", "1y", "2y", "5y", "max"], index=4)

    if st.sidebar.button("Run Analysis"):
        with st.spinner(f"Analyzing {ticker}..."):
            res = AnalysisEngine.analyze_ticker(ticker, period=period, mode=mode_code)
            if "error" in res:
                st.error(res["error"])
            else:
                data = res["data"]
                # Display Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${res['current_price']:.2f}")

                if mode_code == "edge":
                    col2.metric("Overnight Edge (>2.5%)", f"{res['pred_overnight']*100:.1f}%")
                    col3.metric("5-Day Edge (>5.0%)", f"{res['pred_5day']*100:.1f}%")
                else:
                    col2.metric("Next Day Pred", f"${res['pred_overnight']:.2f}", f"{((res['pred_overnight']-res['current_price'])/res['current_price'])*100:+.2f}%")
                    col3.metric("5-Day Pred", f"${res['pred_5day']:.2f}", f"{((res['pred_5day']-res['current_price'])/res['current_price'])*100:+.2f}%")

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
                    with st.expander("Latest Headlines (Live)"):
                        for h in s['latest_headlines']: st.write(f"• {h}")

                # Options
                st.subheader("💡 Suggested Options")
                # Pass probabilities to strategist
                analysis_enhanced = res.copy()
                strategist = OptionStrategist(ticker, res['current_price'], analysis_enhanced, data)
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
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, ticker):
            pct = current / total
            progress_bar.progress(pct)
            status_text.text(f"Analyzing {ticker}... ({current}/{total})")

        scanner = MarketScanner()
        if scan_type == "High-Volume Top 10":
            results = scanner.get_high_volume_top_10(callback=update_progress, mode=mode_code)
        elif scan_type == "General Low Price Top 25 ($0.5 - $3)":
            results = scanner.get_general_low_price_top_25(callback=update_progress, mode=mode_code)
        elif scan_type == "Short Term Buy Top 25 (1-Week Rocket)":
            results = scanner.get_short_term_buy_top_25(callback=update_progress, mode=mode_code)
        else:
            results = scanner.get_long_term_buy_top_25(callback=update_progress, mode=mode_code)

        if results:
            display_data = []
            for r in results:
                row = {
                    "Ticker": r["ticker"],
                    "Price": f"${r['current_price']:.2f}",
                    "Score": r["score"]
                }
                if mode_code == "edge":
                    row["Overnight Edge"] = f"{r['pred_overnight']*100:.1f}%"
                    row["5-Day Edge"] = f"{r['pred_5day']*100:.1f}%"
                else:
                    row["Pred Price"] = f"${r['pred_overnight']:.2f}"
                    row["Next Day %"] = f"{((r['pred_overnight']-r['current_price'])/r['current_price'])*100:+.2f}%"
                if scan_type == "Long Term Buy Top 25 (High Dividends)":
                    row["Div Yield"] = f"{r['dividend_yield']*100:.2f}%"
                if scan_type == "Short Term Buy Top 25 (1-Week Rocket)":
                    row["Sentiment"] = f"{r['sentiment_score']*100:.0f}%"
                display_data.append(row)
            st.table(pd.DataFrame(display_data))
        else:
            st.warning("No stocks found matching criteria.")

elif page == "Top Daily Gainers":
    st.title("🔥 Top Predicted Daily Gainers")
    st.write("Stocks with the highest predicted percentage increase for the next trading day.")

    if st.button("Calculate Top Gainers"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, ticker):
            pct = current / total
            progress_bar.progress(pct)
            status_text.text(f"Analyzing {ticker}... ({current}/{total})")

        scanner = MarketScanner()
        top_gainers = scanner.get_top_10_daily_gainers(callback=update_progress, mode=mode_code)

        if top_gainers:
            # 1. Table
            display_data = []
            for r in top_gainers:
                row = {
                    "Ticker": r["ticker"],
                    "Price": f"${r['current_price']:.2f}",
                    "Score": r["score"]
                }
                if mode_code == "edge":
                    row["Overnight Edge"] = f"{r['pred_overnight']*100:.1f}%"
                    row["5-Day Edge"] = f"{r['pred_5day']*100:.1f}%"
                else:
                    row["Pred Price"] = f"${r['pred_overnight']:.2f}"
                    row["Next Day %"] = f"{((r['pred_overnight']-r['current_price'])/r['current_price'])*100:+.2f}%"
                display_data.append(row)
            st.table(pd.DataFrame(display_data))

            # 2. Portfolio Simulator ($100)
            st.divider()
            st.header("💰 $100 Profit Optimizer")
            st.write("If the model were given $100 to start the next day, here is how it would optimize profit:")

            best = top_gainers[0]
            shares = 100 / best['current_price']

            if mode_code == "edge":
                # In Edge mode, we assume a win if it hits the threshold
                predicted_exit = best['current_price'] * 1.025
                profit_reason = f"This ticker has a **{best['pred_overnight']*100:.1f}%** probability of exceeding a **+2.5%** move."
            else:
                # In Regression mode, use the predicted price
                predicted_exit = best['pred_overnight']
                profit_reason = f"The model predicts a **{((best['pred_overnight']-best['current_price'])/best['current_price'])*100:+.2f}%** return."

            total_exit = shares * predicted_exit
            profit = total_exit - 100
            margin = (profit / 100) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal Ticker", best['ticker'])
            c2.metric("Shares Purchased", f"{shares:.4f}")
            c3.metric("Predicted Profit", f"${profit:.2f}", f"{margin:+.2f}% Margin")

            st.info(f"**Strategy:** Buy **{shares:.4f}** shares of **{best['ticker']}** at **${best['current_price']:.2f}**. " + profit_reason)
        else:
            st.error("Could not retrieve gainer data.")

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
                # Add patterned image if available
                st.write(p['description'])
                st.info(f"**What it means:** {p['indication']}")
            with col_img:
                st.image(p['image'], width=200)
            st.divider()
