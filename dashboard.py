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
    "AI Visual Trader",
    "Sector Alpha Finder",
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
                    col2.metric("Overnight EV", f"{res['ev_overnight']*100:.2f}%", f"{res['pred_overnight']*100:.1f}% Prob")
                    col3.metric("5-Day EV", f"{res['ev_5day']*100:.2f}%", f"{res['pred_5day']*100:.1f}% Prob")
                    st.caption(f"Model Reliability: Overnight ({res['acc_overnight']*100:.1f}%) | 5-Day ({res['acc_5day']*100:.1f}%)")
                else:
                    col2.metric("Next Day Pred", f"${res['pred_overnight']:.2f}", f"{((res['pred_overnight']-res['current_price'])/res['current_price'])*100:+.2f}%")
                    col3.metric("5-Day Pred", f"${res['pred_5day']:.2f}", f"{((res['pred_5day']-res['current_price'])/res['current_price'])*100:+.2f}%")
                    st.caption(f"Model Accuracy (1-MAPE): Tomorrow ({res['acc_overnight']*100:.1f}%) | 1-Week ({res['acc_5day']*100:.1f}%)")

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

                # Backtest Section
                st.divider()
                st.subheader("📈 Historical Backtest (Last 100 Days)")
                if st.button("Run Historical Backtest"):
                    with st.spinner("Simulating historical performance..."):
                        bt = AnalysisEngine.backtest_ticker(ticker, data)
                        if "error" in bt:
                            st.error(bt["error"])
                        else:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Return", f"{bt['total_return']*100:.2f}%")
                            c2.metric("Win Rate", f"{bt['win_rate']*100:.1f}%")
                            c3.metric("Sharpe Ratio", f"{bt['sharpe_ratio']:.2f}")
                            c4.metric("Profit Factor", f"{bt['profit_factor']:.2f}")

                            st.metric("Max Drawdown", f"{bt['max_drawdown']*100:.2f}%")

                            # Equity Curve
                            fig_bt = go.Figure()
                            fig_bt.add_trace(go.Scatter(y=bt['equity_curve'], name="Equity Curve", line=dict(color='green')))
                            fig_bt.update_layout(title=f"{ticker} Backtest Equity Curve ($10,000 Starting)", xaxis_title="Trades", yaxis_title="Portfolio Value")
                            st.plotly_chart(fig_bt, use_container_width=True)
                            st.write(f"Executed **{bt['num_trades']}** simulated trades over the last 100 trading days.")

elif page == "Market Scanners":
    st.title("🏆 Market Scanning Opportunity Finder")
    scan_type = st.selectbox("Select List Type", [
        "High-Volume Top 10",
        "General Low Price Top 25 ($0.5 - $5)",
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
        elif scan_type == "General Low Price Top 25 ($0.5 - $5)":
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
                    row["Overnight EV"] = f"{r['ev_overnight']*100:.2f}%"
                    row["5-Day EV"] = f"{r['ev_5day']*100:.2f}%"
                    row["Overnight EV"] = f"{r['ev_overnight']*100:.2f}%"
                    row["5-Day EV"] = f"{r['ev_5day']*100:.2f}%"
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

elif page == "AI Visual Trader":
    st.title("📸 AI Visual Chart Analysis")
    st.write("Upload a chart screenshot to generate an instant intraday options strategy.")

    uploaded_file = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg", "webp"])

    manual_ticker = st.text_input("Manual Ticker Override (Optional)", placeholder="e.g. NVDA").upper()

    strategy_type = st.radio("AI Analysis Strategy", [
        "Intraday Scalp (Edge)",
        "Overnight Swing (Edge)",
        "Price Forecast (Regression)"
    ], horizontal=True)

    if strategy_type == "Intraday Scalp (Edge)":
        v_mode, v_target = "edge", "intraday_edge"
    elif strategy_type == "Overnight Swing (Edge)":
        v_mode, v_target = "edge", "overnight"
    else:
        v_mode, v_target = "regression", "overnight"

    if uploaded_file is not None:
        # Save temp file
        temp_path = "temp_screenshot.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Input Screenshot", use_container_width=True)

        if st.button("Analyze Screenshot"):
            with st.spinner("Processing image and running AI models..."):
                res = AnalysisEngine.analyze_visual(temp_path, manual_ticker=manual_ticker, mode=v_mode, target_override=v_target)

                if "error" in res:
                    st.error(res["error"])
                else:
                    quant = res["quant_analysis"]

                    st.success(f"Successfully analyzed **{res['ticker']}**")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Ticker", res["ticker"])
                    col2.metric("Detected Price", f"${res['current_price']:.2f}")
                    col3.metric("AI Confidence", f"{res['confidence']*100:.0f}%")

                    st.divider()
                    st.subheader("💡 Intraday Options Recommendation")

                    # Generate options logic for vision result
                    # We pass the enhanced quant results to the strategist
                    strategist = OptionStrategist(res['ticker'], res['current_price'], quant, quant['data'])
                    trades = strategist.suggest_trades()

                    if isinstance(trades, list):
                        trade = trades[0]
                        # Calculate SL/TP for structured JSON output as requested
                        # Intraday SL: -1.5% TP: +3.0%
                        sl = res['current_price'] * (0.985 if trade['type'] == 'CALL' else 1.015)
                        tp = res['current_price'] * (1.03 if trade['type'] == 'CALL' else 0.97)

                        exp = "Same Day (0DTE)" if strategy_type == "Intraday Scalp (Edge)" else "1-2 Weeks"
                        prob_key = "estimated_success_probability" if v_mode == "edge" else "predicted_return"
                        prob_val = f"{quant['pred_overnight']*100:.1f}%" if v_mode == "edge" else f"{((quant['pred_overnight']-res['current_price'])/res['current_price'])*100:+.2f}%"

                        output_json = {
                            "ticker": res["ticker"],
                            "strategy": strategy_type,
                            "direction": res["direction"],
                            "option_type": trade["type"],
                            "strike_price": trade["strike"],
                            "expiration": exp,
                            "entry_price": f"${res['current_price']:.2f}",
                            "stop_loss": f"${sl:.2f}",
                            "take_profit_levels": [f"${tp:.2f}"],
                            "risk_reward_ratio": "1:2",
                            "confidence_score": f"{res['confidence']*100:.0f}%",
                            prob_key: prob_val,
                            "predicted_duration_minutes": "30-60" if strategy_type == "Intraday Scalp (Edge)" else "1-5 Days"
                        }

                        st.json(output_json)

                        st.info(f"**Trade Details:** {trade['reason']}")
                    else:
                        st.warning(trades)

elif page == "Sector Alpha Finder":
    st.title("🔥 Sector Alpha High-Impact Finder")
    st.write("Scan entire sectors for high-impact intraday moves (>3.0% returns) using news, research, and technical velocity.")

    selected_sector = st.selectbox("Select Target Sector", [
        "Technology", "Biotech/Healthcare", "Energy", "Finance", "Consumer"
    ])

    if st.button("Find Sector Alpha"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, ticker):
            pct = current / total
            progress_bar.progress(pct)
            status_text.text(f"Scanning {ticker} in {selected_sector}... ({current}/{total})")

        scanner = MarketScanner()
        results = scanner.get_sector_top_gains(selected_sector, callback=update_progress)

        if results:
            best = results[0]
            st.success(f"Best Sector Opportunity Found: **{best['ticker']}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${best['current_price']:.2f}")
            c2.metric("High-Impact Prob", f"{best['pred_overnight']*100:.1f}%")
            c3.metric("Expected Value", f"{best['ev_overnight']*100:.2f}%")

            st.divider()
            st.subheader("🎯 Optimal Call Option Strategy")

            # Use strategist for the best ticker
            strategist = OptionStrategist(best['ticker'], best['current_price'], best, best['data'])
            trades = strategist.suggest_trades()

            if isinstance(trades, list):
                trade = trades[0]
                col_t, col_j = st.columns([1, 1])
                with col_t:
                    st.markdown(f"### Recommended Call: {trade['strike']}C")
                    st.write(f"**Rationale:** {trade['reason']}")
                    st.write(f"**Sentiment Velocity:** {best['data']['sentiment_velocity'].iloc[-1]*100:+.1f}%")
                    st.write(f"**News Velocity:** {best['data']['news_velocity'].iloc[-1]:.2f}")
                with col_j:
                    st.json({
                        "ticker": best["ticker"],
                        "sector": selected_sector,
                        "option_strategy": "High-Delta Call Scalp",
                        "strike": trade["strike"],
                        "expiration": "Same Day / Next Day",
                        "target_gain": "> +3.0% (Stock Move)",
                        "confidence": f"{best['acc_overnight']*100:.0f}%",
                        "news_research_alignment": "Positive" if best['sentiment_score'] > 0 else "Neutral"
                    })

            st.divider()
            st.subheader("Sector Comparison")
            display_data = []
            for r in results:
                display_data.append({
                    "Ticker": r["ticker"],
                    "Price": f"${r['current_price']:.2f}",
                    "High-Impact Prob": f"{r['pred_overnight']*100:.1f}%",
                    "EV": f"{r['ev_overnight']*100:.2f}%",
                    "News Score": f"{r['sentiment_score']*100:.0f}%"
                })
            st.table(pd.DataFrame(display_data))
        else:
            st.error("No significant alpha detected in this sector.")

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
                    row["Overnight EV"] = f"{r['ev_overnight']*100:.2f}%"
                    row["5-Day EV"] = f"{r['ev_5day']*100:.2f}%"
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
                profit_reason = f"This ticker has a **{best['ev_overnight']*100:.2f}%** Expected Value with a **{best['pred_overnight']*100:.1f}%** probability."
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

    st.divider()
    st.subheader("📊 Key Performance Metrics Explained")
    st.write("Understand the numbers behind the strategy to choose the most profitable trades.")

    c1, c2 = st.columns(2)
    with c1:
        st.info("### Sharpe Ratio")
        st.write("**What it is:** A measure of risk-adjusted return.")
        st.write("**Interpretation:** Higher is better. A Sharpe > 1.0 is good, > 2.0 is very good, and > 3.0 is professional grade. It tells you if your profits are worth the volatility you endure.")

        st.info("### Max Drawdown")
        st.write("**What it is:** The largest 'peak-to-valley' decline in your account value.")
        st.write("**Interpretation:** It shows the worst-case scenario. If a backtest shows a 20% drawdown, you should be prepared for your account to drop by that much at some point.")

    with c2:
        st.success("### Profit Factor")
        st.write("**What it is:** Total gross profit divided by total gross loss.")
        st.write("**Interpretation:** A value > 1.0 means the strategy is profitable. A Profit Factor of 2.0 means you make $2 for every $1 you lose.")

        st.success("### Win Rate")
        st.write("**What it is:** Percentage of trades that result in a profit.")
        st.write("**Interpretation:** Don't be fooled by high win rates alone! A 90% win rate can still lose money if the losses are huge. Combine this with Profit Factor for a real view.")

    st.divider()
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
