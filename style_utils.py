import streamlit as st

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:       #080b10;
    --surface:  #0d1117;
    --panel:    #111820;
    --border:   #1e2a38;
    --border2:  #243040;
    --accent:   #00e5ff;
    --accent2:  #00ff9d;
    --warn:     #ffb830;
    --danger:   #ff4560;
    --muted:    #4a6070;
    --text:     #ffffff;
    --text-dim: #e1eaf2;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }

  .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-weight: 700 !important;
  }

  /* Override Streamlit elements */
  .stSidebar {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  [data-testid="stHeader"] {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border);
  }

  h1, h2, h3 {
    font-family: var(--font-head) !important;
    color: #fff !important;
  }

  .stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--font-mono);
    font-weight: 700;
    border: none;
    text-transform: uppercase;
    letter-spacing: .08em;
    padding: 10px 20px;
  }

  .stButton > button:hover {
    background: #fff !important;
    box-shadow: 0 0 20px rgba(0,229,255,.4);
  }

  /* Custom UI Components */
  .topbar-logo {
    font-family: var(--font-head);
    font-weight: 800;
    font-size: 18px;
    letter-spacing: 0.04em;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
  }
  .topbar-logo .dot {
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
  }

  .metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    margin-bottom: 10px;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
  }
  .metric-card.green::before { background: var(--accent2); }
  .metric-card.warn::before { background: var(--warn); }
  .metric-card.danger::before { background: var(--danger); }

  .metric-label {
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text);
    font-weight: 800;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: var(--font-mono);
    font-size: 22px;
    color: #fff;
    font-weight: 700;
    letter-spacing: -.02em;
  }
  .metric-delta {
    font-family: var(--font-mono);
    font-size: 11px;
    margin-top: 3px;
  }
  .metric-delta.up { color: var(--accent2); }
  .metric-delta.down { color: var(--danger); }

  .info-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .panel-title {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 14px;
    color: #fff;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .panel-badge {
    font-family: var(--font-mono);
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: .07em;
  }
  .badge-green { background: rgba(0,255,157,.12); color: var(--accent2); border: 1px solid rgba(0,255,157,.25); }
  .badge-warn  { background: rgba(255,184,48,.12); color: var(--warn); border: 1px solid rgba(255,184,48,.25); }
  .badge-danger{ background: rgba(255,69,96,.12); color: var(--danger); border: 1px solid rgba(255,69,96,.25); }
  .badge-blue  { background: rgba(0,229,255,.1); color: var(--accent); border: 1px solid rgba(0,229,255,.2); }

  .json-block {
    background: #060c10;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent2);
    line-height: 1.7;
    white-space: pre-wrap;
  }

  .nav-section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: #fff;
    font-weight: 800;
    padding: 16px 0 6px;
  }

  .sidebar-stat {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .sidebar-stat-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: #fff;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 2px;
  }
  .sidebar-stat-val {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--accent2);
    font-weight: 700;
  }

  /* Custom Radio Group */
  div[role="radiogroup"] {
    flex-direction: row !important;
    gap: 10px;
  }

</style>
"""

def inject_styles():
    st.markdown(CSS, unsafe_allow_html=True)

def render_logo():
    st.sidebar.markdown('<div class="topbar-logo"><div class="dot"></div>AI STOCK SUITE PRO</div>', unsafe_allow_html=True)

def render_metric_card(label, value, delta="", color=""):
    delta_class = "up" if "+" in delta else "down" if "-" in delta else ""
    st.markdown(f"""
        <div class="metric-card {color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta}</div>
        </div>
    """, unsafe_allow_html=True)

def render_section_header(title, subtitle):
    st.markdown(f"""
        <div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border);">
            <div style="font-family: var(--font-head); font-weight: 800; font-size: 24px; color: #fff;">{title}</div>
            <div style="font-size: 13px; color: #fff; font-weight: 700; margin-top: 4px;">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar_stats(spy_val, vix_val):
    st.sidebar.markdown('<div class="nav-section-label">Market Pulse</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">S&P 500</div>
            <div class="sidebar-stat-val">{spy_val}</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">VIX</div>
            <div class="sidebar-stat-val" style="color:var(--warn)">{vix_val}</div>
        </div>
    """, unsafe_allow_html=True)

def render_json_strategy(data):
    import json
    json_str = json.dumps(data, indent=4)
    st.markdown(f'<div class="json-block">{json_str}</div>', unsafe_allow_html=True)
