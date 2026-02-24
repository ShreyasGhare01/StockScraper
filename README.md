# 🚀 Stock AI Analyzer & Pump Detector

Welcome to the **Stock AI Analyzer & Pump Detector**, a comprehensive suite of tools designed to help traders identify "Strong Buy" signals, forecast price movements, and detect potential "Pump and Dump" activity using Machine Learning and Sentiment Analysis.

## 🌟 Key Features

- **Technical Analysis Engine:** High-performance indicator logic (RSI, MACD, EMA, etc.) implemented in pure `pandas` for maximum compatibility across Python versions.
- **AI Price Predictor:** A Random Forest Regressor trained on historical technical indicators to forecast the next day's closing price.
- **ML Pump Detector:** Identifies anomalous volume and price spikes using Z-score classification to alert you to potential market manipulation.
- **Sentiment & Research Scanner:** Scans Google News RSS feeds for "pump" related keywords and research to gauge market hype.
- **Option Strategy Suggester:** Recommends specific CALL or PUT strategies based on analysis sentiment and predicted volatility.
- **Visual Dashboard:** An interactive web interface built with Streamlit and Plotly for deep-dive analysis.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.9 through 3.14+** supported.
- Basic knowledge of terminal/command prompt usage.

### 2. Clone and Prepare Environment
Open your terminal and run the following:

```bash
# Navigate to the project directory
cd stock-analyzer-ai

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ IDE Setup Instructions

To get the most out of this project, we recommend using a professional IDE.

### **Visual Studio Code (VS Code)**
1. **Install Python Extension:** Search for "Python" in the Extensions view (`Ctrl+Shift+X`) and install the one by Microsoft.
2. **Open Folder:** Open the project folder in VS Code.
3. **Select Interpreter:** Press `Ctrl+Shift+P`, type "Python: Select Interpreter", and choose the one in your `./venv` folder.
4. **Running Scripts:** You can right-click any `.py` file and select "Run Python File in Terminal".

### **PyCharm**
1. **Open Project:** Select the project folder when starting PyCharm.
2. **Configure Interpreter:**
   - Go to `File > Settings > Project: stock-analyzer-ai > Python Interpreter`.
   - Click `Add Interpreter > Local Interpreter` and select the `python` executable inside your `venv/bin` (or `venv/Scripts`) folder.
3. **Run Configuration:** Right-click `main.py` or `dashboard.py` and select "Run".

---

## 🚀 How to Use

### **1. Visual Dashboard (Recommended)**
The dashboard provides the most intuitive experience with interactive charts.

```bash
streamlit run dashboard.py
```

**Troubleshooting:** If the `streamlit` command is not recognized, use:
```bash
python -m streamlit run dashboard.py
```

- A new tab will open in your browser (usually at `http://localhost:8501`).
- Enter a ticker (e.g., `NVDA`, `TSLA`, `AAPL`) in the sidebar.
- Click **Run Analysis** to see the AI's results.

### **2. Command Line Interface (CLI)**
For quick analysis without a browser:

```bash
python3 main.py [TICKER]
```
Example:
```bash
python3 main.py AAPL
```

---

## 🧪 Running Tests
To ensure everything is working correctly:

```bash
# Run integration tests
python3 tests/test_integration.py

# Run feature-specific tests
export PYTHONPATH=$PYTHONPATH:.
python3 tests/test_new_features.py
```

---

## ⚠️ Disclaimer
*This tool is for educational and research purposes only. Stock trading involves significant risk. Always consult with a certified financial advisor before making any investment decisions. The AI predictions are based on historical data and do not guarantee future results.*
