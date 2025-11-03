import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")


# ----------------------- # Lấy tên công ty # -----------------------
def get_company_name(stock_code):
    stock_code = stock_code.strip().upper()
    try:
        t = yf.Ticker(stock_code + ".VN")
        info = {}
        try:
            info = t.info or {}
        except Exception:
            try:
                t2 = yf.Ticker(stock_code)
                info = t2.info or {}
            except Exception:
                info = {}
        if info:
            name = info.get("longName") or info.get("shortName") or info.get("companyShortName")
            if name and isinstance(name, str) and len(name.strip()) > 1:
                return name.strip()
    except Exception:
        pass
    try:
        url = f"https://www.hnx.vn/vi-vn/co-phieu-{stock_code}.html"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=8, verify=False)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            h1 = soup.find("h1")
            if h1 and h1.text.strip():
                return h1.text.strip()
            title = soup.find("title")
            if title and title.text.strip():
                return title.text.split("|")[0].strip()
    except Exception:
        pass
    return "Không tìm thấy tên công ty"


# ----------------------- # Tính MACD, RSI, EMA và mua tốt # -----------------------
def calculate_indicators(df, short=12, long=26, signal=9, rsi_period=14):
    df["EMA_short"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]  # RSI delta = df["Close"].diff()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Golden / Death Cross
    df["GC"] = (df["EMA_short"] > df["EMA_long"]) & (df["EMA_short"].shift(1) <= df["EMA_long"].shift(1))
    df["DC"] = (df["EMA_short"] < df["EMA_long"]) & (df["EMA_short"].shift(1) >= df["EMA_long"].shift(1))

    # Mua tốt: Golden Cross + RSI < 30
    df["BuySignal"] = df["GC"] & (df["RSI"] < 30)

    return df


# ----------------------- # Load dữ liệu # -----------------------
@st.cache_data(ttl=1800)
def load_stock_data(stock_code, period="6mo"):
    try:
        df = yf.download(f"{stock_code}.VN", period=period, progress=False)
        if df.empty:
            df = yf.download(stock_code, period=period, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Date" not in df.columns or "Close" not in df.columns:
        return None
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)
    df = calculate_indicators(df)
    return df


# ----------------------- # Dự báo Linear Regression # -----------------------
def linear_forecast(df, days_ahead=14):
    df_local = df.copy().reset_index(drop=True)
    df_local["t"] = np.arange(len(df_local))
    X = df_local[["t"]].values.reshape(-1, 1)
    y = df_local["Close"].values.ravel()
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(df_local), len(df_local) + days_ahead).reshape(-1, 1)
    preds = model.predict(future_t)
    future_dates = pd.date_range(df_local["Date"].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead, freq="B")
    forecast = pd.DataFrame({"Date": future_dates, "Predicted": preds})
    return forecast, model


# ----------------------- # Streamlit UI # -----------------------
st.set_page_config(page_title="Cổ phiếu + MACD + RSI + GC/DC", layout="wide")
st.title("📈 Phân tích & dự báo cổ phiếu (Nên mua, nên bán, nên giữ hay theo dõi thêm)")
stock_code = st.text_input("Nhập mã cổ phiếu:", "FPT").strip().upper()
period = st.selectbox("Khoảng thời gian dữ liệu:", ["3mo", "6mo", "1y"], index=1)
days_to_predict = st.slider("Số ngày dự đoán:", 5, 60, 14)

if st.button("🚀 Phân tích"):
    df = load_stock_data(stock_code, period)
    if df is None:
        st.error("Không tìm thấy dữ liệu cổ phiếu.")
    else:
        company_name = get_company_name(stock_code)
        st.subheader(f"{stock_code} — {company_name}")

        # Biểu đồ giá + EMA
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_short"], mode="lines", name="EMA12"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_long"], mode="lines", name="EMA26"))

        # đánh dấu Golden/Death Cross
        gc_dates = df.loc[df["GC"], "Date"]
        gc_prices = df.loc[df["GC"], "Close"]
        dc_dates = df.loc[df["DC"], "Date"]
        dc_prices = df.loc[df["DC"], "Close"]
        fig.add_trace(go.Scatter(x=gc_dates, y=gc_prices, mode="markers", name="Golden Cross",
                                 marker=dict(symbol="triangle-up", color="green", size=12)))
        fig.add_trace(go.Scatter(x=dc_dates, y=dc_prices, mode="markers", name="Death Cross",
                                 marker=dict(symbol="triangle-down", color="red", size=12)))

        # Mua tốt
        buy_dates = df.loc[df["BuySignal"], "Date"]
        buy_prices = df.loc[df["BuySignal"], "Close"]
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode="markers+text", name="Mua tốt",
                                 marker=dict(symbol="star", color="blue", size=15), text=["Mua tốt"] * len(buy_dates),
                                 textposition="top center"))
        fig.update_layout(title="Giá + EMA12/26 + Golden/Death Cross + Mua tốt", xaxis_title="Ngày", yaxis_title="Giá")
        st.plotly_chart(fig, use_container_width=True)

        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="blue")))
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["Signal"], name="Signal", line=dict(color="orange")))
        fig_macd.add_trace(go.Bar(x=df["Date"], y=df["Histogram"], name="Histogram", marker_color="gray"))
        fig_macd.update_layout(title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI", line=dict(color="green")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
        fig_rsi.update_layout(title="RSI (70=quá mua, 30=quá bán)")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Dự báo giá
        forecast, _ = linear_forecast(df, days_ahead=days_to_predict)
        last_price = df["Close"].iloc[-1]
        avg_pred = forecast["Predicted"].mean()
        diff_pct = (avg_pred - last_price) / last_price * 100
        trend = "📈 TĂNG" if diff_pct > 0 else "📉 GIẢM"
        st.markdown(f"""
        ### 🔍 Xu hướng dự báo
        - Giá hiện tại: **{last_price:,.2f}**
        - Giá trung bình {days_to_predict} ngày: **{avg_pred:,.2f}**
        - Chênh lệch: **{diff_pct:+.2f}%**
        - Xu hướng tổng thể: **{trend}**
        """)

        # Kết luận đầu tư
        latest_macd = df["MACD"].iloc[-1]
        latest_signal = df["Signal"].iloc[-1]
        latest_rsi = df["RSI"].iloc[-1]
        macd_bullish = latest_macd > latest_signal
        buy_signal_today = df["BuySignal"].iloc[-1]

        # Logic để xác định quyết định giữ, bán hoặc mua thêm
        if buy_signal_today:
            decision = "🚀 **MUA THÊM** (Golden Cross + RSI < 30)"
        elif macd_bullish and latest_rsi < 70:
            decision = "✅ **NÊN GIỮ** (Xu hướng tăng, chưa quá mua)"
        elif macd_bullish and latest_rsi >= 70:
            decision = "⚠️ **THEO DÕI** (RSI trên 70, có thể quá mua)"
        elif not macd_bullish or latest_rsi > 70:
            decision = "❌ **NÊN BÁN** (Xu hướng giảm hoặc quá mua)"
        elif latest_rsi < 30:
            decision = "⚠️ **CHỜ THEO DÕI** (RSI dưới 30, có thể đảo chiều tăng)"
        else:
            decision = "⏸️ **CHỜ THÊM TÍN HIỆU**"

        st.markdown(f"""
        ### 💡 Kết luận đầu tư:
        - Tín hiệu MACD: {"Tăng" if macd_bullish else "Giảm"}
        - RSI hiện tại: **{latest_rsi:.2f}**
        - **Quyết định:** {decision}
        """)
