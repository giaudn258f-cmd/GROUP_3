# -*- coding: utf-8 -*-
"""
Web App: Kiểm Định Tính Hiệu Quả Của Chiến Lược Đầu Tư Định Lượng
Dựa trên dữ liệu sàn HOSE (2020 - 2023)
Các chỉ số hiệu suất:
  1. Return (Tổng lợi suất & CAGR)
  2. Buy & Hold (Danh mục cơ sở & VN-Index)
  3. Winning Rate (Tỷ lệ phiên sinh lời)
  4. Volatility (Độ biến động quy năm)
  5. MDD (Mức sụt giảm tối đa - Max Drawdown)
  6. Sharpe Ratio (Lợi nhuận điều chỉnh theo tổng rủi ro)
  7. Sortino Ratio (Lợi nhuận điều chỉnh theo rủi ro giảm giá)
  8. Calmar Ratio (Tỷ số sinh lời / Sụt giảm tối đa)
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kiểm Định Chiến Lược Đầu Tư Định Lượng | HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách Modern FinTech
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
    }
    .metric-sub.negative {
        color: #EF4444;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
    }
    .highlight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HÀM XỬ LÝ DỮ LIỆU & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc và chuẩn hóa dữ liệu...")
def load_and_clean_data(file_source):
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None, f"File không tồn tại: {file_source}"
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)

    required_cols = ["date", "ticker", "close", "adj_close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"File CSV thiếu các cột bắt buộc: {missing}"

    df = df.dropna(subset=["date", "ticker", "close", "adj_close"]).copy()
    try:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"] != ""].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ticker", "date"]).reset_index(drop=True)

    return df, None

def rsi_wilder(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

@st.cache_data(show_spinner="Đang tính toán các chỉ báo kỹ thuật trên tập Train...")
def compute_indicators_df(train_stock_df):
    px_all = train_stock_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    val_all = (train_stock_df.assign(v=train_stock_df["close"] * train_stock_df["volume"])
               .pivot(index="date", columns="ticker", values="v").sort_index())

    enough = px_all.notna().sum() >= 0.90 * len(px_all)
    px_filtered = px_all.loc[:, enough].ffill()
    val_filtered = val_all[px_filtered.columns].fillna(0)

    if len(px_filtered) < 200:
        return None, px_filtered, "Tập Train cần có tối thiểu 200 phiên để tính chỉ báo SMA200."

    sma50 = px_filtered.rolling(50).mean()
    sma200 = px_filtered.rolling(200).mean()

    rsi_last = px_filtered.apply(lambda col: rsi_wilder(col).iloc[-1])
    vol_daily = px_filtered.pct_change(fill_method=None).std()
    mom_6m = px_filtered.iloc[-21] / px_filtered.iloc[-126] - 1

    ind = pd.DataFrame({
        "px_vs_sma50": px_filtered.iloc[-1] / sma50.iloc[-1] - 1,
        "sma50_vs_sma200": sma50.iloc[-1] / sma200.iloc[-1] - 1,
        "mom_6m": mom_6m,
        "rsi14": rsi_last,
        "vol_daily": vol_daily,
        "liq_value": val_filtered.tail(60).median()
    }).dropna()

    return ind, px_filtered, None

def apply_screening_and_scoring(ind, weights, n_liquid=30, rsi_max=75, max_corr=0.70, n_stocks=5, px_train=None):
    pool = ind.nlargest(min(n_liquid, len(ind)), "liq_value")
    ok_mask = (pool["rsi14"] <= rsi_max) & (pool["px_vs_sma50"] >= 0)
    eligible = pool[ok_mask].copy()
    removed = pool[~ok_mask].copy()

    if len(eligible) < n_stocks:
        eligible = pool.copy()

    r = lambda s: s.rank(pct=True)
    sc = pd.DataFrame(index=eligible.index)
    sc["trend"] = (r(eligible["px_vs_sma50"]) + r(eligible["sma50_vs_sma200"])) / 2
    sc["momentum"] = (r(eligible["mom_6m"]) + r(-(eligible["rsi14"] - 60).abs())) / 2
    sc["risk"] = r(-eligible["vol_daily"])
    sc["liquidity"] = r(eligible["liq_value"])

    total_w = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    sc["score"] = sum(sc[k] * (weights[k] / total_w) for k in weights)
    ranked = sc.sort_values("score", ascending=False)

    chosen = []
    if px_train is not None and not ranked.empty:
        ret_train = px_train[ranked.index].pct_change(fill_method=None)
        corr_matrix = ret_train.corr()

        for t in ranked.index:
            if all(corr_matrix.loc[t, c] <= max_corr for c in chosen):
                chosen.append(t)
            if len(chosen) == n_stocks:
                break

        if len(chosen) < n_stocks:
            remaining = [t for t in ranked.index if t not in chosen]
            chosen += remaining[:(n_stocks - len(chosen))]
    else:
        chosen = ranked.index[:n_stocks].tolist()

    return eligible, removed, ranked, chosen

# -----------------------------------------------------------------------------
# 3. TỐI ƯU HÓA DANH MỤC: LEDOIT-WOLF SHRINKAGE MIN-VOLATILITY
# -----------------------------------------------------------------------------
def optimize_shrinkage_min_vol(px_train, tickers, max_weight=0.40):
    sub_px = px_train[tickers].ffill().dropna()
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    try:
        from pypfopt import risk_models, EfficientFrontier
        S_shrink = risk_models.CovarianceShrinkage(sub_px).ledoit_wolf()
        ef = EfficientFrontier(None, S_shrink, weight_bounds=(0.0, max_weight))
        raw_w = ef.min_volatility()
        cleaned_w = ef.clean_weights()
        return {t: float(cleaned_w.get(t, 0.0)) for t in tickers}
    except Exception:
        pass

    try:
        from scipy.optimize import minimize
        returns = sub_px.pct_change(fill_method=None).dropna().values
        T, p = returns.shape
        mean_ret = np.mean(returns, axis=0)
        X = returns - mean_ret

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(store_precision=False, assume_centered=True)
            cov_matrix = lw.fit(X).covariance_
        except Exception:
            S = np.dot(X.T, X) / T
            m = np.trace(S) / p
            d2 = np.linalg.norm(S - m * np.identity(p), "fro") ** 2
            b_bar2 = 0
            for i in range(T):
                Xi = X[i, :, np.newaxis]
                b_bar2 += np.linalg.norm(np.dot(Xi, Xi.T) - S, "fro") ** 2
            b2 = min(b_bar2 / (T ** 2), d2)
            shrinkage = b2 / d2 if d2 > 0 else 0
            cov_matrix = shrinkage * (m * np.identity(p)) + (1 - shrinkage) * S

        def obj(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, max_weight) for _ in range(n))
        init_guess = np.full(n, 1.0 / n)

        res = minimize(obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            w_opt = res.x
            w_opt = np.maximum(0, w_opt)
            w_opt = w_opt / np.sum(w_opt)
            return {t: float(w_opt[i]) for i, t in enumerate(tickers)}
    except Exception:
        pass

    return {t: 1.0 / n for t in tickers}

# -----------------------------------------------------------------------------
# 4. TÍNH TOÁN HIỆU SUẤT & 8 CHỈ SỐ
# -----------------------------------------------------------------------------
def calculate_all_metrics(equity_series, rf_annual=0.03, trading_days=252, initial_capital=100_000_000):
    eq = equity_series.dropna()
    if len(eq) < 2:
        return {
            "Total Return": 0.0, "CAGR": 0.0, "Winning Rate": 0.0,
            "Win Days": 0, "Loss Days": 0, "Total Days": 0,
            "Volatility": 0.0, "Max Drawdown": 0.0, "Sharpe": 0.0,
            "Sortino": 0.0, "Calmar": 0.0, "Final Equity": initial_capital,
            "Net Profit": 0.0
        }

    daily_ret = eq.pct_change(fill_method=None).dropna()
    n_days = len(daily_ret)

    # 1. Return
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    if n_days > 0 and eq.iloc[-1] > 0 and eq.iloc[0] > 0:
        cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (trading_days / n_days)) - 1.0
    else:
        cagr = np.nan

    # 3. Winning Rate
    win_days = int((daily_ret > 0).sum())
    loss_days = int((daily_ret < 0).sum())
    active_days = win_days + loss_days
    win_rate = (win_days / n_days) if n_days > 0 else 0.0
    win_rate_active = (win_days / active_days) if active_days > 0 else 0.0

    # 4. Volatility
    volatility = daily_ret.std(ddof=1) * np.sqrt(trading_days) if len(daily_ret) > 1 else 0.0

    # 5. Max Drawdown
    cummax = eq.cummax()
    drawdown = (eq / cummax) - 1.0
    max_dd = float(drawdown.min())

    # 6. Sharpe Ratio
    excess_ret = daily_ret.mean() * trading_days - rf_annual
    sharpe = (excess_ret / volatility) if (volatility > 0) else np.nan

    # 7. Sortino Ratio
    downside_returns = daily_ret[daily_ret < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(trading_days) if len(downside_returns) > 0 else 0.0
    sortino = (excess_ret / downside_std) if (downside_std > 0) else np.nan

    # 8. Calmar Ratio
    calmar = (cagr / abs(max_dd)) if (max_dd < 0 and not np.isnan(cagr)) else np.nan

    final_equity = float(eq.iloc[-1])
    net_profit = final_equity - initial_capital

    return {
        "Total Return": total_return, "CAGR": cagr, "Winning Rate": win_rate,
        "Winning Rate Active": win_rate_active, "Win Days": win_days,
        "Loss Days": loss_days, "Total Days": n_days, "Volatility": volatility,
        "Max Drawdown": max_dd, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar, "Final Equity": final_equity, "Net Profit": net_profit,
        "Drawdown Series": drawdown
    }

# -----------------------------------------------------------------------------
# 5. BACKTESTING ENGINE
# -----------------------------------------------------------------------------
def run_simulation(px_df, weights_dict, signal_series=None, initial_capital=100_000_000, cost=0.0015):
    tickers = list(weights_dict.keys())
    sub_px = px_df[tickers].ffill().dropna()

    if sub_px.empty:
        return pd.Series(dtype=float)

    daily_ret = sub_px.pct_change(fill_method=None).fillna(0)
    w_vec = np.array([weights_dict[t] for t in tickers])
    base_ret = daily_ret.dot(w_vec)

    if signal_series is not None:
        sig_aligned = signal_series.reindex(sub_px.index).fillna(0)
        strategy_ret = base_ret * sig_aligned
    else:
        strategy_ret = base_ret

    equity = initial_capital * (1.0 - cost) * (1.0 + strategy_ret).cumprod()
    return equity

def run_vnindex_simulation(benchmark_df, start_date, end_date, initial_capital=100_000_000):
    sub_b = benchmark_df.set_index("date")["close"].sort_index()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    sub_b = sub_b.loc[(sub_b.index >= start_dt) & (sub_b.index <= end_dt)].dropna()
    if sub_b.empty:
        return pd.Series(dtype=float)
    eq = initial_capital * (sub_b / sub_b.iloc[0])
    return eq

# -----------------------------------------------------------------------------
# 6. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu Hình Hệ Thống")

st.sidebar.subheader("1. Dữ Liệu Thị Trường")
data_option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mặc định (HOSE 2020-2023)", "Tải lên file CSV mới"],
    index=0
)

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "HOSE_2020_2023_in.csv")
if not os.path.exists(csv_path):
    csv_path = "HOSE_2020_2023_in.csv"

uploaded_file = None

if data_option == "Tải lên file CSV mới":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is None:
        st.sidebar.info("Vui lòng tải lên file CSV để tiếp tục.")

if uploaded_file is not None:
    raw_df, err = load_and_clean_data(uploaded_file)
else:
    raw_df, err = load_and_clean_data(csv_path)

if err or raw_df is None:
    st.error(f"Lỗi đọc dữ liệu: {err}")
    st.stop()

benchmark_full = raw_df[raw_df["ticker"] == "VNINDEX"].copy()
stock_full = raw_df[raw_df["ticker"] != "VNINDEX"].copy()

min_date = raw_df["date"].min().date()
max_date = raw_df["date"].max().date()

# Hàm kẹp ngày an toàn chống lỗi StreamlitValueBelowMinError / StreamlitValueAboveMaxError
def clamp_date(target, min_d, max_d):
    return max(min_d, min(max_d, target))

default_train_start = clamp_date(datetime(2020, 1, 1).date(), min_date, max_date)
default_train_end = clamp_date(datetime(2021, 12, 31).date(), min_date, max_date)
default_test_start = clamp_date(datetime(2022, 1, 1).date(), min_date, max_date)
default_test_end = clamp_date(datetime(2022, 12, 31).date(), min_date, max_date)

st.sidebar.subheader("2. Phân Chia Thời Gian")
col_s1, col_s2 = st.sidebar.columns(2)
train_start = col_s1.date_input("Train Bắt đầu", default_train_start, min_value=min_date, max_value=max_date)
train_end = col_s2.date_input("Train Kết thúc", default_train_end, min_value=min_date, max_value=max_date)

col_s3, col_s4 = st.sidebar.columns(2)
test_start = col_s3.date_input("Test Bắt đầu", default_test_start, min_value=min_date, max_value=max_date)
test_end = col_s4.date_input("Test Kết thúc", default_test_end, min_value=min_date, max_value=max_date)

if train_end >= test_start:
    st.sidebar.warning("Lưu ý: Thời gian Train nên trước thời gian Test để chống Look-ahead Bias!")

st.sidebar.subheader("3. Bộ Lọc Cổ Phiếu")
mode_selection = st.sidebar.radio(
    "Chế độ chọn danh mục:",
    ["Tự động theo thuật toán Notebook", "Người dùng tự chọn thủ công"],
    index=0
)

w_trend = st.sidebar.slider("Xu hướng (Trend)", 0, 100, 30, 5)
w_mom = st.sidebar.slider("Động lượng (Momentum)", 0, 100, 30, 5)
w_risk = st.sidebar.slider("Rủi ro thấp (Low Risk)", 0, 100, 20, 5)
w_liq = st.sidebar.slider("Thanh khoản (Liquidity)", 0, 100, 20, 5)

weights_factor = {
    "trend": w_trend / 100.0,
    "momentum": w_mom / 100.0,
    "risk": w_risk / 100.0,
    "liquidity": w_liq / 100.0
}

n_stocks_input = st.sidebar.number_input("Số cổ phiếu trong danh mục (N):", min_value=2, max_value=20, value=5)
n_liquid_input = st.sidebar.number_input("Top thanh khoản ban đầu:", min_value=10, max_value=100, value=30)
rsi_max_input = st.sidebar.slider("Ngưỡng RSI tối đa:", 50, 90, 75, 5)
max_corr_input = st.sidebar.slider("Ngưỡng tương quan tối đa:", 0.3, 0.95, 0.70, 0.05)

st.sidebar.subheader("4. Tham Số Giao Dịch & Mô Hình")
initial_capital = st.sidebar.number_input("Vốn ban đầu (VND):", min_value=10_000_000, value=100_000_000, step=10_000_000, format="%d")
rf_rate = st.sidebar.number_input("Lãi suất phi rủi ro Rf (%/năm):", min_value=0.0, max_value=20.0, value=3.0, step=0.5) / 100.0
trading_cost = st.sidebar.number_input("Phí giao dịch ban đầu (%):", min_value=0.0, max_value=2.0, value=0.15, step=0.05) / 100.0
max_stock_weight = st.sidebar.slider("Tỷ trọng tối đa 1 mã (Shrinkage bound):", 0.20, 1.00, 0.40, 0.05)
sma_timing_window = st.sidebar.number_input("Chu kỳ SMA Market Timing (phiên):", min_value=20, max_value=300, value=200, step=10)

# -----------------------------------------------------------------------------
# 7. XỬ LÝ LỌC & TÍNH TOÁN
# -----------------------------------------------------------------------------
train_stocks = stock_full[(stock_full["date"] >= pd.to_datetime(train_start)) & (stock_full["date"] <= pd.to_datetime(train_end))].copy()
test_stocks = stock_full[(stock_full["date"] >= pd.to_datetime(test_start)) & (stock_full["date"] <= pd.to_datetime(test_end))].copy()

if train_stocks.empty:
    st.error("Tập Train không có dữ liệu giao dịch! Vui lòng chọn lại khoảng thời gian Train.")
    st.stop()

if test_stocks.empty:
    st.error("Tập Test không có dữ liệu giao dịch! Vui lòng chọn lại khoảng thời gian Test.")
    st.stop()

indicators_df, px_train_all, ind_err = compute_indicators_df(train_stocks)

if ind_err:
    st.warning(ind_err)
    st.stop()

eligible_stocks, removed_stocks, ranked_stocks, default_top = apply_screening_and_scoring(
    indicators_df, weights_factor,
    n_liquid=int(n_liquid_input), rsi_max=rsi_max_input,
    max_corr=max_corr_input, n_stocks=int(n_stocks_input),
    px_train=px_train_all
)

if mode_selection == "Người dùng tự chọn thủ công":
    all_available = sorted(list(px_train_all.columns))
    selected_tickers = st.sidebar.multiselect(
        "Chọn các mã cổ phiếu:",
        all_available,
        default=default_top if len(default_top) <= len(all_available) else all_available[:n_stocks_input]
    )
    if len(selected_tickers) < 2:
        st.sidebar.warning("Vui lòng chọn tối thiểu 2 mã.")
        selected_tickers = default_top
else:
    selected_tickers = default_top
    st.sidebar.success(f"Top {len(selected_tickers)} được chọn: {', '.join(selected_tickers)}")

px_train_selected = train_stocks[train_stocks["ticker"].isin(selected_tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()
px_test_selected = test_stocks[test_stocks["ticker"].isin(selected_tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()

px_train_selected = px_train_selected.loc[:, px_train_selected.iloc[0].notna()]
px_test_selected = px_test_selected.loc[:, px_test_selected.iloc[0].notna()]
valid_tickers = [t for t in selected_tickers if t in px_train_selected.columns and t in px_test_selected.columns]

if len(valid_tickers) < 2:
    st.error("Không đủ cổ phiếu có dữ liệu xuyên suốt Train và Test để chạy kiểm định.")
    st.stop()

# -----------------------------------------------------------------------------
# 8. TÍNH TỶ TRỌNG & TÍN HIỆU MARKET TIMING
# -----------------------------------------------------------------------------
w_equal = {t: 1.0 / len(valid_tickers) for t in valid_tickers}
w_shrink = optimize_shrinkage_min_vol(px_train_selected, valid_tickers, max_weight=max_stock_weight)

vnindex_series = benchmark_full.set_index("date")["close"].sort_index()
vnindex_sma = vnindex_series.rolling(window=sma_timing_window).mean()
market_signal = (vnindex_series > vnindex_sma).astype(int).shift(1).fillna(0)

# Mô phỏng Equity
eq_train_baseline = run_simulation(px_train_selected, w_equal, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_train_shrink = run_simulation(px_train_selected, w_shrink, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_train_mt = run_simulation(px_train_selected, w_equal, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_train_combined = run_simulation(px_train_selected, w_shrink, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_train_vnindex = run_vnindex_simulation(benchmark_full, train_start, train_end, initial_capital=initial_capital)

eq_test_baseline = run_simulation(px_test_selected, w_equal, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_test_shrink = run_simulation(px_test_selected, w_shrink, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_test_mt = run_simulation(px_test_selected, w_equal, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_test_combined = run_simulation(px_test_selected, w_shrink, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_test_vnindex = run_vnindex_simulation(benchmark_full, test_start, test_end, initial_capital=initial_capital)

metrics_map = {
    "Train": {
        "Baseline (Buy & Hold 1/N)": calculate_all_metrics(eq_train_baseline, rf_rate, initial_capital=initial_capital),
        "Shrinkage Min-Vol": calculate_all_metrics(eq_train_shrink, rf_rate, initial_capital=initial_capital),
        "Market Timing (SMA)": calculate_all_metrics(eq_train_mt, rf_rate, initial_capital=initial_capital),
        "Kết Hợp (Shrinkage + MT)": calculate_all_metrics(eq_train_combined, rf_rate, initial_capital=initial_capital),
        "VNINDEX Benchmark": calculate_all_metrics(eq_train_vnindex, rf_rate, initial_capital=initial_capital)
    },
    "Test": {
        "Baseline (Buy & Hold 1/N)": calculate_all_metrics(eq_test_baseline, rf_rate, initial_capital=initial_capital),
        "Shrinkage Min-Vol": calculate_all_metrics(eq_test_shrink, rf_rate, initial_capital=initial_capital),
        "Market Timing (SMA)": calculate_all_metrics(eq_test_mt, rf_rate, initial_capital=initial_capital),
        "Kết Hợp (Shrinkage + MT)": calculate_all_metrics(eq_test_combined, rf_rate, initial_capital=initial_capital),
        "VNINDEX Benchmark": calculate_all_metrics(eq_test_vnindex, rf_rate, initial_capital=initial_capital)
    }
}

# -----------------------------------------------------------------------------
# 9. GIAO DIỆN CHÍNH
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 Hệ Thống Kiểm Định Chiến Lược Đầu Tư Định Lượng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Nghiên cứu kiểm chứng mô hình Lựa Chọn Cổ Phiếu Đa Nhân Tố, Tối Ưu Hóa Rủi Ro Co-variance Shrinkage & Định Thời Điểm Thị Trường (Market Timing) trên sàn HOSE</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Dữ Liệu & Phân Chia",
    "🎯 2. Lọc & Xếp Hạng Mã",
    "🚀 3. Kiểm Định 8 Chỉ Số & Hiệu Suất",
    "⚖️ 4. Phân Bổ Tỷ Trọng Danh Mục",
    "📑 5. Báo Cáo Chuyên Sâu & Xuất Dữ Liệu"
])

with tab1:
    st.subheader("1. Tổng Quan Dữ Liệu Thị Trường")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Số Dòng Dữ Liệu", f"{len(raw_df):,}")
    c2.metric("Số Mã Cổ Phiếu (HOSE)", f"{stock_full['ticker'].nunique():,} mã")
    c3.metric("Khoảng Thời Gian", f"{min_date.strftime('%d/%m/%Y')} → {max_date.strftime('%d/%m/%Y')}")
    c4.metric("Dữ Liệu Tham Chiếu", "Chỉ số VN-INDEX")

    st.markdown("---")
    st.subheader("2. Chi Tiết Phân Chia Tập Train - Test")
    split_summary = pd.DataFrame({
        "Giai đoạn": ["Tập TRAIN (In-Sample)", "Tập TEST (Out-of-Sample)"],
        "Bối cảnh": ["Tăng trưởng mạnh (Uptrend)", "Điều chỉnh sâu (Downtrend)"],
        "Ngày bắt đầu": [train_start.strftime('%d/%m/%Y'), test_start.strftime('%d/%m/%Y')],
        "Ngày kết thúc": [train_end.strftime('%d/%m/%Y'), test_end.strftime('%d/%m/%Y')],
        "Số phiên": [train_stocks["date"].nunique(), test_stocks["date"].nunique()],
        "Số mã": [train_stocks["ticker"].nunique(), test_stocks["ticker"].nunique()]
    })
    st.dataframe(split_summary, use_container_width=True, hide_index=True)

    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(x=benchmark_full["date"], y=benchmark_full["close"], mode="lines", name="VNINDEX", line=dict(color="#1E3A8A", width=2)))
    fig_overview.add_trace(go.Scatter(x=vnindex_sma.index, y=vnindex_sma.values, mode="lines", name=f"SMA {sma_timing_window}", line=dict(color="#F59E0B", width=1.5, dash="dash")))
    fig_overview.add_vrect(x0=str(train_start), x1=str(train_end), fillcolor="rgba(16, 185, 129, 0.12)", layer="below", line_width=0, annotation_text="TRAIN (Uptrend)")
    fig_overview.add_vrect(x0=str(test_start), x1=str(test_end), fillcolor="rgba(239, 68, 68, 0.12)", layer="below", line_width=0, annotation_text="TEST (Downtrend)")
    fig_overview.update_layout(title="Biến Động Chỉ Số VNINDEX & SMA200 (2020 - 2023)", xaxis_title="Thời gian", yaxis_title="Điểm số VN-Index", template="plotly_white", height=450)
    st.plotly_chart(fig_overview, use_container_width=True)

with tab2:
    st.subheader(f"1. Chấm Điểm Đa Nhân Tố & Chọn Top {n_stocks_input}")
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.write("##### Bảng Xếp Hạng Điểm Tổng Hợp")
        display_ranked = ranked_stocks.copy()
        display_ranked["Điểm Tổng"] = display_ranked["score"]
        display_ranked["Được Chọn"] = display_ranked.index.isin(valid_tickers)
        st.dataframe(
            display_ranked[["score", "trend", "momentum", "risk", "liquidity", "Được Chọn"]],
            column_config={
                "score": st.column_config.ProgressColumn(
                    "Điểm Tổng",
                    help="Điểm tổng hợp đa nhân tố chuẩn hóa (0.0 - 1.0)",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "trend": st.column_config.NumberColumn("Xu Hướng", format="%.3f"),
                "momentum": st.column_config.NumberColumn("Động Lượng", format="%.3f"),
                "risk": st.column_config.NumberColumn("Rủi Ro Thấp", format="%.3f"),
                "liquidity": st.column_config.NumberColumn("Thanh Khoản", format="%.3f"),
                "Được Chọn": st.column_config.CheckboxColumn("Được Chọn", help="Mã được chọn vào danh mục")
            },
            use_container_width=True,
            height=360
        )
    with col_t2:
        st.write(f"##### Điểm Tổng Hợp Top {len(valid_tickers)}")
        top_scores = ranked_stocks.loc[valid_tickers, "score"].sort_values(ascending=True)
        fig_bar = px.bar(x=top_scores.values, y=top_scores.index, orientation='h', labels={"x": "Điểm", "y": "Mã CP"}, color=top_scores.values, color_continuous_scale="Blues")
        fig_bar.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.write("##### Ma Trận Tương Quan Lợi Suất Ngày (Train)")
    ret_selected = px_train_selected.pct_change(fill_method=None).dropna()
    fig_corr = px.imshow(ret_selected.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn_r", zmin=0.0, zmax=1.0)
    fig_corr.update_layout(title=f"Tương Quan Chéo (Lọc Đa Dạng Hóa <= {max_corr_input})", height=380)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.subheader("1. Bảng Điều Khiển 8 Chỉ Số Hiệu Suất")
    phase_choice = st.radio("Chọn giai đoạn kiểm tra:", ["Tập TEST (Out-of-sample / 2022)", "Tập TRAIN (In-sample / 2020-2021)"], horizontal=True)
    active_phase = "Test" if "TEST" in phase_choice else "Train"

    p_metrics = metrics_map[active_phase]
    best_strategy_name = "Kết Hợp (Shrinkage + MT)"
    best_m = p_metrics[best_strategy_name]
    baseline_m = p_metrics["Baseline (Buy & Hold 1/N)"]

    st.markdown(f"#### 8 Chỉ Số Hiệu Suất Của Chiến Lược [{best_strategy_name}] ({active_phase})")
    kpi_cols1 = st.columns(4)
    kpi_cols1[0].metric("1. RETURN (Tổng lợi suất)", f"{best_m['Total Return']:.2%}", delta=f"{(best_m['Total Return'] - baseline_m['Total Return']):+.2%} vs B&H")
    kpi_cols1[1].metric("2. BUY & HOLD (Baseline)", f"{baseline_m['Total Return']:.2%}", delta=f"{(baseline_m['Total Return'] - p_metrics['VNINDEX Benchmark']['Total Return']):+.2%} vs VNINDEX")
    kpi_cols1[2].metric("3. WINNING RATE (Tỷ lệ thắng)", f"{best_m['Winning Rate']:.2%}", help=f"{best_m['Win Days']} thắng / {best_m['Loss Days']} thua")
    kpi_cols1[3].metric("4. VOLATILITY (Biến động năm)", f"{best_m['Volatility']:.2%}", delta=f"{(best_m['Volatility'] - baseline_m['Volatility']):+.2%} vs B&H", delta_color="inverse")

    kpi_cols2 = st.columns(4)
    kpi_cols2[0].metric("5. MDD (Sụt giảm tối đa)", f"{best_m['Max Drawdown']:.2%}", delta=f"{(best_m['Max Drawdown'] - baseline_m['Max Drawdown']):+.2%} vs B&H")
    kpi_cols2[1].metric("6. SHARPE RATIO (Rf=3%)", f"{best_m['Sharpe']:.2f}" if not np.isnan(best_m['Sharpe']) else "N/A")
    kpi_cols2[2].metric("7. SORTINO RATIO", f"{best_m['Sortino']:.2f}" if not np.isnan(best_m['Sortino']) else "N/A")
    kpi_cols2[3].metric("8. CALMAR RATIO (CAGR/MDD)", f"{best_m['Calmar']:.2f}" if not np.isnan(best_m['Calmar']) else "N/A")

    st.markdown("---")
    st.subheader(f"2. Bảng So Sánh Toàn Diện Các Chiến Lược ({active_phase})")
    comparison_rows = []
    for s_name, s_m in p_metrics.items():
        comparison_rows.append({
            "Chiến Lược": s_name,
            "1. Tổng Return": f"{s_m['Total Return']:.2%}",
            "Lợi suất/Năm (CAGR)": f"{s_m['CAGR']:.2%}" if not np.isnan(s_m['CAGR']) else "N/A",
            "3. Winning Rate": f"{s_m['Winning Rate']:.2%}",
            "4. Volatility (Năm)": f"{s_m['Volatility']:.2%}",
            "5. Max Drawdown": f"{s_m['Max Drawdown']:.2%}",
            "6. Sharpe": f"{s_m['Sharpe']:.2f}" if not np.isnan(s_m['Sharpe']) else "N/A",
            "7. Sortino": f"{s_m['Sortino']:.2f}" if not np.isnan(s_m['Sortino']) else "N/A",
            "8. Calmar": f"{s_m['Calmar']:.2f}" if not np.isnan(s_m['Calmar']) else "N/A",
            "Vốn Cuối Kỳ (VND)": f"{s_m['Final Equity']:,.0f}"
        })
    st.dataframe(pd.DataFrame(comparison_rows).set_index("Chiến Lược"), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Biểu Đồ Tăng Trưởng Tài Sản & Mức Sụt Giảm")
    chart_eqs = {
        "Baseline (Buy & Hold 1/N)": eq_test_baseline if active_phase == "Test" else eq_train_baseline,
        "Shrinkage Min-Vol": eq_test_shrink if active_phase == "Test" else eq_train_shrink,
        "Market Timing (SMA)": eq_test_mt if active_phase == "Test" else eq_train_mt,
        "Kết Hợp (Shrinkage + MT)": eq_test_combined if active_phase == "Test" else eq_train_combined,
        "VNINDEX Benchmark": eq_test_vnindex if active_phase == "Test" else eq_train_vnindex
    }
    colors = {"Baseline (Buy & Hold 1/N)": "#EF4444", "Shrinkage Min-Vol": "#8B5CF6", "Market Timing (SMA)": "#F59E0B", "Kết Hợp (Shrinkage + MT)": "#10B981", "VNINDEX Benchmark": "#1E3A8A"}
    
    fig_equity = go.Figure()
    for name, s_eq in chart_eqs.items():
        if not s_eq.empty:
            fig_equity.add_trace(go.Scatter(x=s_eq.index, y=s_eq / 1_000_000, mode="lines", name=name, line=dict(color=colors.get(name, "#000"), width=3 if "Kết Hợp" in name else 1.8)))
    fig_equity.add_hline(y=initial_capital / 1_000_000, line=dict(color="gray", dash="dash"), annotation_text="Vốn ban đầu")
    fig_equity.update_layout(title=f"Đường Phát Triển Vốn (Equity Curve) - Giai đoạn {active_phase}", xaxis_title="Thời gian", yaxis_title="Triệu VNĐ", template="plotly_white", height=480)
    st.plotly_chart(fig_equity, use_container_width=True)

    fig_dd = go.Figure()
    for name, s_eq in chart_eqs.items():
        if not s_eq.empty:
            dd = (s_eq / s_eq.cummax() - 1.0) * 100
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=name, line=dict(color=colors.get(name, "#000"), width=2)))
    fig_dd.update_layout(title="Đồ Thị Sụt Giảm Từ Đỉnh (Underwater Drawdown %)", xaxis_title="Thời gian", yaxis_title="Mức sụt giảm (%)", template="plotly_white", height=380)
    st.plotly_chart(fig_dd, use_container_width=True)

with tab4:
    st.subheader("1. So Sánh Cơ Cấu Tỷ Trọng Danh Mục")
    df_weights = pd.DataFrame({
        "Mã Cổ Phiếu": valid_tickers,
        "Equal Weight (1/N)": [w_equal[t] for t in valid_tickers],
        "Shrinkage Min-Vol": [w_shrink.get(t, 0.0) for t in valid_tickers]
    })
    col_w1, col_w2 = st.columns([2, 3])
    with col_w1:
        st.dataframe(df_weights.style.format({"Equal Weight (1/N)": "{:.2%}", "Shrinkage Min-Vol": "{:.2%}"}), use_container_width=True, hide_index=True)
    with col_w2:
        fig_pie = px.bar(df_weights.melt(id_vars=["Mã Cổ Phiếu"], var_name="Mô hình", value_name="Tỷ trọng"), x="Mã Cổ Phiếu", y="Tỷ trọng", color="Mô hình", barmode="group", text_auto=".1%")
        fig_pie.update_layout(height=350, yaxis_tickformat=".0%")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab5:
    st.subheader("1. Đối Chiếu Kết Quả Train (In-Sample) vs Test (Out-of-Sample)")
    metric_names = ["Tổng Lợi Suất (Return)", "Lợi Suất Quy Năm (CAGR)", "Tỷ Lệ Thắng (Winning Rate)", "Biến Động Quy Năm (Volatility)", "Mức Sụt Giảm Tối Đa (MDD)", "Chỉ Số Sharpe (Rf=3%)", "Chỉ Số Sortino", "Chỉ Số Calmar"]
    strategies_to_show = ["Baseline (Buy & Hold 1/N)", "Shrinkage Min-Vol", "Market Timing (SMA)", "Kết Hợp (Shrinkage + MT)"]
    final_report_data = {"Chỉ Số": metric_names}
    for s in strategies_to_show:
        for phase in ["Train", "Test"]:
            m = metrics_map[phase][s]
            final_report_data[f"{s} ({phase.upper()})"] = [m["Total Return"], m["CAGR"], m["Winning Rate"], m["Volatility"], m["Max Drawdown"], m["Sharpe"], m["Sortino"], m["Calmar"]]
    df_final_report = pd.DataFrame(final_report_data).set_index("Chỉ Số")
    st.dataframe(df_final_report.style.format(lambda val: f"{val:.2f}" if isinstance(val, (int, float)) and abs(val) < 10 and not np.isnan(val) and abs(val) > 0.0001 and val != 0 else (f"{val:.2%}" if isinstance(val, (int, float)) and not np.isnan(val) else "N/A")), use_container_width=True)

    csv_buffer = io.StringIO()
    df_final_report.to_csv(csv_buffer)
    st.download_button("📥 Tải Xuống Báo Cáo Hiệu Suất (CSV)", data=csv_buffer.getvalue().encode('utf-8-sig'), file_name="Bao_Cao_Kiem_Dinh_Chien_Luoc_Dau_Tu.csv", mime="text/csv")
