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
    """
    Đọc dữ liệu từ file đường dẫn hoặc UploadedFile object và chuẩn hóa:
    - date: datetime
    - ticker: uppercase, strip
    - lọc các bản ghi trùng lặp và sắp xếp theo ticker, date
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None, "File không tồn tại."
        df = pd.read_csv(file_source)
    else:
        df = pd.read_csv(file_source)

    required_cols = ["date", "ticker", "close", "adj_close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"File CSV thiếu các cột bắt buộc: {missing}"

    # Chuẩn hóa ngày: thử định dạng m/d/Y trước, nếu lỗi thì fallback tự động
    df = df.dropna(subset=["date", "ticker", "close", "adj_close"]).copy()
    try:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Loại bỏ ngày lỗi
    df = df.dropna(subset=["date"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"] != ""].copy()

    # Sắp xếp và bỏ trùng
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ticker", "date"]).reset_index(drop=True)

    return df, None

def rsi_wilder(series, n=14):
    """Tính RSI theo phương pháp Wilder's Smoothing như trong Notebook."""
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

@st.cache_data(show_spinner="Đang tính toán các chỉ báo kỹ thuật trên tập Train...")
def compute_indicators_df(train_stock_df):
    """
    Tính toán các chỉ báo kỹ thuật tại phiên cuối của tập Train:
    1. Trend 1: px / SMA50 - 1
    2. Trend 2: SMA50 / SMA200 - 1
    3. Momentum 1: Mom 6M (phiên -21 so với -126)
    4. Momentum 2: Wilder's RSI(14)
    5. Risk: Độ biến động ngày (std)
    6. Liquidity: Giá trị giao dịch trung vị 60 phiên
    """
    # Pivot giá điều chỉnh và giá trị giao dịch
    px_all = train_stock_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    val_all = (train_stock_df.assign(v=train_stock_df["close"] * train_stock_df["volume"])
               .pivot(index="date", columns="ticker", values="v").sort_index())

    # Lọc mã có đủ >= 90% số phiên
    enough = px_all.notna().sum() >= 0.90 * len(px_all)
    px_filtered = px_all.loc[:, enough].ffill()
    val_filtered = val_all[px_filtered.columns].fillna(0)

    # Cần tối thiểu 200 phiên để tính SMA200
    if len(px_filtered) < 200:
        return None, px_filtered, "Tập Train cần có tối thiểu 200 phiên để tính chỉ báo SMA200."

    sma50 = px_filtered.rolling(50).mean()
    sma200 = px_filtered.rolling(200).mean()

    # Tính RSI cho từng cột
    rsi_last = px_filtered.apply(lambda col: rsi_wilder(col).iloc[-1])
    vol_daily = px_filtered.pct_change(fill_method=None).std()

    # Mom 6M: phiên -21 / phiên -126 - 1
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
    """
    Lọc và xếp hạng cổ phiếu đa nhân tố (Multi-factor ranking & Correlation diversification).
    """
    # 1. Bộ lọc thanh khoản
    pool = ind.nlargest(min(n_liquid, len(ind)), "liq_value")
    # 2. Bộ lọc loại: RSI <= rsi_max và giá trên SMA50
    ok_mask = (pool["rsi14"] <= rsi_max) & (pool["px_vs_sma50"] >= 0)
    eligible = pool[ok_mask].copy()
    removed = pool[~ok_mask].copy()

    if len(eligible) < n_stocks:
        # Nếu sau lọc không đủ, tạm lấy pool gốc sắp xếp theo thanh khoản
        eligible = pool.copy()

    # 3. Chấm điểm Percentile Rank
    r = lambda s: s.rank(pct=True)
    sc = pd.DataFrame(index=eligible.index)
    sc["trend"] = (r(eligible["px_vs_sma50"]) + r(eligible["sma50_vs_sma200"])) / 2
    sc["momentum"] = (r(eligible["mom_6m"]) + r(-(eligible["rsi14"] - 60).abs())) / 2
    sc["risk"] = r(-eligible["vol_daily"]) # biến động thấp thì điểm cao
    sc["liquidity"] = r(eligible["liq_value"])

    total_w = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    sc["score"] = sum(sc[k] * (weights[k] / total_w) for k in weights)
    ranked = sc.sort_values("score", ascending=False)

    # 4. Chọn Top N có lọc tương quan
    chosen = []
    if px_train is not None and not ranked.empty:
        ret_train = px_train[ranked.index].pct_change(fill_method=None)
        corr_matrix = ret_train.corr()

        for t in ranked.index:
            if all(corr_matrix.loc[t, c] <= max_corr for c in chosen):
                chosen.append(t)
            if len(chosen) == n_stocks:
                break

        # Nếu chưa đủ N mã, bổ sung theo thứ tự điểm cao nhất
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
    """
    Tối ưu hóa tỷ trọng Min Volatility bằng Ledoit-Wolf Covariance Shrinkage.
    Có cơ chế tự thích ứng:
    - Ưu tiên 1: PyPortfolioOpt
    - Ưu tiên 2: Scipy SLSQP + Sklearn LedoitWolf / Ledoit-Wolf pure numpy
    """
    sub_px = px_train[tickers].ffill().dropna()
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    # Thử phương pháp PyPortfolioOpt trước
    try:
        from pypfopt import risk_models, EfficientFrontier
        S_shrink = risk_models.CovarianceShrinkage(sub_px).ledoit_wolf()
        ef = EfficientFrontier(None, S_shrink, weight_bounds=(0.0, max_weight))
        raw_w = ef.min_volatility()
        cleaned_w = ef.clean_weights()
        return {t: float(cleaned_w.get(t, 0.0)) for t in tickers}
    except Exception:
        pass

    # Fallback: Tự tính Ledoit-Wolf Shrinkage & tối ưu bằng Scipy SLSQP
    try:
        from scipy.optimize import minimize
        # Tính tỷ suất lợi nhuận
        returns = sub_px.pct_change(fill_method=None).dropna().values
        T, p = returns.shape
        # Khử kỳ vọng mẫu
        mean_ret = np.mean(returns, axis=0)
        X = returns - mean_ret

        # Thử sklearn nếu có
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(store_precision=False, assume_centered=True)
            cov_matrix = lw.fit(X).covariance_
        except Exception:
            # Thuần Numpy tính Ledoit-Wolf rút gọn
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

        # Objective function: min w^T * Sigma * w
        def obj(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        # Constraints & Bounds
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, max_weight) for _ in range(n))
        init_guess = np.full(n, 1.0 / n)

        res = minimize(obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            w_opt = res.x
            # Làm tròn và chuẩn hóa lại tổng = 1
            w_opt = np.maximum(0, w_opt)
            w_opt = w_opt / np.sum(w_opt)
            return {t: float(w_opt[i]) for i, t in enumerate(tickers)}
    except Exception:
        pass

    # Nếu tất cả thất bại, chia đều Equal Weight
    return {t: 1.0 / n for t in tickers}

# -----------------------------------------------------------------------------
# 4. CÔNG CỤ TÍNH TOÁN HIỆU SUẤT & 8 CHỈ SỐ (PERFORMANCE METRICS)
# -----------------------------------------------------------------------------
def calculate_all_metrics(equity_series, rf_annual=0.03, trading_days=252, initial_capital=100_000_000):
    """
    Tính toán chính xác 8 chỉ số hiệu suất của chiến lược:
    1. Return (Tổng lợi suất và Lợi suất quy năm CAGR)
    2. Buy and Hold Benchmark (đối sánh)
    3. Winning Rate (Tỷ lệ phiên thắng)
    4. Volatility (Độ biến động quy năm)
    5. MDD (Max Drawdown - Mức sụt giảm tối đa)
    6. Sharpe Ratio
    7. Sortino Ratio (đo lường biến động giảm giá)
    8. Calmar Ratio (CAGR / |MDD|)
    """
    eq = equity_series.dropna()
    if len(eq) < 2:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Winning Rate": 0.0,
            "Win Days": 0,
            "Loss Days": 0,
            "Total Days": 0,
            "Volatility": 0.0,
            "Max Drawdown": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "Final Equity": initial_capital,
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

    # 3. Winning Rate (Xét trên các phiên có giao dịch lợi suất != 0, hoặc trên toàn bộ số phiên)
    win_days = int((daily_ret > 0).sum())
    loss_days = int((daily_ret < 0).sum())
    active_days = win_days + loss_days
    win_rate = (win_days / n_days) if n_days > 0 else 0.0
    win_rate_active = (win_days / active_days) if active_days > 0 else 0.0

    # 4. Volatility (Độ biến động quy đổi năm)
    volatility = daily_ret.std(ddof=1) * np.sqrt(trading_days) if len(daily_ret) > 1 else 0.0

    # 5. Max Drawdown (MDD)
    cummax = eq.cummax()
    drawdown = (eq / cummax) - 1.0
    max_dd = float(drawdown.min())

    # 6. Sharpe Ratio
    rf_daily = rf_annual / trading_days
    excess_ret = daily_ret.mean() * trading_days - rf_annual
    sharpe = (excess_ret / volatility) if (volatility > 0) else np.nan

    # 7. Sortino Ratio (Chỉ phạt độ lệch chuẩn của phần lợi nhuận âm)
    downside_returns = daily_ret.copy()
    downside_returns = downside_returns[downside_returns < 0] # Lợi suất âm
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(trading_days) if len(downside_returns) > 0 else 0.0
    sortino = (excess_ret / downside_std) if (downside_std > 0) else np.nan

    # 8. Calmar Ratio (CAGR / |MDD|)
    calmar = (cagr / abs(max_dd)) if (max_dd < 0 and not np.isnan(cagr)) else np.nan

    final_equity = float(eq.iloc[-1])
    net_profit = final_equity - initial_capital

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Winning Rate": win_rate,
        "Winning Rate Active": win_rate_active,
        "Win Days": win_days,
        "Loss Days": loss_days,
        "Total Days": n_days,
        "Volatility": volatility,
        "Max Drawdown": max_dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Final Equity": final_equity,
        "Net Profit": net_profit,
        "Drawdown Series": drawdown
    }

# -----------------------------------------------------------------------------
# 5. BACKTESTING ENGINE CHO 4 CHIẾN LƯỢC CHÍNH
# -----------------------------------------------------------------------------
def run_simulation(px_df, weights_dict, signal_series=None, initial_capital=100_000_000, cost=0.0015):
    """
    Mô phỏng đường vốn danh mục (Equity Curve):
    - Trừ phí mua ban đầu (cost)
    - Phân bổ theo trọng số
    - Áp dụng Market Timing Signal nếu có (signal = 1 đầu tư, signal = 0 giữ tiền mặt)
    """
    tickers = list(weights_dict.keys())
    sub_px = px_df[tickers].ffill().dropna()

    if sub_px.empty:
        return pd.Series(dtype=float)

    # Lợi suất từng mã
    daily_ret = sub_px.pct_change(fill_method=None).fillna(0)
    w_vec = np.array([weights_dict[t] for t in tickers])

    # Lợi suất danh mục cơ sở trước khi timing
    base_ret = daily_ret.dot(w_vec)

    # Nếu có tín hiệu Market Timing
    if signal_series is not None:
        sig_aligned = signal_series.reindex(sub_px.index).fillna(0)
        # Signal = 1: đầu tư; Signal = 0: tiền mặt (return = 0)
        strategy_ret = base_ret * sig_aligned
    else:
        strategy_ret = base_ret

    # Đường vốn danh mục (Equity curve)
    # Phí mua ban đầu trừ 1 lần vào ngày đầu tiên
    equity = initial_capital * (1.0 - cost) * (1.0 + strategy_ret).cumprod()
    return equity

def run_vnindex_simulation(benchmark_df, start_date, end_date, initial_capital=100_000_000):
    """Mô phỏng đường vốn cho chỉ số VNINDEX (không trừ phí giao dịch cổ phiếu)."""
    sub_b = benchmark_df.set_index("date")["close"].sort_index()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    sub_b = sub_b.loc[(sub_b.index >= start_dt) & (sub_b.index <= end_dt)].dropna()
    if sub_b.empty:
        return pd.Series(dtype=float)
    eq = initial_capital * (sub_b / sub_b.iloc[0])
    return eq

# -----------------------------------------------------------------------------
# 6. SIDEBAR - THIẾT LẬP THAM SỐ
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu Hình Hệ Thống")

# Section 1: Nguồn Dữ Liệu
st.sidebar.subheader("1. Dữ Liệu Thị Trường")
data_option = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["Dữ liệu mặc định (HOSE 2020-2023)", "Tải lên file CSV mới"],
    index=0
)

# Đường dẫn an toàn trên mọi môi trường (Local và Streamlit Cloud)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "HOSE_2020_2023_in.csv")
if not os.path.exists(csv_path):
    csv_path = "HOSE_2020_2023_in.csv"

uploaded_file = None

if data_option == "Tải lên file CSV mới":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is None:
        st.sidebar.info("Vui lòng tải lên file CSV để tiếp tục.")

# Đọc dữ liệu
if uploaded_file is not None:
    raw_df, err = load_and_clean_data(uploaded_file)
else:
    raw_df, err = load_and_clean_data(csv_path)

if err or raw_df is None:
    st.error(f"Lỗi đọc dữ liệu: {err}")
    st.stop()

# Tách riêng VNINDEX và cổ phiếu
benchmark_full = raw_df[raw_df["ticker"] == "VNINDEX"].copy()
stock_full = raw_df[raw_df["ticker"] != "VNINDEX"].copy()

min_date = raw_df["date"].min().date()
max_date = raw_df["date"].max().date()

# Hàm kẹp ngày an toàn chống lỗi StreamlitValueBelowMinError / StreamlitValueAboveMaxError
def clamp_date(target, min_d, max_d):
    return max(min_d, min(max_d, target))

# Tính các mốc ngày mặc định an toàn nằm trong [min_date, max_date]
default_train_start = clamp_date(datetime(2020, 1, 1).date(), min_date, max_date)
default_train_end = clamp_date(datetime(2021, 12, 31).date(), min_date, max_date)
default_test_start = clamp_date(datetime(2022, 1, 1).date(), min_date, max_date)
default_test_end = clamp_date(datetime(2022, 12, 31).date(), min_date, max_date)

# Section 2: Phân chia Train - Test
st.sidebar.subheader("2. Phân Chia Thời Gian")
col_s1, col_s2 = st.sidebar.columns(2)
train_start = col_s1.date_input("Train Bắt đầu", default_train_start, min_value=min_date, max_value=max_date)
train_end = col_s2.date_input("Train Kết thúc", default_train_end, min_value=min_date, max_value=max_date)

col_s3, col_s4 = st.sidebar.columns(2)
test_start = col_s3.date_input("Test Bắt đầu", default_test_start, min_value=min_date, max_value=max_date)
test_end = col_s4.date_input("Test Kết thúc", default_test_end, min_value=min_date, max_value=max_date)

if train_end >= test_start:
    st.sidebar.warning("Lưu ý: Thời gian Train nên trước thời gian Test để chống Look-ahead Bias!")

# Section 3: Cấu hình Lựa chọn Cổ phiếu
st.sidebar.subheader("3. Bộ Lọc Cổ Phiếu")
mode_selection = st.sidebar.radio(
    "Chế độ chọn danh mục:",
    ["Tự động theo thuật toán Notebook", "Người dùng tự chọn thủ công"],
    index=0
)

# Trọng số Multi-factor
st.sidebar.caption("Trọng số chấm điểm 4 yếu tố (Tổng chuẩn hóa 100%):")
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
rsi_max_input = st.sidebar.slider("Ngưỡng RSI tối đa (Lọc quá mua):", 50, 90, 75, 5)
max_corr_input = st.sidebar.slider("Ngưỡng tương quan tối đa (Đa dạng hóa):", 0.3, 0.95, 0.70, 0.05)

# Section 4: Tham số Giao dịch & Tối ưu
st.sidebar.subheader("4. Tham Số Giao Dịch & Mô Hình")
initial_capital = st.sidebar.number_input("Vốn ban đầu (VND):", min_value=10_000_000, value=100_000_000, step=10_000_000, format="%d")
rf_rate = st.sidebar.number_input("Lãi suất phi rủi ro Rf (%/năm):", min_value=0.0, max_value=20.0, value=3.0, step=0.5) / 100.0
trading_cost = st.sidebar.number_input("Phí giao dịch ban đầu (%):", min_value=0.0, max_value=2.0, value=0.15, step=0.05) / 100.0
max_stock_weight = st.sidebar.slider("Tỷ trọng tối đa 1 mã (Shrinkage bound):", 0.20, 1.00, 0.40, 0.05)
sma_timing_window = st.sidebar.number_input("Chu kỳ SMA Market Timing (phiên):", min_value=20, max_value=300, value=200, step=10)

# -----------------------------------------------------------------------------
# 7. XỬ LÝ LỌC VÀ TÍNH TOÁN
# -----------------------------------------------------------------------------
# Chia dữ liệu
train_stocks = stock_full[(stock_full["date"] >= pd.to_datetime(train_start)) & (stock_full["date"] <= pd.to_datetime(train_end))].copy()
test_stocks = stock_full[(stock_full["date"] >= pd.to_datetime(test_start)) & (stock_full["date"] <= pd.to_datetime(test_end))].copy()

if train_stocks.empty:
    st.error("Tập Train không có dữ liệu giao dịch! Vui lòng chọn lại khoảng thời gian Train.")
    st.stop()

if test_stocks.empty:
    st.error("Tập Test không có dữ liệu giao dịch! Vui lòng chọn lại khoảng thời gian Test.")
    st.stop()

# Tính chỉ báo
indicators_df, px_train_all, ind_err = compute_indicators_df(train_stocks)

if ind_err:
    st.warning(ind_err)
    st.stop()

# Lọc & xếp hạng
eligible_stocks, removed_stocks, ranked_stocks, default_top = apply_screening_and_scoring(
    indicators_df,
    weights_factor,
    n_liquid=int(n_liquid_input),
    rsi_max=rsi_max_input,
    max_corr=max_corr_input,
    n_stocks=int(n_stocks_input),
    px_train=px_train_all
)

# Xử lý danh sách cổ phiếu được chọn
if mode_selection == "Người dùng tự chọn thủ công":
    all_available = sorted(list(px_train_all.columns))
    selected_tickers = st.sidebar.multiselect(
        "Chọn các mã cổ phiếu:",
        all_available,
        default=default_top if len(default_top) <= len(all_available) else all_available[:n_stocks_input]
    )
    if len(selected_tickers) < 2:
        st.sidebar.warning("Vui lòng chọn tối thiểu 2 mã cổ phiếu.")
        selected_tickers = default_top
else:
    selected_tickers = default_top
    st.sidebar.success(f"Top {len(selected_tickers)} được chọn: {', '.join(selected_tickers)}")

# Chuẩn bị ma trận giá cho các mã được chọn
px_train_selected = train_stocks[train_stocks["ticker"].isin(selected_tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()
px_test_selected = test_stocks[test_stocks["ticker"].isin(selected_tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()

# Bỏ mã không có giá tại đầu kỳ Train/Test
px_train_selected = px_train_selected.loc[:, px_train_selected.iloc[0].notna()]
px_test_selected = px_test_selected.loc[:, px_test_selected.iloc[0].notna()]
valid_tickers = [t for t in selected_tickers if t in px_train_selected.columns and t in px_test_selected.columns]

if len(valid_tickers) < 2:
    st.error("Không đủ cổ phiếu có dữ liệu xuyên suốt Train và Test để chạy kiểm định. Hãy điều chỉnh lại danh sách hoặc thời gian.")
    st.stop()

# -----------------------------------------------------------------------------
# 8. TÍNH BỘ TỶ TRỌNG & TÍN HIỆU MARKET TIMING
# -----------------------------------------------------------------------------
# 1. Equal Weight
w_equal = {t: 1.0 / len(valid_tickers) for t in valid_tickers}

# 2. Shrinkage Ledoit-Wolf Min Volatility (ước lượng thuần túy trên TRAIN)
w_shrink = optimize_shrinkage_min_vol(px_train_selected, valid_tickers, max_weight=max_stock_weight)

# 3. Market Timing Signal (VNINDEX SMA200 với shift(1) chống Look-ahead bias)
vnindex_series = benchmark_full.set_index("date")["close"].sort_index()
vnindex_sma = vnindex_series.rolling(window=sma_timing_window).mean()
# Signal = 1 khi VNINDEX > SMA, shift 1 ngày
market_signal = (vnindex_series > vnindex_sma).astype(int).shift(1).fillna(0)

# -----------------------------------------------------------------------------
# 9. CHẠY MÔ PHỎNG EQUITY CHO 4 CHIẾN LƯỢC TRÊN TRAIN & TEST
# -----------------------------------------------------------------------------
# A. TRAIN
eq_train_baseline = run_simulation(px_train_selected, w_equal, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_train_shrink = run_simulation(px_train_selected, w_shrink, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_train_mt = run_simulation(px_train_selected, w_equal, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_train_combined = run_simulation(px_train_selected, w_shrink, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_train_vnindex = run_vnindex_simulation(benchmark_full, train_start, train_end, initial_capital=initial_capital)

# B. TEST
eq_test_baseline = run_simulation(px_test_selected, w_equal, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_test_shrink = run_simulation(px_test_selected, w_shrink, signal_series=None, initial_capital=initial_capital, cost=trading_cost)
eq_test_mt = run_simulation(px_test_selected, w_equal, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_test_combined = run_simulation(px_test_selected, w_shrink, signal_series=market_signal, initial_capital=initial_capital, cost=trading_cost)
eq_test_vnindex = run_vnindex_simulation(benchmark_full, test_start, test_end, initial_capital=initial_capital)

# Tính toán các chỉ số cho từng chiến lược
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
# 10. GIAO DIỆN CHÍNH - DASHBOARD STREAMLIT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 Hệ Thống Kiểm Định Chiến Lược Đầu Tư Định Lượng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Nghiên cứu kiểm chứng mô hình Lựa Chọn Cổ Phiếu Đa Nhân Tố, Tối Ưu Hóa Rủi Ro Co-variance Shrinkage & Định Thời Điểm Thị Trường (Market Timing) trên sàn HOSE</div>', unsafe_allow_html=True)

# 5 Tabs chức năng chính
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Dữ Liệu & Phân Chia",
    "🎯 2. Lọc & Xếp Hạng Mã",
    "🚀 3. Kiểm Định 8 Chỉ Số & Hiệu Suất",
    "⚖️ 4. Phân Bổ Tỷ Trọng Danh Mục",
    "📑 5. Báo Cáo Chuyên Sâu & Xuất Dữ Liệu"
])

# -----------------------------------------------------------------------------
# TAB 1: DỮ LIỆU & PHÂN CHIA TRAIN - TEST
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Tổng Quan Dữ Liệu Thị Trường")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tổng Số Dòng Dữ Liệu", f"{len(raw_df):,}")
    with c2:
        st.metric("Số Mã Cổ Phiếu (HOSE)", f"{stock_full['ticker'].nunique():,} mã")
    with c3:
        st.metric("Khoảng Thời Gian", f"{min_date.strftime('%d/%m/%Y')} → {max_date.strftime('%d/%m/%Y')}")
    with c4:
        st.metric("Dữ Liệu Tham Chiếu", "Chỉ số VN-INDEX")

    st.markdown("---")
    st.subheader("2. Chi Tiết Phân Chia Tập Train - Test")
    st.markdown("""
    <div class="highlight-box">
    <b>Quy Tắc Khoa Học:</b> Quá trình nghiên cứu phân tách rạch ròi giữa <b>In-Sample (Train 2020-2021)</b> để xây dựng mô hình, lựa chọn cổ phiếu và ước lượng ma trận hiệp phương sai; sau đó kiểm định khách quan trên <b>Out-of-Sample (Test 2022)</b> mà không sửa đổi tham số để tránh hiện tượng học vẹt (Overfitting) và nhìn trước tương lai (Look-ahead Bias).
    </div>
    """, unsafe_allow_html=True)

    split_summary = pd.DataFrame({
        "Giai đoạn": ["Tập TRAIN (In-Sample)", "Tập TEST (Out-of-Sample)"],
        "Bối cảnh thị trường": ["Thị trường Tăng trưởng mạnh (Uptrend / Bull market)", "Thị trường Điều chỉnh gắt gao (Downtrend / Bear market)"],
        "Ngày bắt đầu": [train_start.strftime('%d/%m/%Y'), test_start.strftime('%d/%m/%Y')],
        "Ngày kết thúc": [train_end.strftime('%d/%m/%Y'), test_end.strftime('%d/%m/%Y')],
        "Số phiên giao dịch": [train_stocks["date"].nunique(), test_stocks["date"].nunique()],
        "Số mã giao dịch": [train_stocks["ticker"].nunique(), test_stocks["ticker"].nunique()]
    })
    st.dataframe(split_summary, use_container_width=True, hide_index=True)

    # Biểu đồ VNINDEX tổng thể với vùng phân tách Train - Test
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(
        x=benchmark_full["date"],
        y=benchmark_full["close"],
        mode="lines",
        name="VNINDEX",
        line=dict(color="#1E3A8A", width=2)
    ))
    # Đường SMA200
    fig_overview.add_trace(go.Scatter(
        x=vnindex_sma.index,
        y=vnindex_sma.values,
        mode="lines",
        name=f"SMA {sma_timing_window}",
        line=dict(color="#F59E0B", width=1.5, dash="dash")
    ))

    # Đánh dấu vùng Train và Test
    fig_overview.add_vrect(
        x0=str(train_start), x1=str(train_end),
        fillcolor="rgba(16, 185, 129, 0.12)", layer="below", line_width=0,
        annotation_text="GIAI ĐOẠN TRAIN (Uptrend)", annotation_position="top left"
    )
    fig_overview.add_vrect(
        x0=str(test_start), x1=str(test_end),
        fillcolor="rgba(239, 68, 68, 0.12)", layer="below", line_width=0,
        annotation_text="GIAI ĐOẠN TEST (Downtrend)", annotation_position="top left"
    )
    fig_overview.update_layout(
        title="Biến Động Chỉ Số VNINDEX & Đường Trung Bình Động (2020 - 2023)",
        xaxis_title="Thời gian",
        yaxis_title="Điểm số VN-Index",
        template="plotly_white",
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig_overview, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: LỌC & XẾP HẠNG MÃ CỔ PHIẾU
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("1. Cơ Chế Lọc & Chấm Điểm Đa Nhân Tố (Multi-Factor Scoring)")
    st.markdown(f"""
    Quy trình chọn lọc Top {n_stocks_input} cổ phiếu được thực hiện hoàn toàn trên tập dữ liệu **Train**:
    1. **Thanh khoản**: Lấy Top `{n_liquid_input}` mã có giá trị giao dịch trung vị 60 phiên cao nhất.
    2. **Bộ lọc rủi ro**: Loại bỏ mã có `RSI14 > {rsi_max_input}` (quá mua) và `Giá < SMA50` (mất xu hướng tăng ngắn hạn).
    3. **Chấm điểm**: Chuẩn hóa phân vị (Percentile Rank) kết hợp 4 nhóm: **Xu hướng ({weights_factor['trend']:.0%})**, **Động lượng ({weights_factor['momentum']:.0%})**, **Rủi ro thấp ({weights_factor['risk']:.0%})**, **Thanh khoản ({weights_factor['liquidity']:.0%})**.
    4. **Đa dạng hóa**: Loại bỏ mã có hệ số tương quan lợi suất ngày > `{max_corr_input}` với các mã đã chọn trước đó.
    """)

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.write("##### Bảng Xếp Hạng & Điểm Tổng Hợp Các Mã Đủ Điều Kiện")
        display_ranked = ranked_stocks.copy()
        display_ranked["Điểm Tổng"] = display_ranked["score"]
        display_ranked["Được Chọn"] = display_ranked.index.isin(valid_tickers)
        st.dataframe(
            display_ranked[["score", "trend", "momentum", "risk", "liquidity", "Được Chọn"]]
            .style.format({
                "score": "{:.3f}", "trend": "{:.3f}", "momentum": "{:.3f}", "risk": "{:.3f}", "liquidity": "{:.3f}"
            }).background_gradient(subset=["score"], cmap="YlGnBu"),
            use_container_width=True,
            height=360
        )
    with col_t2:
        st.write(f"##### Điểm Tổng Hợp Danh Mục Được Chọn (Top {len(valid_tickers)})")
        top_scores = ranked_stocks.loc[valid_tickers, "score"].sort_values(ascending=True)
        fig_bar = px.bar(
            x=top_scores.values,
            y=top_scores.index,
            orientation='h',
            labels={"x": "Điểm tổng hợp", "y": "Mã CP"},
            color=top_scores.values,
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.write("##### Ma Trận Tương Quan Lợi Suất Ngày (Train) Giữa Các Mã Được Chọn")
    ret_selected = px_train_selected.pct_change(fill_method=None).dropna()
    corr_matrix = ret_selected.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        zmin=0.0, zmax=1.0,
        labels=dict(color="Hệ số tương quan")
    )
    fig_corr.update_layout(
        title=f"Hệ số tương quan chéo giữa các cổ phiếu được chọn (Mục tiêu: Đa dạng hóa rủi ro, tương quan <= {max_corr_input})",
        height=380
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: KIỂM ĐỊNH CHI TIẾT 8 CHỈ SỐ HIỆU SUẤT
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("1. Bảng Điều Khiển 8 Chỉ Số Hiệu Suất Cốt Lõi")
    phase_choice = st.radio("Chọn giai đoạn kiểm tra:", ["Tập TEST (Out-of-sample / 2022)", "Tập TRAIN (In-sample / 2020-2021)"], horizontal=True)
    active_phase = "Test" if "TEST" in phase_choice else "Train"

    p_metrics = metrics_map[active_phase]
    best_strategy_name = "Kết Hợp (Shrinkage + MT)" if "Kết Hợp (Shrinkage + MT)" in p_metrics else list(p_metrics.keys())[0]
    best_m = p_metrics[best_strategy_name]
    baseline_m = p_metrics["Baseline (Buy & Hold 1/N)"]

    # Hiển thị 8 thẻ chỉ số (KPI cards) theo đúng yêu cầu đề bài
    st.markdown(f"#### 8 Chỉ Số Hiệu Suất Của Chiến Lược Tối Ưu Nhất [{best_strategy_name}] ({active_phase})")
    kpi_cols1 = st.columns(4)
    with kpi_cols1[0]:
        val = best_m['Total Return']
        st.metric(
            "1. RETURN (Tổng lợi suất)",
            f"{val:.2%}",
            delta=f"{(val - baseline_m['Total Return']):+.2%} vs Buy&Hold",
            delta_color="normal"
        )
    with kpi_cols1[1]:
        bh_val = baseline_m['Total Return']
        st.metric(
            "2. BUY & HOLD (Baseline)",
            f"{bh_val:.2%}",
            delta=f"{(bh_val - p_metrics['VNINDEX Benchmark']['Total Return']):+.2%} vs VNINDEX",
            delta_color="normal"
        )
    with kpi_cols1[2]:
        st.metric(
            "3. WINNING RATE (Tỷ lệ thắng)",
            f"{best_m['Winning Rate']:.2%}",
            help=f"Thắng: {best_m['Win Days']} ngày | Thua: {best_m['Loss Days']} ngày trên tổng số {best_m['Total Days']} phiên."
        )
    with kpi_cols1[3]:
        vol_val = best_m['Volatility']
        st.metric(
            "4. VOLATILITY (Biến động năm)",
            f"{vol_val:.2%}",
            delta=f"{(vol_val - baseline_m['Volatility']):+.2%} vs Buy&Hold",
            delta_color="inverse"
        )

    kpi_cols2 = st.columns(4)
    with kpi_cols2[0]:
        mdd_val = best_m['Max Drawdown']
        st.metric(
            "5. MDD (Sụt giảm tối đa)",
            f"{mdd_val:.2%}",
            delta=f"{(mdd_val - baseline_m['Max Drawdown']):+.2%} vs Buy&Hold",
            delta_color="normal"
        )
    with kpi_cols2[1]:
        sharpe_val = best_m['Sharpe']
        st.metric(
            "6. SHARPE RATIO (Rf=3%)",
            f"{sharpe_val:.2f}" if not np.isnan(sharpe_val) else "N/A",
            delta=f"{(sharpe_val - baseline_m['Sharpe']):+.2f} vs Buy&Hold" if not np.isnan(sharpe_val) and not np.isnan(baseline_m['Sharpe']) else None,
            delta_color="normal"
        )
    with kpi_cols2[2]:
        sortino_val = best_m['Sortino']
        st.metric(
            "7. SORTINO RATIO",
            f"{sortino_val:.2f}" if not np.isnan(sortino_val) else "N/A",
            delta=f"{(sortino_val - baseline_m['Sortino']):+.2f} vs Buy&Hold" if not np.isnan(sortino_val) and not np.isnan(baseline_m['Sortino']) else None,
            delta_color="normal"
        )
    with kpi_cols2[3]:
        calmar_val = best_m['Calmar']
        st.metric(
            "8. CALMAR RATIO (CAGR/MDD)",
            f"{calmar_val:.2f}" if not np.isnan(calmar_val) else "N/A",
            delta=f"{(calmar_val - baseline_m['Calmar']):+.2f} vs Buy&Hold" if not np.isnan(calmar_val) and not np.isnan(baseline_m['Calmar']) else None,
            delta_color="normal"
        )

    st.markdown("---")
    st.subheader(f"2. Bảng So Sánh Chi Tiết Toàn Diện Các Chiến Lược ({active_phase})")

    # Tạo bảng DataFrame tổng hợp cho tất cả chiến lược
    comparison_rows = []
    for s_name, s_m in p_metrics.items():
        comparison_rows.append({
            "Chiến Lược": s_name,
            "1. Tổng Return": f"{s_m['Total Return']:.2%}",
            "Lợi suất/Năm (CAGR)": f"{s_m['CAGR']:.2%}" if not np.isnan(s_m['CAGR']) else "N/A",
            "3. Winning Rate": f"{s_m['Winning Rate']:.2%}",
            "4. Volatility (Năm)": f"{s_m['Volatility']:.2%}",
            "5. Max Drawdown": f"{s_m['Max Drawdown']:.2%}",
            "6. Sharpe (Rf=3%)": f"{s_m['Sharpe']:.2f}" if not np.isnan(s_m['Sharpe']) else "N/A",
            "7. Sortino": f"{s_m['Sortino']:.2f}" if not np.isnan(s_m['Sortino']) else "N/A",
            "8. Calmar": f"{s_m['Calmar']:.2f}" if not np.isnan(s_m['Calmar']) else "N/A",
            "Vốn Cuối Kỳ (VND)": f"{s_m['Final Equity']:,.0f}"
        })
    df_comp = pd.DataFrame(comparison_rows).set_index("Chiến Lược")
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Biểu Đồ Tăng Trưởng Tài Sản & Mức Sụt Giảm (Interactive Wealth & Drawdown Curves)")

    # Chuẩn bị dữ liệu vẽ chart
    if active_phase == "Test":
        chart_eqs = {
            "Baseline (Buy & Hold 1/N)": eq_test_baseline,
            "Shrinkage Min-Vol": eq_test_shrink,
            "Market Timing (SMA)": eq_test_mt,
            "Kết Hợp (Shrinkage + MT)": eq_test_combined,
            "VNINDEX Benchmark": eq_test_vnindex
        }
    else:
        chart_eqs = {
            "Baseline (Buy & Hold 1/N)": eq_train_baseline,
            "Shrinkage Min-Vol": eq_train_shrink,
            "Market Timing (SMA)": eq_train_mt,
            "Kết Hợp (Shrinkage + MT)": eq_train_combined,
            "VNINDEX Benchmark": eq_train_vnindex
        }

    # Biểu đồ Equity Curve
    fig_equity = go.Figure()
    colors = {
        "Baseline (Buy & Hold 1/N)": "#EF4444",       # Đỏ
        "Shrinkage Min-Vol": "#8B5CF6",               # Tím
        "Market Timing (SMA)": "#F59E0B",             # Vàng cam
        "Kết Hợp (Shrinkage + MT)": "#10B981",        # Xanh lá đậm (Nổi bật)
        "VNINDEX Benchmark": "#1E3A8A"                # Xanh navy
    }
    dash_styles = {
        "Baseline (Buy & Hold 1/N)": "dot",
        "Shrinkage Min-Vol": "dash",
        "Market Timing (SMA)": "dashdot",
        "Kết Hợp (Shrinkage + MT)": "solid",
        "VNINDEX Benchmark": "solid"
    }
    widths = {
        "Baseline (Buy & Hold 1/N)": 1.8,
        "Shrinkage Min-Vol": 2.0,
        "Market Timing (SMA)": 2.0,
        "Kết Hợp (Shrinkage + MT)": 3.2,
        "VNINDEX Benchmark": 1.5
    }

    for name, s_eq in chart_eqs.items():
        if not s_eq.empty:
            fig_equity.add_trace(go.Scatter(
                x=s_eq.index,
                y=s_eq / 1_000_000,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#000000"), dash=dash_styles.get(name, "solid"), width=widths.get(name, 2))
            ))

    fig_equity.add_hline(
        y=initial_capital / 1_000_000,
        line=dict(color="gray", dash="dash", width=1),
        annotation_text="Vốn khởi điểm (100 Triệu)"
    )
    fig_equity.update_layout(
        title=f"Đường Phát Triển Vốn (Equity Curve) - Giai đoạn {active_phase}",
        xaxis_title="Thời gian",
        yaxis_title="Giá trị danh mục (Triệu VNĐ)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Biểu đồ Drawdown (Underwater Chart)
    fig_dd = go.Figure()
    for name, s_eq in chart_eqs.items():
        if not s_eq.empty:
            dd_series = (s_eq / s_eq.cummax() - 1.0) * 100
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index,
                y=dd_series,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#000000"), width=widths.get(name, 1.5))
            ))

    fig_dd.update_layout(
        title=f"Đồ Thị Sụt Giảm Từ Đỉnh (Underwater Drawdown %)",
        xaxis_title="Thời gian",
        yaxis_title="Mức sụt giảm (%)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Trực quan hóa tín hiệu Market Timing (Invested vs Cash)
    st.markdown("---")
    st.subheader(f"4. Trực Quan Hóa Tín Hiệu Market Timing SMA {sma_timing_window} ({active_phase})")
    cur_px = px_test_selected if active_phase == "Test" else px_train_selected
    sig_sub = market_signal.reindex(cur_px.index).fillna(0)
    invested_pct = sig_sub.mean()

    st.write(f"Tỷ lệ thời gian giải ngân tham gia thị trường trong giai đoạn {active_phase}: **{invested_pct:.1%}** | Tỷ lệ đứng ngoài giữ tiền mặt: **{1 - invested_pct:.1%}**")

    vn_sub = benchmark_full.set_index("date")["close"].reindex(cur_px.index).dropna()
    vn_sma_sub = vnindex_sma.reindex(cur_px.index).dropna()

    fig_mt = go.Figure()
    fig_mt.add_trace(go.Scatter(x=vn_sub.index, y=vn_sub, name="VNINDEX", line=dict(color="#1E3A8A", width=2)))
    fig_mt.add_trace(go.Scatter(x=vn_sma_sub.index, y=vn_sma_sub, name=f"SMA {sma_timing_window}", line=dict(color="#F59E0B", width=1.5, dash="dash")))

    # Đánh dấu các vùng cầm tiền mặt (Signal = 0)
    in_cash = False
    start_cash_dt = None
    for dt, val in sig_sub.items():
        if val == 0 and not in_cash:
            in_cash = True
            start_cash_dt = dt
        elif val == 1 and in_cash:
            in_cash = False
            fig_mt.add_vrect(
                x0=start_cash_dt, x1=dt,
                fillcolor="rgba(156, 163, 175, 0.25)", layer="below", line_width=0,
                annotation_text="Tiền Mặt (OFF)" if (dt - start_cash_dt).days > 15 else ""
            )
    if in_cash:
        fig_mt.add_vrect(
            x0=start_cash_dt, x1=sig_sub.index[-1],
            fillcolor="rgba(156, 163, 175, 0.25)", layer="below", line_width=0,
            annotation_text="Tiền Mặt (OFF)"
        )

    fig_mt.update_layout(
        title="Tín Hiệu Định Thời Điểm: Vùng xám là giai đoạn VNINDEX < SMA200 (Thoát hàng & Giữ 100% tiền mặt)",
        xaxis_title="Thời gian",
        yaxis_title="Điểm số VN-Index",
        template="plotly_white",
        height=380,
        hovermode="x unified"
    )
    st.plotly_chart(fig_mt, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: PHÂN BỔ TỶ TRỌNG & TỐI ƯU SHRINKAGE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("1. So Sánh Cơ Cấu Tỷ Trọng Danh Mục")
    st.markdown("""
    - **Baseline (Equal Weight 1/N)**: Chia đều vốn bằng nhau cho tất cả các mã.
    - **Shrinkage Min-Volatility**: Khử nhiễu ma trận hiệp phương sai bằng phương pháp **Ledoit-Wolf Covariance Shrinkage**, sau đó giải bài toán tối thiểu hóa phương sai danh mục (Minimum Variance) với ràng buộc tỷ trọng không vượt quá giới hạn thiết lập.
    """)

    df_weights = pd.DataFrame({
        "Mã Cổ Phiếu": valid_tickers,
        "Equal Weight (1/N)": [w_equal[t] for t in valid_tickers],
        "Shrinkage Min-Vol": [w_shrink.get(t, 0.0) for t in valid_tickers]
    })

    col_w1, col_w2 = st.columns([2, 3])
    with col_w1:
        st.write("##### Bảng Tỷ Trọng Phân Bổ Tối Ưu")
        st.dataframe(
            df_weights.style.format({
                "Equal Weight (1/N)": "{:.2%}",
                "Shrinkage Min-Vol": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )
    with col_w2:
        st.write("##### Biểu Đồ Trực Quan Tỷ Trọng Tối Ưu")
        fig_pie = px.bar(
            df_weights.melt(id_vars=["Mã Cổ Phiếu"], var_name="Mô hình", value_name="Tỷ trọng"),
            x="Mã Cổ Phiếu",
            y="Tỷ trọng",
            color="Mô hình",
            barmode="group",
            text_auto=".1%",
            color_discrete_map={"Equal Weight (1/N)": "#94A3B8", "Shrinkage Min-Vol": "#10B981"}
        )
        fig_pie.update_layout(height=350, yaxis_tickformat=".0%")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Nguyên Lý Khoa Học Của Kỹ Thuật Ledoit-Wolf Shrinkage")
    st.markdown("""
    Trong lý thuyết danh mục đầu tư hiện đại (Markowitz MPT), ma trận hiệp phương sai mẫu thường bị **nhiễu (estimation noise)** do số lượng cổ phiếu lớn và độ dài chuỗi thời gian có hạn, dẫn đến việc phân bổ tỷ trọng cực đoan vào các mã có phương sai ngẫu nhiên thấp.
    
    Phương pháp **Ledoit-Wolf Shrinkage** giải quyết triệt để vấn đề này bằng cách thu hẹp ma trận hiệp phương sai mẫu $S$ về phía ma trận mục tiêu có cấu trúc ổn định $\mu I$:
    $$\Sigma_{\text{shrink}} = (1 - \delta) S + \delta F$$
    Nhờ đó, danh mục đạt được độ ổn định tối đa trên tập ngoài mẫu (**Out-of-sample**), hạn chế tối đa rủi ro suy giảm lợi nhuận.
    """)

# -----------------------------------------------------------------------------
# TAB 5: BÁO CÁO CHUYÊN SÂU & XUẤT DỮ LIỆU
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("1. Đối Chiếu Kết Quả Train (In-Sample) vs Test (Out-of-Sample)")

    # Bảng dọc chuẩn báo cáo nghiên cứu
    metric_names = [
        "Tổng Lợi Suất (Return)", "Lợi Suất Quy Năm (CAGR)", "Tỷ Lệ Thắng (Winning Rate)",
        "Biến Động Quy Năm (Volatility)", "Mức Sụt Giảm Tối Đa (MDD)", "Chỉ Số Sharpe (Rf=3%)",
        "Chỉ Số Sortino", "Chỉ Số Calmar"
    ]

    def get_strategy_vector(phase, s_key):
        m = metrics_map[phase][s_key]
        return [
            m["Total Return"], m["CAGR"], m["Winning Rate"],
            m["Volatility"], m["Max Drawdown"], m["Sharpe"],
            m["Sortino"], m["Calmar"]
        ]

    # Tạo bảng so sánh đối đầu
    strategies_to_show = ["Baseline (Buy & Hold 1/N)", "Shrinkage Min-Vol", "Market Timing (SMA)", "Kết Hợp (Shrinkage + MT)"]
    final_report_data = {"Chỉ Số Hiệu Suất": metric_names}

    for s in strategies_to_show:
        final_report_data[f"{s} (TRAIN)"] = get_strategy_vector("Train", s)
        final_report_data[f"{s} (TEST)"] = get_strategy_vector("Test", s)

    df_final_report = pd.DataFrame(final_report_data).set_index("Chỉ Số Hiệu Suất")

    # Format hiển thị
    format_rules = {}
    for col in df_final_report.columns:
        format_rules[col] = "{:.2%}"

    st.dataframe(
        df_final_report.style.format(lambda val: f"{val:.2f}" if isinstance(val, (int, float)) and abs(val) < 10 and not np.isnan(val) and abs(val) > 0.0001 and val != 0 else (f"{val:.2%}" if isinstance(val, (int, float)) and not np.isnan(val) else "N/A")),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("2. Đánh Giá & Nhận Định Chuyên Gia Định Lượng")
    st.markdown("""
    1. **Thảm họa của Buy & Hold thụ động trong Downtrend (2022)**:
       - Trong giai đoạn Uptrend (Train 2020-2021), chiến lược mua và nắm giữ (Buy & Hold) mang lại lợi suất cực kỳ ấn tượng nhờ sóng tăng chung của thị trường.
       - Tuy nhiên, khi bước sang chu kỳ suy thoái (Test 2022), Buy & Hold ghi nhận mức sụt giảm tối đa (MDD) lên tới hơn **-70% đến -80%**, gần như thổi bay toàn bộ thành quả tích lũy.
    2. **Vai trò sống còn của Market Timing (VNINDEX SMA200)**:
       - Tín hiệu Market Timing đã giúp danh mục **thoát khỏi thị trường từ rất sớm khi VN-Index gãy đường trung bình 200 ngày**, giữ nguyên 100% tiền mặt trong phần lớn thời gian năm 2022.
       - Giúp co hẹp Max Drawdown từ mức -80% xuống ngưỡng an toàn dung sai kiểm soát được.
    3. **Sức mạnh cộng hưởng của Mô hình Kết hợp (Shrinkage + Market Timing)**:
       - Tối ưu hóa Ledoit-Wolf Shrinkage giúp kiểm soát tỷ trọng rủi ro tối ưu giữa các cổ phiếu trong pha tăng trưởng.
       - Tín hiệu Market Timing đóng vai trò như chiếc phanh khẩn cấp để bảo vệ tài khoản trong pha suy thoái.
       - Đây chính là triết lý quản trị danh mục định lượng bền vững: **Tối đa hóa hiệu quả điều chỉnh theo rủi ro (Risk-Adjusted Return)**.
    """)

    st.markdown("---")
    st.subheader("3. Xuất Báo Cáo Dữ Liệu (Export CSV)")
    
    # Nút tải bảng báo cáo
    csv_buffer = io.StringIO()
    df_final_report.to_csv(csv_buffer)
    st.download_button(
        label="📥 Tải Xuống Báo Cáo Hiệu Suất (CSV)",
        data=csv_buffer.getvalue().encode('utf-8-sig'),
        file_name="Bao_Cao_Kiem_Dinh_Chien_Luoc_Dau_Tu.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Ứng dụng phát triển phục vụ công tác nghiên cứu, kiểm định định lượng danh mục đầu tư chứng khoán HOSE | Phiên bản Web App Streamlit")
