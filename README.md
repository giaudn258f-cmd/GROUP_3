# 📈 ỨNG DỤNG KIỂM ĐỊNH TÍNH HIỆU QUẢ CỦA CHIẾN LƯỢC ĐẦU TƯ ĐỊNH LƯỢNG (HOSE 2020 - 2023)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ứng dụng Web tương tác được xây dựng trên nền tảng **Streamlit**, phục vụ công tác nghiên cứu định lượng (Quantitative Finance) và kiểm định (Backtesting) danh mục đầu tư cổ phiếu niêm yết trên sàn HOSE giai đoạn 2020 – 2023.

---

## 🎯 8 CHỈ SỐ HIỆU SUẤT ĐƯỢC KIỂM ĐỊNH

Hệ thống tính toán và đối chiếu đồng thời **8 chỉ số đo lường hiệu quả danh mục** theo chuẩn phân tích tài chính quốc tế:

1. **Return (Tổng Lợi Suất & Lợi Suất Quy Năm - CAGR)**:
   - *Total Return*: $R_{\text{total}} = \frac{V_{\text{cuối}}}{V_{\text{đầu}}} - 1$
   - *CAGR (Compound Annual Growth Rate)*: $R_{\text{ann}} = \left(\frac{V_{\text{cuối}}}{V_{\text{đầu}}}\right)^{\frac{252}{N}} - 1$
2. **Buy and Hold (Mốc Đối Sánh Cơ Sở)**:
   - Chiến lược Mua & Nắm giữ thụ động với tỷ trọng đều ($1/N$) của các cổ phiếu được chọn, đối sánh trực tiếp với mức sinh lời của chỉ số toàn thị trường **VN-INDEX**.
3. **Winning Rate (Tỷ Lệ Phiên Sinh Lời Dương)**:
   - $\text{Win Rate} = \frac{\text{Số ngày sinh lời } > 0}{\text{Tổng số ngày giao dịch}}$ (và tỷ lệ tính trên số ngày giải ngân hoạt động).
4. **Volatility (Độ Biến Động Lợi Suất Quy Năm)**:
   - $\sigma_{\text{ann}} = \text{std}(R_{\text{daily}}, ddof=1) \times \sqrt{252}$
5. **MDD (Max Drawdown - Mức Sụt Giảm Tối Đa)**:
   - Biên độ giảm lớn nhất từ đỉnh tài sản tích lũy: $\text{Drawdown}_t = \frac{V_t}{\max_{\tau \le t} V_\tau} - 1 \implies \text{MDD} = \min_t (\text{Drawdown}_t)$
6. **Sharpe Ratio (Tỷ Số Phần Bù Rủi Ro / Tổng Biến Động)**:
   - $\text{Sharpe} = \frac{R_{\text{ann\_mean}} - R_f}{\sigma_{\text{ann}}}$, với lãi suất phi rủi ro tham chiếu $R_f = 3\%$/năm.
7. **Sortino Ratio (Tỷ Số Điều Chỉnh Theo Rủi Ro Giảm Giá)**:
   - Chỉ phạt độ lệch chuẩn của phần lợi nhuận âm (Downside Deviation): $\text{Sortino} = \frac{R_{\text{ann\_mean}} - R_f}{\sigma_{\text{downside}} \times \sqrt{252}}$
8. **Calmar Ratio (Tỷ Số Sinh Lời Quy Năm Trên Sụt Giảm Tối Đa)**:
   - $\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$, phản ánh chất lượng tạo ra lợi nhuận trên mỗi đơn vị sụt giảm vốn sâu nhất.

---

## 🔬 4 CHIẾN LƯỢC ĐẦU TƯ ĐƯỢC SO SÁNH

| Chiến lược | Phương pháp phân bổ | Quản trị rủi ro thị trường |
| :--- | :--- | :--- |
| **1. Baseline (Buy & Hold 1/N)** | Tỷ trọng đều $1/N$ giữa các cổ phiếu Top được chọn | Nắm giữ 100% qua mọi điều kiện thị trường |
| **2. Shrinkage Min-Volatility** | Tối ưu hóa Ledoit-Wolf Covariance Shrinkage (tránh Overfitting) | Giảm thiểu biến động danh mục, chặn tỷ trọng tối đa mỗi mã $\le 40\%$ |
| **3. Market Timing (SMA200)** | Tỷ trọng đều $1/N$ | Khi VN-Index $>$ SMA200: Đầu tư 100%. Khi $\le$ SMA200: Giữ 100% tiền mặt (*Tránh Look-ahead bias bằng shift(1)*) |
| **4. Mô hình Kết Hợp (Shrinkage + MT)** | Tỷ trọng tối ưu biến động thấp Ledoit-Wolf Shrinkage | Tự động kích hoạt phòng thủ tiền mặt khi VN-Index dưới SMA200 |

---

## 💻 HƯỚNG DẪN CÀI ĐẶT & CHẠY LOCAL (MÁY CÁ NHÂN)

```bash
# 1. Cài đặt các thư viện
pip install -r requirements.txt

# 2. Khởi chạy Web App
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

---

## 🚀 HƯỚNG DẪN DEPLOY MIỄN PHÍ TRÊN STREAMLIT CLOUD

1. **Đẩy code lên GitHub**:
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "Deploy streamlit quantitative backtest app"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git
   git push -u origin main
   ```
2. **Deploy lên Streamlit Cloud**:
   - Truy cập **[share.streamlit.io](https://share.streamlit.io)** và đăng nhập bằng GitHub.
   - Nhấn **"New app"** $\rightarrow$ Chọn Repository vừa tạo $\rightarrow$ Branch: `main` $\rightarrow$ Main file path: `app.py`.
   - Nhấn nút **"Deploy!"** và đợi hệ thống chạy.
