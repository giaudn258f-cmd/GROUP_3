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

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── app.py                      # Mã nguồn ứng dụng web chính (Streamlit Dashboard)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Hướng dẫn chi tiết sử dụng & deploy
├── HOSE_2020_2023_in.csv       # File dữ liệu giao dịch các mã HOSE (2020 - 2023)
└── NHOM_3_CHIEN_LUOC_DAU_TU_V4 (FINAL).ipynb # Notebook nghiên cứu gốc
```

---

## 💻 HƯỚNG DẪN CÀI ĐẶT & CHẠY LOCAL (MÁY TÍNH CÁ NHÂN)

### Bước 1: Chuẩn bị môi trường Python
Yêu cầu Python phiên bản `3.9` trở lên. Mở Terminal / PowerShell và di chuyển vào thư mục dự án:
```bash
cd "đường_dẫn_tới_thư_mục_dự_án"
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Khởi chạy Web App
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở địa chỉ: `http://localhost:8501`

---

## 🚀 HƯỚNG DẪN ĐẨY LÊN GITHUB & DEPLOY MIỄN PHÍ TRÊN STREAMLIT CLOUD

### Bước 1: Khởi tạo và Đẩy code lên GitHub
1. Tạo một tài khoản trên [GitHub.com](https://github.com) (nếu chưa có).
2. Tạo một Repository mới trên GitHub (ví dụ đặt tên: `qldmdt-nhom3-quant`).
3. Mở terminal tại thư mục dự án và chạy các lệnh:
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "Khoi tao web app kiem dinh chien luoc dinh luong Streamlit"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/qldmdt-nhom3-quant.git
   git push -u origin main
   ```

### Bước 2: Deploy lên Streamlit Community Cloud
1. Truy cập vào **[share.streamlit.io](https://share.streamlit.io)** và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc "Create app").
3. Điền các thông tin:
   - **Repository**: Chọn repo vừa tạo (`<tai-khoan-github-cua-ban>/qldmdt-nhom3-quant`)
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Nhấn **"Deploy!"**.
5. Đợi 1 – 2 phút để hệ thống tự động cài đặt thư viện từ `requirements.txt` và khởi chạy web app.
6. Sau khi hoàn tất, bạn sẽ nhận được đường link web app công khai (URL dạng: `https://<ten-ung-dung>.streamlit.app`) để chia sẻ và nộp bài.

---

## 💡 ĐẶC ĐIỂM NỔI BẬT CỦA WEB APP

- **Tự động nhận diện dữ liệu**: Khi mở app, hệ thống tự động tải file `HOSE_2020_2023_in.csv` có sẵn, người dùng không cần thao tác upload thủ công lại.
- **Tùy biến tham số linh hoạt**: Cho phép điều chỉnh trọng số 4 nhân tố (Trend, Momentum, Risk, Liquidity), ngưỡng RSI lọc quá mua, ngưỡng tương quan đa dạng hóa danh mục, chu kỳ SMA định thời điểm và lãi suất phi rủi ro $R_f$.
- **Biểu đồ trực quan hóa cao cấp (Plotly)**:
  - Đường phát triển vốn (Interactive Equity Curve) tương tác phóng to/thu nhỏ (Zoom/Pan).
  - Biểu đồ sụt giảm vốn (Underwater Drawdown Chart).
  - Vùng tín hiệu thị trường (Invested vs 100% Cash Zone).
  - Ma trận tương quan chéo (Correlation Heatmap).
- **Xuất dữ liệu một chạm**: Cho phép tải xuống toàn bộ bảng đối chiếu 8 chỉ số của các chiến lược dưới dạng file `.csv` chuẩn UTF-8.
