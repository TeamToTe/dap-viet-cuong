# 🏭 AI4I 2020: Predictive Maintenance Synthetic Benchmark Dataset

## 📌 1. Giới thiệu tổng quan
**AI4I 2020 Predictive Maintenance Dataset** là bộ dữ liệu công nghiệp tiêu chuẩn được xây dựng và đóng góp vào kho lưu trữ **UCI Machine Learning Repository** bởi Matzka et al. (2020). Bộ dữ liệu phản ánh quá trình gia công phay/tiện công nghiệp thực tế với kích thước **10,000 dòng dữ liệu**, gồm các thông số nhiệt độ, tốc độ quay, mô-men xoắn và hao mòn dao cụ.

Dữ liệu đặc biệt hữu ích cho bài toán **mất cân bằng nhãn cực đoan (Extreme Class Imbalance)**, với tỷ lệ lỗi máy móc thực tế chỉ chiếm **3.39% (339 / 10,000 mẫu)**.

---

## 📊 2. Cấu trúc và Ý nghĩa chi tiết các cột

| Tên cột | Đơn vị / Kiểu | Phân loại | Ý nghĩa công nghiệp & Kỹ thuật |
| :--- | :--- | :--- | :--- |
| **`UDI`** | Integer ($1 - 10000$) | Mã định danh | Unique Device Identifier - Số thứ tự từng chu kỳ đo (thường loại bỏ khi huấn luyện). |
| **`Product ID`** | String (Ví dụ: `M14860`) | Mã sản phẩm | Ký tự đầu đại diện cho cấp chất lượng sản phẩm (`L`: Low, `M`: Medium, `H`: High) kèm mã định danh biến thể. |
| **`Type`** | Categorical (`L`, `M`, `H`) | Cấp sản phẩm | Tỷ lệ sản phẩm sản xuất: `L` (50%), `M` (30%), `H` (20%). |
| **`Air temperature [K]`** | Kelvin ($K$) | Cảm biến vật lý | Nhiệt độ không khí xung quanh máy công cụ (dao động quanh $300\ K \approx 27^\circ C$). |
| **`Process temperature [K]`** | Kelvin ($K$) | Cảm biến vật lý | Nhiệt độ sinh ra trong quá trình phôi tiếp xúc với dao cụ (thường cao hơn nhiệt độ phòng $\approx 10\ K$). |
| **`Rotational speed [rpm]`** | Vòng/phút ($rpm$) | Động lực học | Tốc độ quay của trục chính (Spindle speed). |
| **`Torque [Nm]`** | Newton mét ($Nm$) | Cơ học | Mô-men xoắn tạo ra bởi động cơ chính. |
| **`Tool wear [min]`** | Phút ($min$) | Hao mòn | Tổng thời gian dao cụ đã gia công liên tục. Dao mòn vượt ngưỡng sẽ dẫn đến gãy hoặc phế phẩm. |
| **`Machine failure`** | Binary (`0` hoặc `1`) | **Nhãn mục tiêu (Target)** | **0:** Máy vận hành an toàn (Normal).<br>**1:** Xảy ra sự cố dừng máy (Failure / Anomaly). |
| **`TWF`** | Binary (`0` hoặc `1`) | Chẩn đoán lỗi | Tool Wear Failure - Lỗi do dao cụ bị mòn vượt ngưỡng cho phép. |
| **`HDF`** | Binary (`0` hoặc `1`) | Chẩn đoán lỗi | Heat Dissipation Failure - Lỗi tản nhiệt kém (chênh lệch nhiệt độ $< 8.6\ K$ và tốc độ quay $< 1380\ rpm$). |
| **`PWF`** | Binary (`0` hoặc `1`) | Chẩn đoán lỗi | Power Failure - Lỗi công suất (tích mô-men xoắn và tốc độ quay vượt ngoài khoảng an toàn). |
| **`OSF`** | Binary (`0` hoặc `1`) | Chẩn đoán lỗi | Overstrain Failure - Lỗi quá tải mô-men xoắn kết hợp hao mòn dao cụ. |
| **`RNF`** | Binary (`0` hoặc `1`) | Chẩn đoán lỗi | Random Failure - Lỗi ngẫu nhiên phát sinh ngoài tầm kiểm soát (tần suất $\approx 0.1\%$). |

---

## 🔬 3. Chiến lược phân chia dữ liệu cho Generative Anomaly Benchmark

- **Các biến đầu vào cho mô hình sinh (Features):** 5 biến số thực lực học liên tục:
  1. `Air temperature [K]`
  2. `Process temperature [K]`
  3. `Rotational speed [rpm]`
  4. `Torque [Nm]`
  5. `Tool wear [min]`
- **Tập huấn luyện (Training Set):** $75\%$ mẫu dữ liệu bình thường (`Machine failure == 0`).
- **Tập kiểm thử (Test Set):** $25\%$ dữ liệu được giữ lại độc lập, chứa cả mẫu bình thường và mẫu lỗi máy thật (`Machine failure == 1`) để đánh giá khả năng tổng quát hóa của bộ phân loại downstream (TSTR).

---

## 🔗 4. Nguồn trích dẫn & Tham khảo
- **UCI Machine Learning Repository:** [AI4I 2020 Predictive Maintenance Dataset (ID: 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Bài báo gốc:** Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I).
