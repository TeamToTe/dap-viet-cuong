# 🚰 SKAB: Skoltech Anomaly Benchmark (Valve1 Industrial Circulation Loop)

## 📌 1. Giới thiệu tổng quan
**SKAB (Skoltech Anomaly Benchmark)** là bộ dữ liệu benchmark công nghiệp chuẩn quốc tế được phát triển bởi Viện Khoa học và Công nghệ Skolkovo (Nga). Bộ dữ liệu mô phỏng một trạm thử nghiệm hệ thống tuần hoàn nước công nghiệp khép kín (Industrial Water Circulation Testbed) với các thành phần:
- Bơm ly tâm (Centrifugal pump)
- Hệ thống van tiết lưu và van điều khiển khí nén (Water valves)
- Bồn chứa và hệ thống trao đổi nhiệt
- Mạng lưới cảm biến đo đa chiều theo thời gian thực (8 biến cảm biến)

Tập con **`valve1`** bao gồm **16 lần chạy (run files: `0.csv` đến `15.csv`)**, mô phỏng các trạng thái vận hành bình thường và các lỗi vật lý cố ý đưa vào hệ thống (như đóng van một phần, kẹt van khí nén, hỏng vòng đệm, rò rỉ lưu lượng).

---

## 📊 2. Cấu trúc và Ý nghĩa chi tiết các cột cảm biến

Mỗi file CSV trong `valve1/` bao gồm các trường thông tin sau:

| Tên cột | Đơn vị / Kiểu | Nhóm cảm biến | Ý nghĩa công nghiệp & Vật lý |
| :--- | :--- | :--- | :--- |
| **`datetime`** | `YYYY-MM-DD hh:mm:ss` | Thời gian | Mốc thời gian lấy mẫu của hệ thống SCADA (tần số 1 Hz). |
| **`Accelerometer1RMS`** | $g$ ($m/s^2$) | Rung động | Giá trị hiệu dụng (Root Mean Square - RMS) gia tốc rung động từ cảm biến gia tốc số 1 gắn trên thân bơm/van. |
| **`Accelerometer2RMS`** | $g$ ($m/s^2$) | Rung động | Giá trị hiệu dụng rung động từ cảm biến gia tốc số 2 (đo rung lắc thứ cấp). |
| **`Current`** | Ampe ($A$) | Điện | Dòng điện tiêu thụ của động cơ bơm tuần hoàn (tăng khi quá tải hoặc kẹt van). |
| **`Pressure`** | Bar | Thủy lực / Áp suất | Áp suất chất lỏng trong đường ống dẫn trước/sau cụm van thử nghiệm. |
| **`Temperature`** | $^{\circ}C$ | Nhiệt | Nhiệt độ của chất lỏng tuần hoàn trong hệ thống. |
| **`Thermocouple`** | $^{\circ}C$ | Nhiệt | Nhiệt độ bề mặt cảm biến nhiệt ngẫu (thermocouple) đo trực tiếp tại cụm truyền động van. |
| **`Voltage`** | Volt ($V$) | Điện | Điện áp nguồn cấp cho hệ thống điện động cơ. |
| **`Volume Flow RateRMS`** | $m^3/h$ | Thủy lực / Lưu lượng | Giá trị hiệu dụng lưu lượng thể tích nước tuần hoàn qua đường ống. |
| **`anomaly`** | Binary (`0.0` hoặc `1.0`) | **Nhãn mục tiêu (Target)** | **0.0:** Trạng thái bình thường (Normal).<br>**1.0:** Trạng thái bất thường vật lý (Physical Anomaly / Defect). |
| **`changepoint`** | Binary (`0.0` hoặc `1.0`) | Điểm chuyển tiếp | Đánh dấu thời điểm bắt đầu xảy ra sự thay đổi chế độ vận hành (Transient / Change Point). |

---

## 🔬 3. Chiến lược phân chia dữ liệu cho Generative Anomaly Benchmark

Trong bài toán học sâu không giám sát / bán giám sát (Semi-supervised / Unsupervised Anomaly Detection):
- **Tập huấn luyện (Training Set):** 9 file đầu tiên (`0.csv` đến `8.csv`), **lọc nghiêm ngặt 100% mẫu bình thường (`anomaly == 0.0`)**. Tuyệt đối không cho mô hình sinh nhìn thấy bất kỳ mẫu lỗi nào trong quá trình học.
- **Tập kiểm thử (Test Set):** 7 file độc lập còn lại (`9.csv` đến `15.csv`), chứa cả chuỗi vận hành bình thường và chuỗi phát sinh lỗi thực tế trong nhà máy.

---

## 🔗 4. Nguồn trích dẫn & Tham khảo
- **GitHub Repository chính thức:** [waico/SKAB](https://github.com/waico/SKAB)
- **Bài báo gốc:** Katser, I., & Kozitsin, V. (2020). *Skoltech Anomaly Benchmark (SKAB)*. arXiv:2009.04388.
