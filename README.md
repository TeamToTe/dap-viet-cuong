# 🏭 Industrial Tabular Anomaly Synthesis Benchmark (Deep Learning)

> **Đề tài:** Sinh dữ liệu bất thường dạng bảng cho dữ liệu công nghiệp sử dụng mô hình học sâu (Tabular Anomaly Data Generation for Industrial Data using Deep Learning).  
> **Kiến trúc:** *Everything is a Plugin* (Dễ dàng mở rộng Dataset, Model, Metric chỉ với 1 cú click / 1 lệnh cmd/bash).

---

## 📌 1. Giới thiệu tổng quan

Dự án tập trung giải quyết bài toán mất cân bằng dữ liệu cực đoan trong môi trường sản xuất công nghiệp (Predictive Maintenance / Anomaly Detection). Bằng cách huấn luyện các mô hình sinh học sâu (Deep Generative Models) trên dữ liệu vận hành bình thường (100% OK samples), hệ thống có khả năng tự động sinh ra các mẫu bất thường có kiểm soát:
1. **TVAE (Tabular VAE):** Sinh lỗi thông qua Outlier Tail Sampling và Latent Directional Shift.
2. **Fence GAN:** Ép Generator sinh mẫu tại biên phân phối chuẩn thông qua hàm mất mát Minimax Boundary Loss $\mathcal{L}_{boundary} + \alpha \mathcal{L}_{disp}$.
3. **TabDDPM (Tabular Denoising Diffusion):** Mô hình khuếch tán SOTA, bảo toàn cấu trúc tương quan phi tuyến đa cảm biến thông qua Perturbed Reverse Guidance.

---

## 📂 2. Cấu trúc thư mục dự án

```text
DAP/
│
├── .venv/                            # Môi trường ảo Python (Virtual Environment)
│
├── figure/                           # Lưu trữ toàn bộ biểu đồ kết quả (PCA, Heatmap, Bar chart, Loss, Dist)
│   ├── 01_pca_projection.png         # Trực quan hóa 2D PCA phân phối Normal, Anomaly & Synthetic
│   ├── 02_correlation_heatmaps.png   # So sánh ma trận tương quan cảm biến & Frobenius Distance
│   ├── 03_downstream_utility.png     # So sánh F1-Score các mô hình downstream (RandomForest, XGBoost)
│   ├── 04_training_losses.png        # Đường cong hàm mất mát (loss curves) theo từng epoch
│   └── 05_feature_distributions.png  # Phân phối mật độ (KDE) của dữ liệu sinh so với dữ liệu gốc theo từng sensor
│
├── configs/                          # Quản lý cấu hình YAML
│   ├── pipeline.yaml                 # Cấu hình chung cho toàn bộ pipeline
│   ├── data/
│   │   ├── skab.yaml                 # Cấu hình SKAB Valve1 Benchmark
│   │   ├── ai4i.yaml                 # Cấu hình AI4I 2020 Predictive Maintenance
│   │   └── custom_template.yaml      # Mẫu cấu hình khi thêm dữ liệu mới
│   └── model/
│       ├── tvae.yaml                 # Tham số TVAE
│       ├── fence_gan.yaml            # Tham số Fence GAN
│       └── tab_ddpm.yaml             # Tham số TabDDPM
│
├── data/                             # Dữ liệu
│   ├── raw/                          # Dữ liệu thô (tự động tải hoặc người dùng thêm vào)
│   ├── processed/                    # Dữ liệu đã chuẩn hóa (StandardScaler)
│   └── synthetic/                    # Dữ liệu sinh ra từ các mô hình
│
├── notebooks/                        # Jupyter Notebooks phục vụ EDA và báo cáo
│   ├── 01_benchmark_generative_ai4i.ipynb
│   ├── 02_semi_supervised_ai4i.ipynb
│   └── 03_denso_skab_benchmark.ipynb
│
├── src/                              # Source code chính (Module hóa)
│   ├── core/                         # Lõi kiến trúc Plugin & Abstract Base Classes
│   │   ├── registry.py               # DATASET_REGISTRY, MODEL_REGISTRY, METRIC_REGISTRY
│   │   ├── base_dataset.py           # Interface chuẩn cho mọi Dataset
│   │   ├── base_model.py             # Interface chuẩn cho mọi Generative Model (fit, sample)
│   │   └── base_metric.py            # Interface chuẩn cho các độ đo đánh giá
│   │
│   ├── data/                         # Data plugins
│   │   └── datasets/
│   │       ├── skab.py               # Plugin bộ dữ liệu SKAB (Auto-download)
│   │       ├── ai4i.py               # Plugin bộ dữ liệu AI4I 2020
│   │       └── generic_csv.py        # Plugin nạp BẤT KỲ file CSV công nghiệp nào
│   │
│   ├── models/                       # Generative model plugins
│   │   ├── tvae.py                   # TVAE implementation
│   │   ├── fence_gan.py              # Fence GAN implementation
│   │   └── tab_ddpm.py               # TabDDPM diffusion implementation
│   │
│   ├── metrics/                      # Metric plugins
│   │   ├── fidelity.py               # KS-Test, Wasserstein Distance (1D marginal distribution)
│   │   ├── correlation.py            # Frobenius Norm Difference (Pearson & Spearman)
│   │   └── utility.py                # Train on Synthetic, Test on Real (TSTR F1/AUC)
│   │
│   └── utils/
│       └── visualizer.py             # Tự động xuất biểu đồ độ phân giải cao vào `figure/`
│
├── experiments/                      # Kết quả thí nghiệm
│   ├── checkpoints/                  # Lưu trọng số mô hình (.pt)
│   └── logs/                         # Lưu log huấn luyện & file kết quả benchmark CSV
│
├── scripts/                          # Entrypoints
│   └── run_pipeline.py               # Script chạy toàn bộ pipeline từ A -> Z
│
├── run_pipeline.bat                  # Chạy 1 click trên Windows CMD
├── run_pipeline.sh                   # Chạy 1 click trên Linux / Git Bash
├── requirements.txt                  # Danh sách thư viện phụ thuộc
├── .gitignore                        # Cấu hình bỏ qua file rác, dữ liệu lớn, .venv
└── README.md                         # Tài liệu hướng dẫn sử dụng
```

---

## ⚡ 3. Hướng dẫn cài đặt nhanh

### Bước 1: Kích hoạt môi trường ảo `.venv`
- **Trên Windows (cmd / powershell):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Trên Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## 🚀 4. Chạy toàn bộ Benchmark chỉ với 1 lệnh

Hệ thống được thiết kế theo tư duy **"One-Command Pipeline"**:

### Cách 1: Sử dụng file script 1-click
- **Windows:**
  ```cmd
  run_pipeline.bat
  ```
- **Linux / Git Bash:**
  ```bash
  bash run_pipeline.sh
  ```

### Cách 2: Chạy trực tiếp qua Python với các tùy chọn

1. **Chạy trên bộ dữ liệu SKAB (Mặc định):**
   ```bash
   python scripts/run_pipeline.py --dataset skab --epochs 20
   ```

2. **Chạy trên bộ dữ liệu AI4I 2020 Predictive Maintenance:**
   ```bash
   python scripts/run_pipeline.py --dataset ai4i --epochs 25
   ```

4. **Đặt tên thí nghiệm riêng biệt (Isolated Experiment Runs):**
   ```bash
   python scripts/run_pipeline.py --dataset skab --exp_name exp_skab_tuned_lr0001
   ```
   *(Nếu không truyền `--exp_name`, hệ thống sẽ tự động sinh mã run định danh kèm timestamp: `<dataset>_YYYYMMDD_HHMMSS`)*

Toàn bộ quy trình sẽ tự động:
1. Tạo thư mục kết quả và biểu đồ riêng biệt cho từng lần chạy:
   - Thư mục thí nghiệm: `experiments/runs/<run_name>/` (chứa `checkpoints/`, `benchmark_summary.csv`, `utility_results.csv`, `run_config.json`).
   - Thư mục biểu đồ: `figure/<run_name>/` (toàn bộ 5 biểu đồ của lần chạy đó).
   - Thư mục xem nhanh: `figure/latest/` (tự động đồng bộ kết quả của lần chạy gần nhất).
   - Bảng tổng hợp chéo mọi lần chạy: `experiments/runs_leaderboard.csv` (dễ dàng theo dõi tiến độ và so sánh các lần thử nghiệm khác nhau).
1. Tải và tiền xử lý dữ liệu, lọc mẫu bình thường cho tập huấn luyện.
2. Huấn luyện lần lượt TVAE, Fence GAN, TabDDPM.
3. Sinh dữ liệu bất thường và tính toán:
   - **Fidelity:** KS-Statistic & Wasserstein Distance.
   - **Correlation:** Frobenius difference cho ma trận Pearson/Spearman.
   - **Utility:** F1-score & ROC-AUC trên bài toán phát hiện lỗi thực tế (RandomForest, XGBoost).
4. Xuất bảng tổng kết ra `experiments/benchmark_*.csv`.
5. Tự động vẽ và lưu 3 biểu đồ chuẩn báo cáo khoa học vào thư mục `figure/`.

---

## 🧩 5. Kiến trúc "Everything is a Plugin" (Cách mở rộng)

### 🔹 Muốn thêm một bộ dữ liệu mới?
Chỉ cần tạo một file trong `src/data/datasets/my_data.py`:
```python
from src.core.registry import register_dataset
from src.core.base_dataset import BaseDataset, TabularDataResult

@register_dataset("my_new_dataset")
class MyDataset(BaseDataset):
    def load(self) -> TabularDataResult:
        # Đọc dữ liệu, chia train/test và trả về TabularDataResult
        return TabularDataResult(...)
```
Sau đó chỉ cần chạy: `python scripts/run_pipeline.py --dataset my_new_dataset`.

### 🔹 Muốn thêm một mô hình sinh mới (ví dụ CTGAN, WGAN-GP)?
Chỉ cần tạo một file trong `src/models/wgan.py`:
```python
from src.core.registry import register_model
from src.core.base_model import BaseGenerativeModel

@register_model("wgan")
class WGANModel(BaseGenerativeModel):
    def fit(self, X_train, epochs=20, **kwargs):
        # Huấn luyện mô hình WGAN
        pass

    def sample(self, n_samples, **kwargs):
        # Trả về mảng numpy (n_samples, n_features)
        return samples
```
Khi chạy pipeline, mô hình mới sẽ tự động được đưa vào danh sách benchmark:
```bash
python scripts/run_pipeline.py --models tvae fence_gan tab_ddpm wgan
```

---

## 📊 6. Kết quả trực quan hóa trong thư mục `figure/`

Sau khi chạy xong, thư mục `figure/` sẽ chứa:
1. `01_pca_projection.png`: Kiểm tra vị trí phân bố của dữ liệu sinh so với vùng dữ liệu bình thường và bất thường thực tế.
2. `02_correlation_heatmaps.png`: Ma trận nhiệt kiểm tra mức độ giữ gìn liên kết vật lý giữa các cảm biến.
3. `03_downstream_utility.png`: Biểu đồ cột thể hiện sự vượt trội về F1-Score khi áp dụng dữ liệu sinh vào bộ phân loại.
4. `04_training_losses.png`: Biểu đồ đường cong hàm mất mát (loss convergence curves) theo từng epoch huấn luyện của TVAE, Fence GAN, TabDDPM.
5. `05_feature_distributions.png`: Biểu đồ phân phối mật độ xác suất (KDE) của dữ liệu sinh so với phân phối dữ liệu gốc theo từng cảm biến (Sensor features).
