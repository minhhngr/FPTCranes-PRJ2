# HƯỚNG DẪN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN (PAGE 04 & PAGE 05)
> **TÀI LIỆU DUY NHẤT & ĐẦY ĐỦ NHẤT:** Toàn bộ nội dung từ điển thuật ngữ lý thuyết và kịch bản thuyết trình 5 bước đã được gộp lại trong 1 file duy nhất:
> 👉 **[`FULL_PRESENTATION_PAGE04_PAGE05.md`](./FULL_PRESENTATION_PAGE04_PAGE05.md)**

---

### TÓM TẮT NHANH CẤU TRÚC FILE DUY NHẤT NÀY:

1. **PHẦN 1: BỘ TỪ ĐIỂN TẤT CẢ KEYWORDS LÝ THUYẾT (PAGE 04 & PAGE 05)**
   * **Nhóm Sai số & Hàm mất mát:** `MAE`, `MSE`, `RMSE`, `MedAE`, `Residual`, `Median Residual`.
   * **Nhóm Độ khớp:** `R²`, `Negative R²` (Bản chất tại sao Linear Regression bị $R^2 = -0.520$), `Test - CV R² Gap` (-0.011).
   * **Nhóm Dữ liệu & Target:** `annual_salary_usd`, `Right-skewed`, `log1p` / `expm1`, `Homoscedasticity`, `Data Leakage`, `Pipeline StandardScaler`.
   * **Nhóm Chia dữ liệu:** `Temporal Sliding Window Cross-Validation`, `Development Pool` (1,201 dòng), `Locked Holdout Test Set` (298 dòng), `Phân biệt Validation vs Test`.
   * **Nhóm 5 Mô hình ứng viên:** `Dummy Median` ($54k), `Linear Regression`, `Ridge L2`, `Random Forest`, `Gradient Boosting`.
   * **Nhóm Siêu tham số Random Forest:** `n_estimators=200`, `max_depth=20`, `min_samples_leaf=1`, `max_features=0.7` (cơ chế de-correlate trees).
   * **Nhóm Tầm quan trọng đặc trưng:** `Permutation Feature Importance` (đo trên raw data bằng USD), `Impurity MDI`, `Feature Drift`, `Ablation Full 13 vs Top-2`.
   * **Nhóm Độ bất định & Kiểm định:** `Empirical q90` (±$40,068), `Coverage` (89.9%), `ECDF`, `Tỷ số RMSE/MedAE = 6.7x`, `Traceable Benchmark Examples`.

2. **PHẦN 2: KỊCH BẢN THUYẾT TRÌNH 5 BƯỚC CHUẨN FORM**
   * **Bước 1:** Xác định Target & Feature engineering `log1p` vì phân phối lệch phải. Khẳng định code Duy đã làm scaling sau khi chia train-test bên trong Pipeline chống rò rỉ dữ liệu.
   * **Bước 2:** Các models sử dụng Dummy Median làm baseline vì trung vị không bị ảnh hưởng bởi đuôi lệch phải. 5 models ứng viên và khác biệt cốt tử giữa Validation và Test.
   * **Bước 3:** Đọc kết quả 5 models. MAE & RMSE thấp, R² cao, MedAE thấp, training time. Đối đầu Random Forest vs Gradient Boosting (RF MAE thấp hơn $759, GB MAE SD thấp hơn).
   * **Bước 4:** Sau tuning Random Forest (n_estimators=200, max_depth=20, min_samples_leaf=1, max_features=0.7 - giải thích cơ chế de-correlate trees).
   * **Bước 5:** Kết quả CV của Full 13-feature tuned model (CV MAE $15,594, CV R² 0.824) và kiểm chứng độc lập trên tập Test (Test MAE $14,735, Test R² 0.813, MedAE $4,347, q90 ±$40,068).

👉 **Xem chi tiết toàn văn tại:** [**`FULL_PRESENTATION_PAGE04_PAGE05.md`**](./FULL_PRESENTATION_PAGE04_PAGE05.md)
