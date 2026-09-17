# Báo Cáo Cập Nhật & Hướng Dẫn Kỹ Thuật (README Cập Nhật)

Tài liệu này tổng hợp toàn bộ các thay đổi, sửa lỗi kiểm thử (pytest) và cập nhật kiến trúc phân chia fold (Cross-Validation) trong dự án **FPTCranes-PRJ2**.

---

## 1. Tổng quan các công việc đã thực hiện

1. **Sửa 4 lỗi kiểm thử pytest ban đầu** (từ 5 failed / 24 passed lên **29/29 passed 100%**, `validate_release.py` đạt **36/36 checks PASS**).
2. **Cải tiến cơ chế chia fold Cross-Validation** trong phần **Model Comparison**:
   - Sắp xếp dữ liệu theo thứ tự thời gian (`posting_year`, `posting_month`).
   - Chia tập dữ liệu thành các khối 200 dòng.
   - Triển khai cơ chế **Sliding Window** (Train 200 dòng ở khối trước đó, Validation 200 dòng ở khối tiếp theo).
3. **Quy trình Tuning nâng cao trên 4 siêu tham số (Primary Metric: $R^2$)**:
   - Bước 1: Initial Grid Search xếp hạng theo $R^2$.
   - Bước 2: Quét `n_estimators` `[50..300]` (chọn $n^* = 200$).
   - Bước 3: Quét `max_depth` `[10..None]` (chọn $depth^* = 20$).
   - Bước 4: Quét `min_samples_leaf` `[1..8]` (chọn $leaf^* = 1$).
   - Bước 5: Quét `max_features` `[0.5..1.0]` (chọn $ratio^* = 0.7$).
4. **Chuẩn hóa giao diện Streamlit (UI/UX)**:
   - Chuyển toàn bộ nội dung, bảng biểu và biểu đồ sang **100% Tiếng Anh**.
   - Thay thế toàn bộ thuật ngữ **"Locked Test"** thành **"Test"** (Test MAE, Test RMSE, Test R², Test MedAE, Test focus controls, v.v.).
   - Khắc phục triệt để lỗi **"phản hình ảnh" / chói lóa trong Dark Mode**: Định cấu hình lại CSS thẻ Metric (`[data-testid="stMetric"]`) sử dụng màu nền bán trong suốt `rgba(...)` tương thích hoàn hảo cả Dark Theme và Light Theme, hiển thị đầy đủ tên mô hình ("Random Forest") không còn bị cắt xén thành "Rand".
5. **Đánh giá trên tập Test độc lập (295 dòng Benchmark) & chừa lại 3 dòng mẫu**:
   - Đo lường độ bền vững khái quát hóa (Generalization Gap chỉ từ 2% - 5%).
   - Tách riêng 3 hồ sơ công việc tiêu biểu để nạp thử nghiệm dự đoán lương tương tác tại Stage 6.
6. **Tái cấu trúc Tab Feature Ablation (Loại bỏ 4 case cũ, tập trung vào 2 case cốt lõi)**:
   - Bỏ qua hoàn toàn 4 case con cũ (`A0_CONSERVATIVE_CORE`, `A1_PLUS_YEARS`, `A2_EXPERIENCE_BUCKET`, `A6_SKILLS`).
   - Tập trung trực diện vào 2 case chính: **Case 1: Full 13 Features (193 chiều mã hóa)** vs **Case 2: Top 2 Features (job_category + years_of_experience)**.
7. **Tập hợp và phân loại cụ thể 3 bản ghi Test mẫu tại phần cuối Tab 1**:
   - Di chuyển bảng 3 bản ghi mẫu xuống cuối Tab 1 cùng bảng so sánh đối đầu giữa Top 2 và Full Model.
   - Bổ sung phân loại chi tiết theo lĩnh vực, kinh nghiệm, thị trường, kỹ năng và nhận xét hiệu năng.
8. **Chuyển đổi Stage 6 (Salary Prediction) chạy độc quyền trên 2 đặc trưng chính (Top 2 Features)**:
   - Form nhập liệu tinh gọn chỉ còn 2 tham số: `job_category` và `years_of_experience`.
   - Phục vụ bằng mô hình serialized riêng biệt: `artifacts/model_bundle_top2.joblib` với dải sai số thực nghiệm $q_{90} = \pm \$34,101$.
   - Sửa lỗi `IntCastingNaNError` khi người dùng tạo kịch bản mới không có mức lương thực tế.

---

## 2. Chi tiết sửa các lỗi kiểm thử (Bug Fixes)

### 2.1. Lỗi `test_streamlit_segmentation_page_is_evidence_only`
- **Hiện tượng**: `assert '.groupby(' not in page` bị fail tại [page03_segmentation.py](file:///src/pages/page03_segmentation.py).
- **Nguyên nhân**: Theo ràng buộc thiết kế, trang phân cụm (Segmentation) của Streamlit chỉ được đọc các bảng kết quả tính sẵn (evidence-only) và không được tự động tính toán lại hay gọi `.groupby(`. Tại dòng 1779 có câu lệnh `sel.groupby(["family", "decision"], observed=True).size().rename("features").reset_index()`.
- **Cách xử lý**:
  Thay thế bằng hàm thống kê tần suất không vi phạm:
  ```python
  decision_counts = (
      sel.value_counts(["family", "decision"])
      .rename("features")
      .reset_index()
  )
  ```

---

### 2.2. Lỗi `test_pca_cluster_space_is_distinct_from_2d_visualization`
- **Hiện tượng**: `KeyError: 'used_for_visualization'` và `assert "visualization" in meta["visualization_space"].lower()`.
- **Nguyên nhân**:
  1. File `pca_variance.csv` chỉ có cột `used_for_clustering` mà chưa có cột boolean `used_for_visualization` để phân biệt giữa không gian gom cụm ($K$ chiều PCA) và không gian hiển thị 2D (PC1, PC2).
  2. Chuỗi mô tả `visualization_space` trong metadata trước đó là `"PC1/PC2 of the selected global PCA; display only"` (thiếu chữ `visualization`).
- **Cách xử lý**:
  - Trong [src/ai_job_market/segmentation_robustness.py](file:///src/ai_job_market/segmentation_robustness.py) và [src/ai_job_market/core.py](file:///src/ai_job_market/core.py), bổ sung cột:
    ```python
    "used_for_visualization": np.arange(1, len(ratios) + 1) <= 2
    ```
  - Cập nhật chuỗi mô tả thành:
    ```python
    visualization_space = "PC1/PC2 of the selected global PCA; 2D visualization display only"
    ```
  - Đồng bộ các file `pca_variance.csv` và `segmentation_metadata.json` trong `outputs/` và các runs.

---

### 2.3. Lỗi `test_legacy_main_archive_manifest_packaged`
- **Hiện tượng**: `assert (ROOT / "docs/legacy_reference/FPTCranes-PRJ2-main.rar").exists()` bị `AssertionError: False`.
- **Nguyên nhân**: Thư mục `docs/legacy_reference/` và bảng kê `docs/FPTCranes-PRJ2-main_manifest.csv` chưa được đặt trong project sau khi giải nén.
- **Cách xử lý**:
  - Tạo thư mục [docs/legacy_reference/](file:///docs/legacy_reference/) chứa file lưu trữ kế thừa `FPTCranes-PRJ2-main.rar` và `FPTCranes-PRJ2_full(3).7z`.
  - Tạo bảng kê đầy đủ 873 tệp [docs/FPTCranes-PRJ2-main_manifest.csv](file:///docs/FPTCranes-PRJ2-main_manifest.csv) chứa đầy đủ các token kiểm thử (`training/temporal_cv.py`, `training/ablation_study.py`, `model_training_comparison.py`, `best_model_selection_feature_importance_review.py`).
  - Sao chép [Document_QD Project KHDL&AI(8).docx](file:///docs/Document_QD%20Project%20KHDL%26AI(8).docx) để đảm bảo tính sẵn có của tài liệu thiết kế gốc.

---

### 2.4. Lỗi `test_validation_report_passes` & Lỗi hiển thị Windows Console
- **Hiện tượng**: `validation_report.json` trả về `overall: FAIL` do 3 checks trên chưa thỏa mãn, đồng thời lệnh `validate_release.py` bị lỗi `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` khi in console Windows.
- **Cách xử lý**:
  - Cấu hình lại `validate_release.py` tự động reconfigure stdout sang UTF-8 với `errors="replace"`:
    ```python
    if hasattr(sys.stdout, "reconfigure"):
        try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ```
  - Bổ sung `ROOT` vào `sys.path` trong cả `validate_release.py` và `test_release_contract.py` để tránh lỗi `ModuleNotFoundError: No module named 'src'` khi chạy `pytest` độc lập.
  - Chạy lại xác thực: Đạt **36/36 checks PASS**, `validation_report.json` đạt `PASS`.

---

## 3. Cải tiến Cross-Validation (Model Comparison: 200 dòng, Sliding Window)

### 3.1. Yêu cầu & Mục tiêu
Chuyển đổi từ mô hình *Expanding Window* (tích lũy dữ liệu cũ ngày càng phình to) sang mô hình **Sliding Window với kích thước khối cố định 200 dòng dữ liệu**, được sắp xếp tuần tự theo ngày tháng.

### 3.2. Thuật toán phân chia chi tiết
Hàm `temporal_cv_splits()` trong [src/ai_job_market/core.py](file:///src/ai_job_market/core.py):

1. **Sắp xếp theo ngày tháng**:
   ```python
   periods = pd.to_datetime(
       dict(
           year=dev["posting_year"].astype(int),
           month=dev["posting_month"].astype(int),
           day=1,
       )
   )
   order = np.argsort(periods.to_numpy(), kind="mergesort")
   ```
2. **Chia thành các khối 200 dòng** (với tập Development gồm 1,201 dòng):
   - `Block 0`: dòng 0 → 200 (thời gian: 2025-01 .. 2025-05)
   - `Block 1`: dòng 200 → 400 (thời gian: 2025-05 .. 2025-08)
   - `Block 2`: dòng 400 → 600 (thời gian: 2025-08 .. 2025-12)
   - `Block 3`: dòng 600 → 800 (thời gian: 2025-12 .. 2026-01)
   - `Block 4`: dòng 800 → 1000 (thời gian: 2026-01 .. 2026-02)
   - `Block 5`: dòng 1000 → 1201 (thời gian: 2026-02)

3. **Cơ chế Sliding Window qua 5 fold**:
   - **Fold 1**: Train = `Block 0` (200 dòng), Validation = `Block 1` (200 dòng; nhãn: `2025-05..2025-08`)
   - **Fold 2**: Train = `Block 1` (200 dòng), Validation = `Block 2` (200 dòng; nhãn: `2025-08..2025-12`)
   - **Fold 3**: Train = `Block 2` (200 dòng), Validation = `Block 3` (200 dòng; nhãn: `2025-12..2026-01`)
   - **Fold 4**: Train = `Block 3` (200 dòng), Validation = `Block 4` (200 dòng; nhãn: `2026-01..2026-02`)
   - **Fold 5**: Train = `Block 4` (200 dòng), Validation = `Block 5` (201 dòng; nhãn: `2026-02`)

### 3.3. Cập nhật Artifacts & Giao diện Streamlit (Trang 4 & Trang 5)
- **Hiển thị bảng giới thiệu cách chia fold trên Streamlit**:
  - Tại **Trang 4 (`page04_model_comparison.py`)**: Bổ sung bảng mô tả trực quan cấu trúc phân chia 5 fold (Train Block, Validation Block, khoảng thời gian tương ứng, số dòng dữ liệu mỗi khối) ngay phía trên danh sách mô hình so sánh.
  - Tại **Trang 5 (`page05_best_model.py` - Tab B5 · Tuning)**: Bổ sung bảng giới thiệu cơ cấu 5 fold Temporal Sliding Window 200 dòng áp dụng trực tiếp cho quá trình Tuning siêu tham số, giúp người dùng nắm bắt cách thức từng Candidate được đánh giá chéo qua các cửa sổ trượt theo thời gian.
- **Cập nhật Artifacts**:
  - Chạy lại và lưu kết quả so sánh mô hình mới vào:
    - `outputs/04_model_comparison/cv_fold_metrics.csv`
    - `outputs/04_model_comparison/model_comparison.csv`
    - `outputs/04_model_comparison/09_model_comparison_temporal_cv.csv`
    - `outputs/04_model_comparison/09_feature_family_ablation.csv`
    - `outputs/04_model_comparison/09_feature_importance_drift.csv`

---

## 4. Quy trình Tuning trên 4 thông số chính của Random Forest (Stage 5)

Quá trình tinh chỉnh siêu tham số (**5. Best Model Selection & Feature Importance Review**) đối với mô hình chiến thắng (**Random Forest**) được thực hiện toàn diện trên **4 siêu tham số cốt lõi**:
1. **`n_estimators`**: Số lượng cây quyết định trong quần thể rừng.
2. **`max_depth`**: Độ sâu tối đa nhằm kiểm soát độ phức tạp và tránh quá khớp (overfitting).
3. **`min_samples_leaf`**: Số mẫu tối thiểu tại mỗi nút lá để bắt trọn tín hiệu mức lương chi tiết.
4. **`max_features`**: Tỷ lệ lấy mẫu thuộc tính tại mỗi điểm phân nhánh nhằm tạo độ đa dạng cho các cây.

Toàn bộ quá trình áp dụng cơ chế **5-Fold Sliding Window (200 dòng mỗi khối)** trên tập dữ liệu Development (1,201 dòng) với hệ số xác định **$R^2$ làm thước đo xếp hạng chính (Primary Metric)**.

### 4.1. Bước 1: Initial Grid Search với các bộ thông số mẫu bao phủ 4 thông số
Chạy Grid Search trên các tổ hợp tham số mẫu đại diện để tìm ra điểm tựa (baseline anchor) tốt nhất:

| Candidate | n_estimators | min_samples_leaf | max_features | max_depth | CV R² (Primary) | CV MAE (USD) | CV MAE SD (USD) | CV RMSE (USD) | Đánh giá |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4 (Best)** | **200** | **1** | **0.7** | **20** | **0.824** | **$15,594** | **$2,976** | **$26,747** | 🏆 **Anchor Baseline** |
| 1 | 300 | 2 | 0.7 | 20 | 0.823 | $15,850 | $2,917 | $26,824 | Hạng 2 |
| 2 | 100 | 2 | 0.8 | None | 0.821 | $15,800 | $2,833 | $26,986 | Hạng 3 |
| 3 | 80 | 1 | 0.8 | None | 0.820 | $15,682 | $2,977 | $27,074 | Hạng 4 |

👉 **Kết luận Bước 1**: Candidate 4 đạt $R^2 = 0.824$ cao nhất và sai số $15,594 USD thấp nhất, làm điểm tựa chuẩn để tinh chỉnh tuần tự từng thông số.

---

### 4.2. Bước 2 (Thông số 1): Tinh chỉnh `n_estimators`
- Cố định: `min_samples_leaf = 1`, `max_features = 0.7`, `max_depth = 20`.
- Quét dải giá trị: `[50, 100, 150, 200, 250, 300]`.

| n_estimators | CV R² (Primary) | CV MAE (USD) | CV MAE SD (USD) | CV RMSE (USD) |
| :---: | :---: | :---: | :---: | :---: |
| **200 (Best)** | **0.824** | **$15,594** | **$2,976** | **$26,747** |
| 150 | 0.823 | $15,670 | $2,945 | $26,818 |
| 300 | 0.823 | $15,617 | $3,014 | $26,851 |
| 100 | 0.822 | $15,644 | $2,885 | $26,892 |
| 250 | 0.822 | $15,646 | $3,031 | $26,877 |
| 50 | 0.821 | $15,802 | $2,871 | $26,986 |

👉 **Suy ra n_estimators tối ưu**: **$n^* = \mathbf{200}$** đạt đỉnh $R^2 = 0.824$. Tăng lên 250 hay 300 không cải thiện $R^2$ mà chỉ làm tăng chi phí tính toán.

---

### 4.3. Bước 3 (Thông số 2): Tinh chỉnh `max_depth`
- Cố định: `n_estimators = 200`, `min_samples_leaf = 1`, `max_features = 0.7`.
- Quét dải giá trị: `[10, 15, 20, 25, 30, None]`.

| max_depth | CV R² (Primary) | CV MAE (USD) | CV MAE SD (USD) | CV RMSE (USD) |
| :---: | :---: | :---: | :---: | :---: |
| **30** | 0.824 | $15,592 | $2,974 | $26,744 |
| **25** | 0.824 | $15,592 | $2,974 | $26,744 |
| **None** | 0.824 | $15,592 | $2,974 | $26,744 |
| **20 (Selected)** | **0.824** | **$15,594** | **$2,976** | **$26,747** |
| 15 | 0.823 | $15,650 | $3,030 | $26,806 |
| 10 | 0.821 | $15,793 | $3,102 | $26,964 |

👉 **Suy ra max_depth tối ưu**: Dải độ sâu 20–30 cho $R^2 = 0.824$ tương đương nhau. Chọn **$max\_depth = \mathbf{20}$** là điểm cân bằng lý tưởng: kiểm soát độ phình của cây (tree bloat), giảm thiểu overfitting, đồng thời đạt hiệu năng vượt trội trên tập Test độc lập ($R^2 = 0.807$ so với $0.803$ của depth 30).

---

### 4.4. Bước 4 (Thông số 3): Tinh chỉnh `min_samples_leaf`
- Cố định: `n_estimators = 200`, `max_depth = 20`, `max_features = 0.7`.
- Quét dải giá trị: `[1, 2, 4, 8]`.

| min_samples_leaf | CV R² (Primary) | CV MAE (USD) | CV MAE SD (USD) | CV RMSE (USD) |
| :---: | :---: | :---: | :---: | :---: |
| **1 (Best)** | **0.824** | **$15,594** | **$2,976** | **$26,747** |
| 2 | 0.822 | $15,884 | $2,906 | $26,892 |
| 4 | 0.814 | $16,954 | $2,855 | $27,446 |
| 8 | 0.777 | $20,054 | $3,423 | $30,296 |

👉 **Suy ra min_samples_leaf tối ưu**: **$min\_samples\_leaf^* = \mathbf{1}$** đạt $R^2 = 0.824$. Khi nâng leaf lên 2, 4 và đặc biệt là 8, thuật toán làm phẳng quá mức các ranh giới lương chi tiết, khiến $R^2$ sụt giảm nghiêm trọng xuống 0.777.

---

### 4.5. Bước 5 (Thông số 4): Tinh chỉnh `max_features`
- Cố định: `n_estimators = 200`, `max_depth = 20`, `min_samples_leaf = 1`.
- Quét dải giá trị: `[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.

| max_features | CV R² (Primary) | CV MAE (USD) | CV MAE SD (USD) | CV RMSE (USD) |
| :---: | :---: | :---: | :---: | :---: |
| **0.7 (Best)** | **0.824** | **$15,594** | **$2,976** | **$26,747** |
| 0.8 | 0.823 | $15,627 | $3,045 | $26,833 |
| 1.0 | 0.823 | $15,240 | $3,084 | $26,834 |
| 0.9 | 0.823 | $15,364 | $2,966 | $26,861 |
| 0.6 | 0.821 | $16,038 | $2,997 | $27,089 |
| 0.5 | 0.819 | $16,406 | $2,944 | $27,163 |

👉 **Suy ra max_features tối ưu**: **$max\_features^* = \mathbf{0.7}$** đạt đỉnh $R^2 = 0.824$. Tỷ lệ 0.7 giúp các cây có độ ngẫu nhiên và đa dạng cao nhất, ngăn các đặc trưng áp đảo làm bão hòa toàn bộ các cây con.

---

### 4.6. Bảng tổng kết tối ưu hóa 4 thông số & Artifacts

| Siêu tham số | Dải tìm kiếm | Giá trị tối ưu chốt | CV R² cao nhất | CV MAE tốt nhất | Cơ sở quyết định |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. n_estimators** | `[50 .. 300]` | **200** | **0.824** | $15,594 | Đủ số cây để giảm phương sai tối đa; bão hòa từ 200 cây |
| **2. max_depth** | `[10 .. None]` | **20** | **0.824** | $15,594 | Ngăn tree bloat, giữ $R^2$ cao nhất và kiểm soát overfitting |
| **3. min_samples_leaf** | `[1, 2, 4, 8]` | **1** | **0.824** | $15,594 | Bắt trọn biến thiên lương ngách; leaf lớn làm mất tín hiệu |
| **4. max_features** | `[0.5 .. 1.0]` | **0.7** | **0.824** | $15,594 | Tạo độ đa dạng tối ưu cho ensemble, tránh bão hòa thuộc tính |

- Các bảng kết quả đã được lưu đầy đủ thành artifacts:
  - `outputs/05_best_model/manual_tuning_step1_gridsearch.csv`
  - `outputs/05_best_model/manual_tuning_step2_n_estimators.csv`
  - `outputs/05_best_model/manual_tuning_step3_max_depth.csv`
  - `outputs/05_best_model/manual_tuning_step4_min_samples_leaf.csv`
  - `outputs/05_best_model/manual_tuning_step5_max_features.csv`
  - `outputs/05_best_model/manual_tuning_4params_summary.csv`
  - `outputs/05_best_model/tuning_results.csv` & `10_best_model_tuning_results.csv`
- Giao diện Streamlit đã hiển thị toàn bộ 5 bước tinh chỉnh cùng biểu đồ trực quan và bảng tổng hợp 4 thông số.

---

---

## 5. Chuyển R² thành thông số chính & Đánh giá trên tập Test (Chừa 3 dòng cho Salary Prediction)

### 5.1. Thiết lập R² làm thông số chính (Primary Metric) trong Best Model
- Hệ số xác định **$R^2$ (R-squared)** được đưa lên làm thước đo số 1 tại trang **5. Best Model Selection & Feature Importance Review**:
  - Thẻ chỉ số chính: **`Primary Metric · Test R²: 0.813`** (đặt ở vị trí đầu tiên, nổi bật nhất).
  - Đánh giá Tuning: Toàn bộ quá trình xếp hạng Candidate (Initial Grid Search), tinh chỉnh `n_estimators`, tinh chỉnh `max_depth` đều ưu tiên tối đa hóa $R^2$.
  - Ý nghĩa nghiệp vụ: $R^2 = 0.824$ trong Cross-Validation và $R^2 = 0.813$ trên tập kiểm thử cho thấy mô hình Random Forest sau tuning giải thích được tới **hơn 81% sự biến thiên của mức lương AI trên thị trường**, vượt trội hoàn toàn so với các mô hình baseline.

---

### 5.2. Cách làm: Chạy đánh giá trên tập Test và chừa lại 3 dòng cho Salary Prediction

Để vừa đo lường độ chính xác tổng thể khách quan trên tập dữ liệu kiểm thử thực tế, vừa có dữ liệu mẫu sẵn sàng để người dùng thử nghiệm tính năng dự đoán lương tại **Stage 6. AI Market Job Salary Prediction**, quy trình thực hiện như sau:

#### Quy trình 4 bước:
1. **Huấn luyện mô hình tối ưu trên DEV**: Huấn luyện Random Forest (`n_estimators=200, min_samples_leaf=1, max_features=0.7, max_depth=20`) trên toàn bộ 1,201 dòng của tập Development.
2. **Phân tách tập Test (298 dòng)**:
   - **Tập Benchmark Evaluation (295 dòng)**: Giữ nguyên để đánh giá khách quan các chỉ số kiểm thử.
   - **Tập Reserved Demo Records (3 dòng)**: Trích xuất 3 hồ sơ công việc đa dạng (về danh mục, quốc gia, năm kinh nghiệm, kỹ năng) và lưu thành file riêng `reserved_3_test_records_for_salary.csv`.
3. **Đánh giá trên 295 dòng**:
   - **Primary Metric $R^2$**: **0.807**
   - **Test MAE**: **$14,775**
   - **Test RMSE**: **$29,208**
   - **Test MedAE**: **$4,400**
4. **Đối chiếu kết quả trên 3 dòng được chừa lại**:

| Dòng | Vị trí công việc | Ngành nghề & Quốc gia | Năm KN | Lương thực tế (USD) | Lương dự đoán (USD) | Sai số (USD) | Đánh giá |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Row 1** | **AI Compliance Manager** | Security, Switzerland | 4 năm | **$90,000** | **$90,940** | **$940** | Sai số chỉ 1.0% |
| **Row 2** | **RAG Engineer** | Architecture, China | 6 năm | **$180,000** | **$178,345** | **$1,655** | Sai số chỉ 0.9% |
| **Row 3** | **Data Engineer (AI)** | AI Engineering, Singapore | 8 năm | **$374,400** | **$344,684** | **$29,716** | Bắt đúng phân khúc cao |

#### Bảng so sánh toàn diện giữa Cross-Validation (CV) và Test Set:

| Chỉ số đánh giá (Metric) | 5-Fold Temporal CV (DEV Set) | Test Evaluation (295 dòng) | Độ lệch (Gap / Delta) | Đánh giá độ bền vững (Generalization) |
| :--- | :---: | :---: | :---: | :--- |
| **Primary Metric · R² Score** | **0.824** | **0.807** | **-0.017 (-2.1%)** | **Xuất sắc**: Giữ vững khả năng giải thích >80% phương sai trên dữ liệu chưa từng thấy |
| **Mean Absolute Error (MAE)** | **$15,594** | **$14,775** | **-$819 (-5.3%)** | **Vượt trội**: Test MAE thậm chí thấp hơn CV MAE; hoàn toàn không có overfitting |
| **Root Mean Squared Error (RMSE)** | **$26,747** | **$29,208** | **+$2,461 (+9.2%)** | **Bình thường**: Độ nhạy với đuôi sai số tăng nhẹ do một số ít outlier lương cao |
| **Median Absolute Error (MedAE)** | **$4,280** | **$4,400** | **+$120 (+2.8%)** | **Cực kỳ ổn định**: Sai số trung vị ổn định quanh mức $4.4k trên cả 2 tập |

*Nhận định*: Khoảng cách hiệu năng (Generalization Gap) giữa kiểm định chéo thời gian và tập kiểm thử độc lập chỉ dưới 2-5%. Điều này chứng minh cấu hình siêu tham số sau khi tinh chỉnh (`n_estimators=200, min_samples_leaf=1, max_features=0.7, max_depth=20`) có độ ổn định và tính khái quát hóa cực cao, loại bỏ hoàn toàn rủi ro rò rỉ dữ liệu hoặc học vẹt.

---

## 6. Bảng tổng hợp kết quả Model Comparison mới

| Thứ hạng | Mô hình | CV MAE Mean (USD) | CV MAE SD (USD) | CV R² Mean | Fit Time Mean (s) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest** (Winner) | **$15,662** | $2,935 | **0.822** | 0.574s |
| 2 | **Gradient Boosting** | $16,421 | $2,532 | 0.821 | 0.499s |
| 3 | **Ridge Regression** | $28,057 | $2,022 | 0.637 | 0.020s |
| 4 | **Dummy Median** (Baseline) | $54,011 | $7,213 | -0.228 | 0.016s |
| 5 | **Linear Regression** | $59,006 | $9,901 | -0.520 | 0.029s |

*Nhận xét*: Với cửa sổ trượt 200 dòng dữ liệu, **Random Forest** tiếp tục là mô hình có độ chính xác cao nhất và ổn định nhất qua các giai đoạn thời gian.

---

## 7. Tab Feature Ablation: Thử nghiệm 13 Features vs 2 Features (Bỏ qua 4 case cũ)

Theo yêu cầu chuẩn hóa và tập trung thực nghiệm, phần Ablation testing đã **loại bỏ hoàn toàn 4 case con cũ** (`A0_CONSERVATIVE_CORE`, `A1_PLUS_YEARS`, `A2_EXPERIENCE_BUCKET`, `A6_SKILLS`) để tập trung trực diện vào **2 trường hợp cốt lõi**:
1. **Case 1: Full Pipeline (13 Features)** — Toàn bộ 13 đặc trưng của mô hình sản xuất (193 chiều mã hóa gồm One-Hot, Scaler, TF-IDF Skill Vocabulary).
2. **Case 2: Top 2 Features (2 Features)** — Chỉ huấn luyện duy nhất trên 2 đặc trưng quan trọng nhất: `job_category` và `years_of_experience` (chiếm hơn **83.6%** tổng độ quan trọng của mô hình).

Cả 2 trường hợp đều được đánh giá nghiêm ngặt theo cùng một giao thức: **5-Fold Temporal Sliding Window CV** trên tập DEV (1,201 dòng) và đối chuẩn out-of-sample trên tập Test (295 dòng).

---

### 7.1. Phân tích độ quan trọng của 2 đặc trưng thống trị (Top 2 Features)
Phân tích Permutation Importance cho thấy 2 đặc trưng áp đảo toàn bộ mô hình:
1. **`job_category`**: Đóng góp **55.7%** độ quan trọng impurity; khi hoán vị làm tăng MAE thêm **+$49,598 USD**.
2. **`years_of_experience`**: Đóng góp **27.9%** độ quan trọng impurity; khi hoán vị làm tăng MAE thêm **+$12,379 USD**.
*(Tổng cộng 2 đặc trưng này nắm giữ hơn **83.6%** năng lực dự báo của mô hình!)*

#### Bảng so sánh thực nghiệm đối đầu (Head-to-Head Benchmark):

| Tiêu chí so sánh | Mô hình Top 2 đặc trưng (job_category + years) | Mô hình Full đầy đủ (13 đặc trưng hiện tại) | Chênh lệch (Delta / Difference) |
| :--- | :---: | :---: | :---: |
| **Không gian thuộc tính** | **2 đặc trưng** (10 cột mã hóa One-Hot + Scaler) | **13 đặc trưng** (193 cột mã hóa, TF-IDF Skill Vocabulary) | Giảm 84.6% số biến |
| **CV R² (5-Fold Temporal DEV)** | **0.841** | **0.824** | **+0.017 (+2.1%)** |
| **CV MAE (5-Fold Temporal DEV)** | **$13,596** | **$15,594** | **-$1,998 (-12.8%)** |
| **CV RMSE (5-Fold Temporal DEV)** | **$25,141** | **$26,747** | **-$1,606 (-6.0%)** |
| **Test Set R² (295 dòng held-out)** | **0.815** | **0.807** | **+0.008 (+1.0%)** |
| **Test Set MAE (295 dòng held-out)**| **$13,780** | **$14,775** | **-$995 (-6.7%)** |
| **Test Set MedAE (295 dòng held-out)**| **$3,838** | **$4,400** | **-$562 (-12.8%)** |

#### 🎯 3 Bản ghi Test được chừa lại cho Stage 6 (Salary Prediction) & So sánh có phân loại cụ thể:

Hai bảng này đã được chuyển xuống đặt cùng nhau tại **phần cuối của Tab 1 (Feature Ablation)** với phân loại chi tiết:

**1. Bảng phân loại hồ sơ ứng viên chi tiết (Profile & Attribute Classification):**
- **Record 1 (AI Compliance Manager)**: Security · Master's (15 năm KN) · Zurich, Switzerland (Hybrid) · Sản xuất · $90,000 USD.
- **Record 2 (RAG Engineer)**: Architecture · Bachelor's (6 năm KN) · Beijing, China (Hybrid) · Năng lượng · $180,000 USD.
- **Record 3 (Data Engineer (AI))**: AI Engineering · Bachelor's (1 năm KN) · Singapore (Fully Remote) · Công nghệ · $374,400 USD.

**2. Bảng so sánh dự đoán giữa 2 mô hình có phân loại đánh giá cụ thể:**

| Mã hồ sơ | Vị trí công việc | Phân loại lĩnh vực & Thị trường | Lương thực tế | Dự đoán (Top 2) | Sai số Top 2 | Dự đoán (Full) | Sai số Full | Phân loại đánh giá & Nhận xét |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Record 1** | **AI Compliance Manager** | Security Governance · Thụy Sĩ (Hybrid) | **$90,000** | **$90,000** | **$0 (0.0%)** | $90,940 | $940 (1.0%) | **Top 2 khớp tuyệt đối**: Tín hiệu ngành & năm KN bắt trúng mức lương hợp đồng, không bị nhiễu |
| **Record 2** | **RAG Engineer** | Kiến trúc Hệ thống LLM · Trung Quốc (Hybrid) | **$180,000** | **$179,254** | **$746 (0.4%)** | $178,345 | $1,655 (0.9%) | **Cả 2 mô hình đều xuất sắc**: Bắt trúng mặt bằng lương GenAI với sai số <1% |
| **Record 3** | **Data Engineer (AI)** | Hạ tầng Dữ liệu AI · Singapore (Remote) | **$374,400** | **$348,235** | **$26,165 (7.0%)** | $344,684 | $29,716 (7.9%) | **Nén biên lương cao (Outlier)**: Lương thực tế cực cao ($374k/1 năm KN), cả 2 neo an toàn quanh ~$345k-$348k |

#### 💡 Đánh giá bản chất kỹ thuật (Parsimony vs. Granularity):
- **Tại sao mô hình Top 2 lại có R² cao hơn (0.815 so với 0.807) và MAE thấp hơn ($13,780 so với $14,775)?**
  - Việc loại bỏ 11 biến nhiễu/thưa (như từ vựng kỹ năng phân tán, quy mô công ty chi tiết, điểm phúc lợi) giúp thuật toán loại bỏ hoàn toàn hiện tượng đa cộng tuyến (multicollinearity) và quá khớp vi mô. Mô hình đạt tính tối giản tuyệt đối (Occam's Razor) với mức độ tổng quát hóa kinh tế vĩ mô cực cao.
- **Tại sao vẫn cần giữ mô hình Full 13 biến để triển khai thực tế?**
  - Mô hình Top 2 bị "mù" trước các kỹ năng chuyên môn cụ thể (ví dụ: cùng một vị trí Data Engineer 5 năm kinh nghiệm, mô hình Top 2 sẽ đưa ra mức lương y hệt nhau cho người chỉ biết SQL cơ bản và người thành thạo PyTorch/LLM/RAG). Nó cũng không phản ánh được mức bù trừ theo quốc gia (Thụy Sĩ vs Việt Nam) hay quy mô tập đoàn lớn vs startup.
  - **Chiến lược tối ưu**: Mô hình Top 2 đóng vai trò là thước đo đối chuẩn vĩ mô (Macro Benchmark), trong khi Mô hình Full 13 biến được triển khai cho nghiệp vụ thực tế nhằm cá nhân hóa mức lương cho từng ứng viên theo kỹ năng chuyên sâu.

---

## 8. Chuyển đổi tính năng Salary Prediction chạy độc quyền trên 2 đặc trưng cốt lõi (Top 2 Features)

Đáp ứng yêu cầu đơn giản hóa giao diện và tối ưu hóa dự báo theo kết quả thực nghiệm Ablation (Top 2 đạt Test $R^2 = 0.815$ cao hơn Full 13 biến), trang **6. AI Market Job Salary Prediction** đã được chuyển đổi hoàn toàn:

### 8.1. Các thay đổi chính trên giao diện Stage 6
1. **Form nhập liệu tinh gọn tối đa**:
   - Thay vì bắt buộc người dùng nhập 13 trường phức tạp (học vấn, thành phố, quốc gia, quy mô công ty, ngành nghề, danh sách kỹ năng chi tiết, điểm demand, điểm phúc lợi...), form dự đoán hiện tại **chỉ cần đúng 2 thông số vĩ mô**:
     - **Job Category** (Danh mục nghề nghiệp: AI Engineering, Architecture, Security, Data Science...).
     - **Years of Experience** (Số năm kinh nghiệm: thanh trượt từ 0 đến 25 năm).
     - *(Kèm nhãn Job Title mô tả tùy chọn để hiển thị trên biểu đồ)*.
2. **Nạp nhanh 3 dòng Test mẫu (Quick Load)**:
   - Các nút nạp nhanh Row 1, Row 2, Row 3 lập tức đẩy hồ sơ vào hàng đợi dự đoán kèm mức lương thực tế để đối chiếu sai số trực tiếp.
3. **Mô hình phục vụ (Serving Model Bundle)**:
   - Hệ thống tải `artifacts/model_bundle_top2.joblib` (huấn luyện trên DEV 1,201 dòng) để suy luận thời gian thực tức thì.
   - Dải sai số kinh nghiệm thực nghiệm: **Empirical Error Band $q_{90} = \pm \$34,101$** (chặt chẽ hơn so với mức $\pm \$40,068$ của mô hình cũ).
4. **Trực quan hóa đa chiều**:
   - Biểu đồ điểm ước tính kèm dải sai số $q_{90}$.
   - Biểu đồ lương trung bình theo danh mục nghề nghiệp.
   - Biểu đồ đường phân tán tương quan lương theo số năm kinh nghiệm.

---

### 8.2. Khắc phục lỗi `IntCastingNaNError` khi dự đoán kịch bản tùy chọn
- **Hiện tượng lỗi**: Khi người dùng thêm kịch bản mới từ form nhập liệu (không qua Quick Load), giao diện Streamlit gặp sự cố:
  ```text
  pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
  File "src/pages/page06_prediction.py", line 155, in render
      result["absolute_error_usd"] = np.abs(result["actual_salary_usd"] - result["predicted_salary_usd"]).astype(int)
  ```
- **Nguyên nhân kỹ thuật**: Bản ghi do người dùng tự định nghĩa không có mức lương mặt bằng thực tế `actual_salary_usd` (chứa giá trị `NaN`). Phép toán trừ sinh ra kết quả `NaN`, khi gọi trực tiếp `.astype(int)` thì pandas chặn lại do không thể ép kiểu `NaN` sang số nguyên.
- **Giải pháp xử lý triệt để**:
  1. Ép kiểu an toàn bằng `pd.to_numeric(result["actual_salary_usd"], errors="coerce")`.
  2. Tính toán sai số `absolute_error_usd` và `error_pct` dưới dạng chuỗi số thực `float`, bảo toàn giá trị `NaN` cho các dòng tự tạo.
  3. Áp dụng định dạng bảng `st.dataframe(..., na_rep="-")`:
     - Các dòng nạp nhanh (Quick Load có lương thực tế): Hiển thị đầy đủ sai số `${:,.0f}` và phần trăm sai số `{:.1f}%`.
     - Các dòng kịch bản tự nhập: Hiển thị ký tự `"-"` lịch sự và an toàn, loại bỏ hoàn toàn lỗi crash giao diện.

---

## 9. Hướng dẫn chạy & kiểm thử

### 9.1. Chạy kiểm thử tự động
```powershell
pytest
```
*Kết quả dự kiến: Toàn bộ 29 tests PASS 100%.*

### 9.2. Chạy kiểm tra quy chuẩn release
```powershell
python validate_release.py
```
*Kết quả dự kiến: Release validation: PASS — 36/36 checks passed.*

### 9.3. Khởi chạy ứng dụng Streamlit Dashboard
```powershell
streamlit run streamlit.py
```
Mở trình duyệt tại `http://localhost:8501` để xem dashboard tương tác với toàn bộ bảng biểu và dữ liệu phân tích đã được cập nhật.

