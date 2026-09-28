# KỊCH BẢN THUYẾT TRÌNH TRANG 04: SO SÁNH MÔ HÌNH & KIỂM ĐỊNH THỜI GIAN
> **Mục tiêu:** Thuyết phục Giảng viên và Hội đồng rằng việc lựa chọn thuật toán **Random Forest** là kết quả của một quy trình so sánh khoa học, minh bạch, kiểm định theo chuỗi thời gian chặt chẽ, hoàn toàn không có rò rỉ dữ liệu (No Data Leakage).

---

## 1. THÔNG ĐIỆP CỐT LÕI (NÓI TRONG 20 GIÂY MỞ ĐẦU)

> *"Kính thưa Thầy/Cô và Hội đồng, tại Trang 04 này, nhóm chúng em giải quyết bài toán: **Lựa chọn họ thuật toán nào là phù hợp và tối ưu nhất để dự báo mức lương thị trường AI?**
> Thay vì dùng phương pháp chia ngẫu nhiên truyền thống vốn dễ gây rò rỉ tương lai, chúng em thiết kế một quy trình **Temporal Sliding-Window Cross-Validation (K-Fold trượt theo thời gian)** gồm 5 vòng lặp độc lập để thẩm định 5 họ thuật toán đối đầu trực tiếp trên cùng một chuẩn dữ liệu."*

---

## 2. BƯỚC 1: ĐỌC VÀ CHỈ VÀO 5 THẺ KPI ĐẦU TRANG (TOP METRICS)

Khi giao diện tải xong, bạn dùng chuột khoanh tròn vào 5 thẻ Metric trên cùng:

| Thẻ Metric trên giao diện | Giá trị hiển thị | Ý nghĩa & Lời thoại thuyết minh |
| :--- | :---: | :--- |
| **Candidates** | **5** | *"Chúng em so sánh 5 họ mô hình từ đơn giản đến phức tạp: Dummy Median, Linear Regression, Ridge Regression, Random Forest và Gradient Boosting."* |
| **Temporal folds** | **5** | *"Toàn bộ 1,201 dòng của tập Development được chia làm 5 fold trượt theo thời gian để mô phỏng chính xác việc mô hình học quá khứ để dự báo tương lai."* |
| **Lowest CV MAE** | **Random Forest** | *"Mô hình đạt sai số kiểm định trung bình thấp nhất là Random Forest."* |
| **Best Mean CV MAE** | **$15,662** | *"Sai số tuyệt đối trung bình trên các tập kiểm định chỉ là $15,662/công việc."* |
| **Dummy reference** | **$54,011** | *"Nếu chỉ đoán mò bằng mức lương trung vị, sai số lên đến $54,011. Random Forest đã giúp **giảm 71.0% sai số** so với mốc cơ sở này."* |

---

## 3. BƯỚC 2: GIẢI THÍCH KHUNG PHƯƠNG PHÁP LUẬN (WORKFLOW EXPANDERS)

Bấm mở expander **"1. Target & Feature Engineering — data splitting"** và trình bày 3 luận điểm khoa học:

### Luận điểm 1: Xử lý Target phân phối lệch phải
* **Hành động:** Chỉ vào biểu đồ minh họa cột màu cam (salary gốc) và đường màu xanh (log1p).
* **Lời thoại:**
  > *"Mức lương thị trường công nghệ có phân phối lệch phải (Right-skewed) với dải đuôi kéo dài tới các vị trí Director hàng trăm nghìn USD. Nếu để nguyên, thuật toán hồi quy sẽ bị kéo lệch bởi các outlier này. Chúng em áp dụng phép biến đổi $y = \log(1 + \text{salary})$ để chuẩn hóa phân phối về dạng đối xứng. Sau khi mô hình dự báo xong, chúng em dùng hàm ngược $e^{\hat{y}} - 1$ để đưa về đúng đơn vị tiền USD nguyên bản."*

### Luận điểm 2: Chống rò rỉ thông tin (Data Leakage Prevention)
* **Lời thoại:**
  > *"Để đảm bảo tính trung thực tuyệt đối, tất cả các bộ tiền xử lý (StandardScaler, One-Hot Encoding) đều chỉ được `.fit()` trên dữ liệu của khối Train. Khối Validation và tập Test hoàn toàn không tham gia vào quá trình tính giá trị trung bình hay phương sai."*

### Luận điểm 3: Thiết kế 5-Fold trượt thời gian (Sliding Window Diagram)
* **Hành động:** Chỉ vào bảng ma trận màu sắc (Train xanh lá, Validation cam, Test đỏ).
* **Lời thoại:**
  > *"Chúng em tách hẳn **298 dòng (tháng 03/2026)** làm tập Test độc lập. 1,201 dòng trước đó được chia làm 5 khối thời gian trượt. Ở mỗi Fold, mô hình học từ khối quá khứ và kiểm định trên khối kế tiếp. Điều này phản ánh trung thực thực tế kinh doanh: không một doanh nghiệp nào có thể dùng dữ liệu tương lai để dự đoán quá khứ."*

---

## 4. BƯỚC 3: PHÂN TÍCH CHI TIẾT 4 TAB SO SÁNH

---

### TAB 1: OVERALL COMPARISON (SO SÁNH TỔNG THỂ)
* **Đây là Tab quan trọng nhất! Dành 50% thời lượng cho Tab này.**

#### 1. Phân tích Biểu đồ kép (Train vs Validation MAE & R²)
* **Biểu đồ bên trái (Train vs Validation MAE):**
  * Cột xanh ngọc là **Validation MAE** (càng thấp càng tốt).
  * Điểm/đường cam là **Train MAE**.
  * **Chỉ ra:**
    * Dummy Median: Cả Train và Validation đều cao ngất ngưởng (~$52k - $54k).
    * Linear Regression: Train MAE rất đẹp ($11.5k) nhưng Validation MAE vọt lên $59k $\rightarrow$ **Quá khớp cực độ (Catastrophic Overfitting)**.
    * Ridge Regression: Điều chỉnh regularization kéo Validation MAE xuống còn **$28,057**.
    * Gradient Boosting: Đạt Validation MAE **$16,421**.
    * Random Forest: Đạt Validation MAE xuất sắc nhất là **$15,662**.
* **Biểu đồ bên phải (Validation R²):**
  * Random Forest đạt $R^2 = \mathbf{0.822}$, Gradient Boosting bám sát với $R^2 = \mathbf{0.821}$.
  * Linear Regression bị âm ($R^2 = -0.520$) do đa cộng tuyến của các biến phân loại.

#### 2. Đọc Bảng quyết định ứng viên (Candidate Decision Table)
* Dừng lại ở bảng số liệu tổng kết:
  * **Random Forest:** Xếp hạng 1, MAE $15,662, thấp hơn Dummy **71.0%**, khoảng cách Train-Val MAE là $8,378.
  * **Gradient Boosting:** Xếp hạng 2, MAE $16,421, thua RF **$759/job**.
  * **Thời gian huấn luyện (Fit time):** Cả RF và GB đều huấn luyện siêu nhanh (~0.21 giây), đảm bảo khả năng mở rộng (scalability).

---

### TAB 2: RANDOM FOREST DEEP-DIVE (ĐỘ ỔN ĐỊNH THEO THỜI GIAN)
* Chuyển sang Tab 2 để chứng minh tính phản biện khoa học:

#### 1. Biểu đồ Fold Combo Chart
* **Hành động:** Chỉ vào 5 cột Fold của Random Forest.
* **Lời thoại:**
  > *"Chúng em không chỉ nhìn vào số trung bình, mà kiểm tra độ ổn định qua từng Fold. Folds 1, 2, 3 và 5 có sai số rất tốt, chỉ quanh mức $12,000 - $16,000. Riêng **Fold 4 (giai đoạn 01/2026 - 02/2026)** sai số tăng lên $20,140 do giai đoạn đầu năm thị trường AI có nhiều biến động lương thưởng mới. Độ lệch chuẩn giữa các fold là **$2,935**, cho thấy tính vững chãi của mô hình."*

#### 2. Biểu đồ Trôi dạt độ quan trọng (Feature Reliance Drift)
* **Hành động:** Chỉ vào biểu đồ phân tán (Mean Importance vs SD).
* **Lời thoại:**
  > *"Hai đặc trưng nằm ở góc phải (độ quan trọng cao và độ biến động thấp qua các fold) là `job_category` và `years_of_experience`. Điều này khẳng định hai biến này là trụ cột thị trường bền vững qua thời gian, không bị nhiễu cục bộ."*

---

### TAB 3: GRADIENT BOOSTING VS. RANDOM FOREST (ĐỐI ĐẦU TRỰC DIỆN)
* Thầy/cô thường hỏi: *"Tại sao không chọn Gradient Boosting (XGBoost/GBM) vốn thường thắng trên Kaggle?"*
* **Lời thoại trả lời ngay tại Tab 3:**
  > *"Thưa Thầy/Cô, khi so sánh đối đầu giữa Random Forest và Gradient Boosting:
  > 1. Về $R^2$: Cả hai tương đương tuyệt đối (0.822 so với 0.821).
  > 2. Về sai số tiền tệ: Random Forest cho sai số MAE thấp hơn **$759/công việc** ($15,662 vs $16,421).
  > 3. Về sai số trung vị (MedAE): Random Forest đạt **$7,452**, tốt hơn hẳn mức **$9,490** của Gradient Boosting.
  > 4. Đặc tính thuật toán: Random Forest (Bagging) ít nhạy cảm với siêu tham số hơn, khó bị overfit với các category nhỏ lẻ hơn Boosting tuần tự. Do đó, nhóm quyết định chọn Random Forest làm kiến trúc cốt lõi để tinh chỉnh chuyên sâu ở Trang 05."*

---

### TAB 4: BASELINE CONTROLS (KIỂM CHỨNG CƠ SỞ)
* Lướt nhanh qua Tab 4:
  * So sánh đường ngang màu đỏ đứt nét (Dummy Median $54k).
  * Khẳng định việc chuyển từ mô hình tuyến tính sang cây phi tuyến tính đã giúp cải thiện bước ngoặt về chất lượng dự báo lương.

---

## 5. BỘ CÂU HỎI PHẢN BIỆN CỦA GIẢNG VIÊN & CÁCH TRẢ LỜI XUẤT SẮC

### Câu 1: "Tại sao nhóm dùng 5-Fold Sliding Window mà không dùng Expanding Window (Cửa sổ mở rộng)?"
* **Trả lời:**
  > *"Dạ thưa Thầy/Cô, với Sliding Window (Cửa sổ trượt cố định), kích thước tập Train ở mỗi Fold được giữ đồng đều (~960 dòng), giúp việc so sánh công bằng về mặt dung lượng mẫu giữa các Fold. Expanding Window bổ sung dần dữ liệu cũ từ quá khứ xa xôi, nhưng trong lĩnh vực AI biến đổi theo từng quý, dữ liệu quá cũ có thể gây nhiễu xu hướng tiền lương hiện tại. Do đó Sliding Window phản ánh độ nhạy thị trường gần nhất."*

### Câu 2: "Tại sao Linear Regression có $R^2$ bị âm (-0.520)?"
* **Trả lời:**
  > *"Dạ thưa Thầy/Cô, $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$. Khi đánh giá out-of-sample trên tập validation, Linear Regression không có cơ chế chặn trọng số. Khi gặp hiện tượng đa cộng tuyến giữa hàng chục biến One-Hot, ma trận nghịch đảo bị bất ổn định khiến các hệ số hồi quy bị thổi phồng, tạo ra những dự đoán sai lệch vượt quá cả phương sai của dữ liệu mẫu, dẫn đến $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$ và $R^2$ bị âm."*

### Câu 3: "Thời gian huấn luyện chênh lệch giữa các mô hình thế nào?"
* **Trả lời:**
  > *"Thời gian fit của cả 5 mô hình đều dưới 0.25 giây (Linear ~0.01s, RF ~0.21s, GB ~0.21s). Vì tập dữ liệu khoảng 1,200 dòng, Random Forest hoàn toàn đáp ứng yêu cầu tính toán tức thời (real-time prediction) trên hệ thống web Streamlit."*
