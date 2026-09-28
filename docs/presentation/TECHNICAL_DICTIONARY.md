# TỪ ĐIỂN THUẬT NGỮ KỸ THUẬT MACHINE LEARNING (PAGE 04 & PAGE 05)
> **Dành cho Sinh viên bảo vệ Đồ án / Báo cáo trước Giảng viên và Hội đồng**  
> *Tài liệu giải thích tường tận bản chất toán học, ý nghĩa thực tế, lý do sử dụng trong dự án và cách trả lời giảng viên bằng tiếng Việt.*

---

## MỤC LỤC
1. [Nhóm chỉ số sai số dự báo (Error Metrics)](#1-nhóm-chỉ-số-sai-số-dự-báo-error-metrics)
2. [Nhóm chỉ số đánh giá độ khớp & giải thích (Fit & Goodness Metrics)](#2-nhóm-chỉ-số-đánh-giá-độ-khớp--giải-thích-fit--goodness-metrics)
3. [Phương pháp chia dữ liệu & Kiểm định thời gian (Validation Strategy)](#3-phương-pháp-chia-dữ-liệu--kiểm-định-thời-gian-validation-strategy)
4. [Biến đổi dữ liệu & Chống rò rỉ (Feature Engineering & Leakage Prevention)](#4-biến-đổi-dữ-liệu--chống-rò-rỉ-feature-engineering--leakage-prevention)
5. [Tập mô hình ứng viên (Model Families)](#5-tập-mô-hình-ứng-viên-model-families)
6. [Siêu tham số tinh chỉnh Random Forest (Hyperparameters)](#6-siêu-tham-số-tinh-chỉnh-random-forest-hyperparameters)
7. [Phương pháp đo độ quan trọng đặc trưng (Feature Importance)](#7-phương-pháp-đo-độ-quan-trọng-đặc-trưng-feature-importance)
8. [Độ bất định & Độ tin cậy (Uncertainty & Reliability Diagnostics)](#8-độ-bất-định--độ-tin-cậy-uncertainty--reliability-diagnostics)

---

## 1. Nhóm chỉ số sai số dự báo (Error Metrics)

### 1.1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)
* **Công thức toán học:**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Đơn vị:** USD (\$).
* **Bản chất trực quan:** Trung bình cộng độ lệch giữa mức lương thực tế và mức lương dự đoán, bỏ qua chiều âm/dương.
* **Tại sao dự án dùng làm chỉ số chính (Primary Metric):**
  * MAE phản ánh trực tiếp giá trị tiền tệ dễ hiểu nhất đối với nhà tuyển dụng và ứng viên: *"Trung bình mô hình đoán lệch khoảng bao nhiêu tiền một job?"*
  * MAE tuyến tính, không phạt quá nặng các outlier như MSE hay RMSE.
* **Số liệu trong bài:** 
  * Random Forest đạt CV MAE = **$15,662**, Historical Test MAE = **$14,735** (so với baseline Dummy Median là **$54,011**).
* **Cách giải thích cho giảng viên:**
  > *"Em chọn MAE làm thước đo tối ưu chính vì MAE có cùng đơn vị USD và phản ánh trực tiếp chi phí sai lệch trung bình trên mỗi vị trí tuyển dụng mà không bị méo mó quá đà bởi một vài mức lương cá biệt triệu USD."*

---

### 1.2. MSE (Mean Squared Error - Sai số bình phương trung bình)
* **Công thức toán học:**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Đơn vị:** $\text{USD}^2$ (Đô la bình phương - không có ý nghĩa vật lý trực quan).
* **Bản chất trực quan:** Bình phương các độ lệch rồi lấy trung bình.
* **Đặc tính:** Vì có phép bình phương, sai số lớn (ví dụ đoán lệch $100,000) sẽ bị phạt gấp hàng trăm lần sai số nhỏ (đoán lệch $1,000).
* **Cách giải thích cho giảng viên:**
  > *"MSE là hàm mất mát (loss function) chuẩn trong huấn luyện mô hình toán học (như OLS hay cây hồi quy). Tuy nhiên trong báo cáo đánh giá, MSE mang đơn vị $\text{USD}^2$ nên khó hình dung, do đó chúng em khai căn bậc hai để đưa về RMSE."*

---

### 1.3. RMSE (Root Mean Squared Error - Căn bậc hai sai số bình phương trung bình)
* **Công thức toán học:**
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Đơn vị:** USD (\$).
* **Bản chất trực quan:** Cùng đơn vị USD như MAE, nhưng **rất nhạy cảm với các sai số lớn (outliers)**. Nếu có một vài job đoán lệch khủng khiếp, RMSE sẽ vọt lên rất cao so với MAE.
* **Số liệu trong bài:** 
  * Historical Test RMSE = **$29,111** (trong khi Test MAE chỉ là **$14,735**).
* **Tỷ lệ vàng cần nhớ:** $\text{RMSE} > \text{MAE}$ luôn đúng. Khi $\text{RMSE} \approx 2 \times \text{MAE}$, điều đó báo hiệu phân phối sai số có đuôi dày (heavy tail) do các mức lương executive/hiếm.
* **Cách giải thích cho giảng viên:**
  > *"RMSE giúp chúng em phát hiện các lỗi dự báo nghiêm trọng. Việc RMSE ($29,111) cao gần gấp đôi MAE ($14,735) phản ánh đúng đặc thù thị trường: đa số job đoán rất sát, nhưng có một lượng nhỏ các vị trí Director/VP có mức lương quá cao tạo ra sai số bình phương lớn."*

---

### 1.4. MedAE (Median Absolute Error - Sai số tuyệt đối trung vị)
* **Công thức toán học:**
  $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_n - \hat{y}_n|)$$
* **Đơn vị:** USD (\$).
* **Bản chất trực quan:** Lấy trung vị của chuỗi sai số tuyệt đối. Nghĩa là: **50% số công việc trong tập kiểm thử có sai số nhỏ hơn con số này**.
* **Đặc tính:** Hoàn toàn miễn nhiễm với các ngoại lai (outlier-robust).
* **Số liệu trong bài:** 
  * Test MedAE = **$4,347** (Cực kỳ thấp!).
* **Cách giải thích cho giảng viên:**
  > *"Thưa thầy/cô, nếu MAE là $14,735 thì MedAE chỉ có $4,347. Điều này chứng minh cho một nửa số hồ sơ công việc trên thị trường, sai số dự đoán chỉ lệch dưới $4,347/năm — tương đương mức sai số chỉ ~2-3% trên mức lương 150k-200k. Mô hình dự đoán cực kỳ chuẩn xác cho phân khúc thị trường đại chúng."*

---

### 1.5. Residual (Phần dư)
* **Công thức:**
  $$e_i = y_i - \hat{y}_i \quad (\text{Thực tế} - \text{Dự đoán})$$
* **Ý nghĩa dấu của phần dư:**
  * $e_i > 0$: Lương thực tế cao hơn dự đoán $\rightarrow$ Mô hình đang đoán thấp (Under-predicting).
  * $e_i < 0$: Lương thực tế thấp hơn dự đoán $\rightarrow$ Mô hình đang đoán cao (Over-predicting).
* **Kiểm định phần dư chuẩn:** Phân phối phần dư phải có hình chuông cân xứng, đỉnh nằm sát số 0.
* **Số liệu trong bài:** Median residual trên tập test là **-$940**, chứng tỏ mô hình không hề bị lệch hệ thống (no systemic bias).

---

## 2. Nhóm chỉ số đánh giá độ khớp & giải thích (Fit & Goodness Metrics)

### 2.1. $R^2$ (Coefficient of Determination - Hệ số xác định)
* **Công thức toán học:**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
* **Đơn vị:** Không thứ nguyên (Unitless), nằm trong khoảng $(-\infty, 1.0]$.
* **Ý nghĩa bản chất:** Tỷ lệ phần trăm phương sai (sự biến thiên) của mức lương được mô hình giải thích thông qua các đặc trưng đầu vào.
  * $R^2 = 1.0$: Dự đoán chính xác tuyệt đối 100%.
  * $R^2 = 0.0$: Mô hình dự đoán bằng đúng giá trị trung bình $\bar{y}$ (không thông minh hơn việc đoán mò trung bình).
  * $R^2 < 0$: Mô hình dự đoán tệ hơn cả việc đoán trung bình.
* **Số liệu trong bài:** 
  * Random Forest đạt CV $R^2 = \mathbf{0.822}$, Historical Test $R^2 = \mathbf{0.813}$.
* **Cách giải thích cho giảng viên:**
  > *"Mô hình Random Forest giải thích được hơn 81.3% sự biến thiên của mức lương thị trường AI. Độ chênh lệch giữa CV (0.824) và Test (0.813) chỉ là 0.011 (1.1%), chứng minh mô hình tổng quát hóa rất vững, không bị overfit."*

---

### 2.2. Tại sao Linear Regression có $R^2$ bị Âm ($R^2 = -0.520$)?
* Giảng viên rất hay hỏi câu này để kiểm tra hiểu biết toán học!
* **Bản chất toán học:** $R^2$ chỉ chắc chắn $\ge 0$ khi tính trên tập train với mô hình hồi quy tuyến tính cổ điển có hằng số chặn (intercept). Khi đánh giá **out-of-sample (trên tập validation/test)**, nếu tổng bình phương sai số phần dư $\text{SS}_{\text{res}}$ lớn hơn tổng phương sai dữ liệu $\text{SS}_{\text{tot}}$, thì $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} < 0$.
* **Nguyên nhân kỹ thuật trong bài:**
  1. Mối quan hệ giữa chức danh, số năm kinh nghiệm và mức lương là phi tuyến tính mạnh (non-linear).
  2. One-hot encoding tạo ra nhiều biến giả (high-dimensional dummy variables), dẫn đến hiện tượng đa cộng tuyến (multicollinearity) nghiêm trọng làm ma trận $(X^TX)$ gần như suy biến, hệ số trọng số OLS bị thổi phồng cực đoan.
* **Cách trả lời giảng viên:**
  > *"Thưa thầy/cô, mô hình Linear Regression không có cơ chế chặn trọng số (regularization). Khi gặp các tổ hợp biến phân loại mới ở tập validation, mô hình tạo ra những dự đoán sai lệch cực lớn vượt qua cả phương sai mẫu, khiến $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$, dẫn đến $R^2 = -0.520$. Trong khi đó, Ridge Regression thêm chuẩn L2 ($\alpha$) đã kéo $R^2$ lên +0.637, khẳng định chẩn đoán đa cộng tuyến là hoàn toàn chính xác."*

---

## 3. Phương pháp chia dữ liệu & Kiểm định thời gian (Validation Strategy)

### 3.1. Temporal Sliding Window Cross-Validation (K-Fold trượt theo chuỗi thời gian)
* **Khác biệt cốt tử với Random K-Fold:**
  * Random K-Fold: Xáo trộn ngẫu nhiên dữ liệu rồi chia 5 phần.
  * Temporal Sliding Window: Dữ liệu được sắp xếp theo thời gian (`work_year`, tháng tuyển dụng). Khối thời gian quá khứ làm Train, khối thời gian kế tiếp làm Validation.
* **Số liệu trong bài:**
  * Toàn bộ tập Development (1,201 dòng) được chia thành 5 fold trượt.
  * Fold 1: Train khối sớm nhất $\rightarrow$ Validate khối tiếp theo.
  * ...
  * Fold 5: Train khối gần cuối $\rightarrow$ Validate khối sát tập test.
* **Tại sao bắt buộc phải dùng cho bài toán lương:**
  > Thị trường lao động có tính biến động theo thời gian (lạm phát, bong bóng công nghệ, sự bùng nổ của GenAI cuối 2023–2025). Nếu dùng Random K-Fold, bạn sẽ lấy dữ liệu của tháng 12/2025 để "dự đoán" lương của tháng 01/2025 $\rightarrow$ **Rò rỉ tương lai (Look-ahead Leakage)** làm cho điểm số trên giấy tờ đẹp ảo nhưng khi triển khai thực tế sẽ gãy!

---

### 3.2. Out-of-Sample Locked Historical Test Set (Tập kiểm thử khóa độc lập)
* **Quy mô:** **298 dòng** (giai đoạn `2026-03`).
* **Quy tắc bất khả xâm phạm:**
  * Tập này **không được tham gia** vào việc chuẩn hóa, không tham gia vào huấn luyện, không tham gia chọn mô hình hay tìm siêu tham số.
  * Chỉ được mở khóa duy nhất 1 lần ở Trang 05 để làm "thước đo sự thật" (ground truth benchmark).

---

## 4. Biến đổi dữ liệu & Chống rò rỉ (Feature Engineering & Leakage Prevention)

### 4.1. $\text{log1p}$ và $\text{expm1}$ Transformation
* **Toán học:**
  $$y_{\text{train}} = \log(1 + x) \quad (\text{log1p})$$
  $$\hat{y}_{\text{dollar}} = e^{\hat{y}_{\text{model}}} - 1 \quad (\text{expm1})$$
* **Lý do áp dụng:** Phân phối lương luôn có dạng **lệch phải (right-skewed)**: đại đa số kỹ sư hưởng lương 80k-250k USD, nhưng có một dải đuôi dài kéo đến 500k-800k USD.
* **Lợi ích:**
  1. Co ngắn dải phân phối đuôi dài, đưa target về phân phối chuẩn (Gaussian-like).
  2. Ổn định phương sai của sai số (Homoscedasticity).
  3. Đảm bảo mô hình khi dự báo ngược lại không bao giờ cho ra mức lương âm.

---

### 4.2. Data Leakage Prevention (Phòng chống rò rỉ dữ liệu)
* **Nguyên tắc:** Bất kỳ phép biến đổi nào tính toán thống kê (Mean, Std, Min, Max, Quantile) đều **chỉ được phép `.fit()` trên fold Train**, sau đó chỉ dùng `.transform()` trên fold Validation và Test.
* **Minh chứng trong code:** Pipeline scikit-learn tích hợp trực tiếp preprocessor vào bên trong estimator cho từng fold.

---

## 5. Tập mô hình ứng viên (Model Families)

Dự án thiết kế 5 mô hình đại diện cho toàn bộ phổ thuật ngữ Machine Learning:

| Mô hình | Vai trò & Bản chất thuật toán | Kết quả CV MAE |
| :--- | :--- | :---: |
| **Dummy (Median)** | **Baseline sàn:** Luôn dự đoán bằng mức lương trung vị của tập huấn luyện. Không dùng bất kỳ feature nào. | **$54,011** |
| **Linear Regression** | **Hồi quy tham số cơ bản:** Tìm siêu phẳng tối ưu tuyến tính bằng OLS (Ordinary Least Squares). | **$59,006** *(thất bại)* |
| **Ridge Regression** | **Tuyến tính có điều chuẩn L2:** Thêm phạt $\alpha \sum w_j^2$ để thu nhỏ các trọng số bị thổi phồng do đa cộng tuyến. | **$28,057** *(giảm 50% lỗi)* |
| **Random Forest** | **Bagging Ensembles:** Rừng cây ngẫu nhiên (nhiều cây độc lập lấy mẫu bootstrap và ngẫu nhiên đặc trưng, lấy trung bình dự đoán). | **$15,662** *(chiến thắng)* |
| **Gradient Boosting** | **Boosting Ensembles:** Xây dựng chuỗi cây tuần tự, mỗi cây sau tập trung học để sửa phần dư (residual) của cây trước. | **$16,421** *(bám rất sát)* |

---

## 6. Siêu tham số tinh chỉnh Random Forest (Hyperparameters)

Sau khi Random Forest được chọn ở Trang 04, dự án tiến hành khảo sát độ nhạy qua 4 tham số chính ở Trang 05:

### 6.1. `n_estimators = 200` (Số lượng cây quyết định)
* **Ý nghĩa:** Số lượng cây đơn lẻ được trồng trong rừng.
* **Khảo sát:** Thử nghiệm dải [100, 150, 200, 250, 300, 400].
* **Kết luận chọn 200:** Từ 100 lên 200 cây, MAE giảm đáng kể và phương sai dự báo ổn định. Trên 200 cây (300, 400), sai số không giảm thêm mà thời gian huấn luyện và độ trễ inference tăng gấp đôi. Do đó `200` là điểm tối ưu chi phí tính toán (compute-efficiency).

### 6.2. `max_depth = 20` (Độ sâu tối đa của mỗi cây)
* **Ý nghĩa:** Số tầng phân nhánh tối đa từ gốc đến lá.
* **Khảo sát:** Thử nghiệm [10, 15, 20, 25, 30, None].
* **Kết luận chọn 20:** Nếu `max_depth` quá nhỏ (<10), cây bị underfit không học được các tương tác sâu (ví dụ: Senior + AI Engineering + US). Nếu để `None` (không giới hạn), cây sẽ phát triển sâu đến 40-50 tầng, học vẹt nhiễu của tập train. Độ sâu `20` chặn đứng nguy cơ bùng nổ kích thước cây mà vẫn bao quát toàn bộ tương tác phức tạp.

### 6.3. `min_samples_leaf = 1` (Số mẫu tối thiểu tại mỗi nút lá)
* **Ý nghĩa:** Một lá (leaf node) được phép chứa tối thiểu bao nhiêu hồ sơ công việc để đưa ra dự báo mức lương.
* **Khảo sát:** Thử nghiệm [1, 2, 4, 8].
* **Kết luận chọn 1:** Với các thị trường chuyên biệt, một số chức danh hiếm (như AI Ethicist, Prompt Engineer cấp cao) chỉ có 1-2 mẫu. Việc cho phép leaf=1 giúp mô hình học được độ phân giải chi tiết, trong khi tính chất lấy trung bình của 200 cây trong Random Forest đã tự động triệt tiêu nguy cơ quá khớp của từng cây riêng lẻ.

### 6.4. `max_features = 0.7` (Tỷ lệ đặc trưng xem xét tại mỗi phân nhánh)
* **ĐÂY LÀ ĐIỂM SÁNG GIẢNG VIÊN ĐÁNH GIÁ RẤT CAO!**
* **Ý nghĩa:** Tại mỗi lần tách nhánh (split), thuật toán chỉ bốc ngẫu nhiên **70% tổng số đặc trưng** để chọn ra đặc trưng tốt nhất.
* **Tại sao không để 1.0 (toàn bộ 100%)?**
  * Nếu để `1.0`, đặc trưng cực mạnh là **`job_category`** sẽ luôn được chọn ở nút gốc của hầu hết mọi cây trong 200 cây. Khi đó, 200 cây trở nên quá giống nhau (correlated trees).
  * Việc ép chỉ lấy ngẫu nhiên 70% bắt buộc các cây phải khám phá các đặc trưng khác (`years_of_experience`, `company_size`, `remote_ratio`), từ đó tạo ra sự **đa dạng sinh học (diversity)** cho rừng cây. Khi lấy trung bình các cây đa dạng, phương sai của toàn bộ ensemble được triệt tiêu tối đa!

---

## 7. Phương pháp đo độ quan trọng đặc trưng (Feature Importance)

Dự án đối chiếu 2 trường phái giải thích mô hình:

### 7.1. Permutation Feature Importance (Đo trên dữ liệu gốc - Raw Data)
* **Cách thực hiện:** Lấy tập dữ liệu kiểm tra, xáo trộn ngẫu nhiên (shuffle) duy nhất một cột (ví dụ cột `job_category`) để phá hủy mối liên hệ của nó với mức lương, sau đó đo xem MAE tăng lên bao nhiêu.
* **Kết quả trong bài:**
  * Xáo trộn **`job_category`** $\rightarrow$ MAE tăng vọt **+$49,598/job**!
  * Xáo trộn **`years_of_experience`** $\rightarrow$ MAE tăng vọt **+$12,379/job**!
* **Ưu điểm vượt trội:** Hoàn toàn khách quan, không bị thiên lệch bởi số lượng giá trị phân loại (cardinality), phản ánh trực tiếp thiệt hại tiền tệ nếu thiếu biến đó.

### 7.2. Impurity-based (MDI) Importance (Đo trên các cột mã hóa - Encoded Columns)
* **Cách thực hiện:** Tính tổng mức giảm phương sai (variance reduction) mà một đặc trưng đóng góp qua tất cả các nút phân nhánh của 200 cây.
* **Kết quả:** `nominal__job_category_AI Engineering` chiếm **55.7%**, `numeric__years_of_experience` chiếm **27.9%**.
* **Ưu điểm:** Cho biết cụ thể phân lớp nào (Category cụ thể nào) tạo ra giá trị khác biệt lớn nhất.

---

## 8. Độ bất định & Độ tin cậy (Uncertainty & Reliability Diagnostics)

### 8.1. Khoảng bất định thực nghiệm $q90$ ($\pm \$40,068$)
* **Bản chất:** Thay vì giả định sai số tuân theo phân phối chuẩn Gaussian (vốn không đúng với dữ liệu tiền tệ), dự án tính **phân vị 90% (90th percentile)** của chuỗi sai số tuyệt đối trên tập test.
* **Ý nghĩa:** $P(|y - \hat{y}| \le \$40,068) = 89.9\% \approx 90\%$.
* **Cách diễn đạt thực tế:**
  > *"Khi mô hình đưa ra con số dự đoán $180,000, chúng ta có thể công bố với ứng viên khoảng tin cậy thực nghiệm là [$140,000 - $220,000] với độ bao phủ thực tế đạt 90%."*

### 8.2. ECDF (Empirical Cumulative Distribution Function - Phân phối tích lũy thực nghiệm)
* **Trục hoành ($X$):** Độ lớn sai số tuyệt đối (\$).
* **Trục tung ($Y$):** Tỷ lệ phần trăm mẫu tích lũy (0% đến 100%).
* **Cách đọc đồ thị:** 
  * Chiếu từ $Y = 50\%$ sang $X$: Gặp giá trị MedAE $\approx \$4,347$.
  * Chiếu từ $Y = 90\%$ sang $X$: Gặp giá trị $q90 \approx \$40,068$.
  * Đường cong dựng đứng rất nhanh ở vùng đầu chứng minh phần lớn dự đoán tập trung sai số rất bé.

### 8.3. Tỷ số $\text{RMSE} / \text{MedAE} = 6.7\times$
* Một chỉ số chẩn đoán chuyên sâu: cho thấy các ngoại lai cực lớn kéo điểm RMSE lên gấp 6.7 lần sai số điển hình của một công việc bình thường.
