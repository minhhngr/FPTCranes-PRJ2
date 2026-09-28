# GIÁO TRÌNH LÝ THUYẾT MACHINE LEARNING CHUẨN MỰC
## VẤN ĐÁP & PHẢN BIỆN LÝ THUYẾT (REGRESSION & TREE-BASED ENSEMBLES)
> **Đặc điểm:** 100% Kiến thức Khoa học Máy tính & Thống kê học thuần túy (Academic Theory).  
> **Không ràng buộc vào bất kỳ dự án cụ thể nào.** Dành để trả lời thi vấn đáp, bảo vệ luận văn hoặc phỏng vấn kỹ thuật Machine Learning.

---

## MỤC LỤC LÝ THUYẾT
1. [Nhóm các Thước đo Sai số (Error Metrics: MAE, MSE, RMSE, MedAE, Residual)](#1-nhóm-các-thước-đo-sai-số-error-metrics)
2. [Nhóm Đánh giá Độ khớp (Goodness of Fit: R², Âm R²)](#2-nhóm-đánh-giá-độ-khớp-goodness-of-fit)
3. [Nhóm Phân phối Dữ liệu & Xử lý Target (Skewness, log1p, expm1, Homoscedasticity)](#3-nhóm-phân-phối-dữ-liệu--xử-lý-target)
4. [Nhóm Chống Rò rỉ Dữ liệu & Quy trình Đánh giá (Data Leakage, Pipeline, 3 Tập Train-Val-Test)](#4-nhóm-chống-rò-rỉ-dữ-liệu--quy-trình-đánh-giá)
5. [Nhóm Chiến lược Chia dữ liệu (K-Fold CV, Time-Series / Sliding Window CV)](#5-nhóm-chiến-lược-chia-dữ-liệu-validation-strategies)
6. [Nhóm Thuật toán Hồi quy (Dummy Baseline, Linear OLS, Ridge L2)](#6-nhóm-thuật-toán-hồi-quy-regression-algorithms)
7. [Nhóm Học kết hợp Cây quyết định (Ensemble: Bagging vs Boosting, Random Forest vs Gradient Boosting)](#7-nhóm-học-kết-hợp-cây-quyết-định-ensemble-learning)
8. [Nhóm Siêu tham số Mô hình Cây (n_estimators, max_depth, min_samples_leaf, max_features)](#8-nhóm-siêu-tham-số-mô-hình-cây-tree-hyperparameters)
9. [Nhóm Phương pháp Đo Tầm quan trọng Đặc trưng (Permutation Importance vs Impurity MDI)](#9-nhóm-phương-pháp-đo-tầm-quan-trọng-đặc-trưng-feature-importance)
10. [Nhóm Đo lường Độ bất định (Quantile q90, ECDF, Tỷ số RMSE/MedAE)](#10-nhóm-đo-lường-độ-bất-định-uncertainty-metrics)

---

# 1. NHÓM CÁC THƯỚC ĐO SAI SỐ (ERROR METRICS)

---

### 1.1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)

* **Công thức toán học:**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  *(Trong đó: $y_i$ là nhãn thực tế, $\hat{y}_i$ là giá trị mô hình dự báo, $n$ là tổng số mẫu dữ liệu)*
* **Đơn vị tính:** Mang cùng đơn vị đo với biến mục tiêu $y$.
* **Mục đích dùng để làm gì?**
  * Đo lường độ lớn sai số trung bình tuyến tính của mô hình.
  * Trả lời trực quan câu hỏi: *"Trung bình một mẫu thử, mô hình dự đoán lệch bao nhiêu đơn vị thực tế?"*
* **Khi nào thì dùng?**
  * Khi muốn một chỉ số phản ánh mức sai số điển hình, dễ hiểu cho con người.
  * Khi hàm chi phí thiệt hại của bài toán tăng tuyến tính theo sai số (lệch 2 đơn vị chịu thiệt hại gấp đôi lệch 1 đơn vị).
  * Khi dữ liệu chứa các giá trị ngoại lai (outliers) mà bạn **không muốn** chỉ số đánh giá bị các ngoại lai này thổi phồng quá mức.
* **Khi nào KHÔNG nên dùng?**
  * Không dùng làm hàm mất mát (loss function) trong tối ưu hóa bằng Gradient Descent cổ điển, vì hàm trị tuyệt đối $|x|$ không có đạo hàm tại điểm 0.
  * Không dùng khi bài toán thực tế coi sai số lớn là thảm họa (ví dụ: an toàn y tế, hàng không, kết cấu công trình) cần phải trừng phạt nghiêm khắc sai số lớn.
* **Câu trả lời lý thuyết ngắn gọn:**
  > *"MAE là trung bình cộng của các sai số tuyệt đối. Mục đích là cung cấp thước đo sai lệch trực quan cùng đơn vị với biến mục tiêu. Ta dùng MAE khi muốn đánh giá sai số trung bình mà không bị chi phối quá nặng bởi các điểm ngoại lai."*

---

### 1.2. MSE (Mean Squared Error - Sai số bình phương trung bình)

* **Công thức toán học:**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Đơn vị tính:** Bình phương đơn vị của $y$ (ví dụ: $\text{m}^2$, $\text{kg}^2$, $\$^2$).
* **Mục đích dùng để làm gì?**
  * Dùng làm **Hàm mất mát (Loss Function)** cơ bản để máy tính tối ưu hóa toán học. Do hàm bình phương là hàm lồi (convex) và khả vi (differentiable - có đạo hàm trơn liên tục tại mọi điểm), thuật toán dễ dàng tính đạo hàm $\frac{\partial \text{MSE}}{\partial w}$ để cập nhật trọng số.
  * Phạt rất nặng các sai số lớn: Sai số bằng 2 phạt 4, sai số bằng 10 phạt 100.
* **Khi nào thì dùng?**
  * Dùng trong quá trình huấn luyện toán học nội tại của mô hình (OLS, Neural Networks, Gradient Boosting).
  * Dùng khi bài toán có bản chất: sai số càng lớn thì mức độ nguy hiểm càng tăng theo hàm số mũ.
* **Khi nào KHÔNG nên dùng?**
  * **Không bao giờ dùng để báo cáo đánh giá kết quả thực tế cho con người**, vì đơn vị bị bình phương không mang ý nghĩa vật lý trực quan.
  * Không dùng khi tập dữ liệu bị nhiễm nhiều nhiễu ngoại lai mà bạn không muốn mô hình hy sinh sự chính xác của số đông chỉ để cố khớp các điểm ngoại lai đó.
* **Câu trả lời lý thuyết ngắn gọn:**
  > *"MSE là trung bình cộng của các sai số bình phương. Mục đích chính là làm hàm mất mát trong huấn luyện mô hình vì có đạo hàm trơn liên tục tại mọi điểm. Nhược điểm là mang đơn vị bình phương nên không dùng để báo cáo thực tế."*

---

### 1.3. RMSE (Root Mean Squared Error - Căn bậc hai sai số bình phương trung bình)

* **Công thức toán học:**
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Đơn vị tính:** Mang cùng đơn vị đo với biến mục tiêu $y$.
* **Mục đích dùng để làm gì?**
  * Đưa đại lượng MSE trở lại **cùng thứ nguyên (đơn vị)** với biến mục tiêu để con người có thể đọc hiểu được, đồng thời **bảo tồn nguyên vẹn tính chất phạt nặng các sai số lớn**.
  * Làm công cụ chẩn đoán sự hiện diện của sai số ngoại lai khi đối chiếu cạnh MAE.
* **Khi nào thì dùng?**
  * Khi bạn cần một chỉ số vừa mang đơn vị thực tế, vừa ưu tiên phạt nặng các lỗi dự báo nghiêm trọng.
  * Dùng kết hợp song song với MAE: Bất đẳng thức $\text{RMSE} \ge \text{MAE}$ luôn đúng.
    * Nếu $\text{RMSE} \approx \text{MAE}$: Mọi mẫu dữ liệu đều có độ lệch tương đương nhau.
    * Nếu $\text{RMSE} \gg \text{MAE}$ (RMSE lớn hơn MAE nhiều lần): Là bằng chứng toán học chứng minh mô hình đang có một số mẫu dự đoán sai lệch cực kỳ nghiêm trọng.
* **Khi nào KHÔNG nên dùng độc lập?**
  * Không nên dùng duy nhất một mình RMSE khi tập dữ liệu có phân phối bất đối xứng hoặc nhiều outlier, vì một vài điểm cực trị sẽ kéo RMSE vọt lên cao, dẫn đến nhận định sai lầm rằng mô hình kém dù nó dự đoán rất tốt cho 95% số mẫu còn lại.
* **Câu trả lời lý thuyết ngắn gọn:**
  > *"RMSE là căn bậc hai của MSE. Mục đích là đưa sai số bình phương về lại cùng đơn vị với biến mục tiêu. Ta dùng RMSE khi muốn phát hiện và phạt nặng các lỗi dự đoán lớn. So sánh khoảng cách giữa RMSE và MAE cho biết mức độ phân tán của các sai số ngoại lai."*

---

### 1.4. MedAE (Median Absolute Error - Sai số tuyệt đối trung vị)

* **Công thức toán học:**
  $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_n - \hat{y}_n|)$$
* **Đơn vị tính:** Mang cùng đơn vị đo với biến mục tiêu $y$.
* **Mục đích dùng để làm gì?**
  * Đo lường sai số của **chính xác 50% số đông dữ liệu**.
  * Cung cấp một thước đo bền vững (Robust Metric) tuyệt đối, hoàn toàn không bị ảnh hưởng bởi bất kỳ giá trị ngoại lai cực đoan nào.
* **Khi nào thì dùng?**
  * Dùng khi dữ liệu có phân phối đuôi dày (Heavy-tailed distribution) hoặc chứa các ngoại lai không thể làm sạch.
  * Dùng khi muốn trả lời câu hỏi: *"Với một trường hợp điển hình bình thường của số đông, sai số là bao nhiêu?"*
* **Khi nào KHÔNG nên dùng?**
  * Không dùng khi bạn cần quản trị rủi ro ở các trường hợp biên nguy hiểm (Tail Risk), vì MedAE hoàn toàn bỏ qua 50% các mẫu có sai số lớn nhất.
* **Câu trả lời lý thuyết ngắn gọn:**
  > *"MedAE là giá trị trung vị của các sai số tuyệt đối. Mục đích là phản ánh sai số điển hình của 50% số đông mà hoàn toàn miễn nhiễm với các ngoại lai cực đoan."*

---

### 1.5. Residual (Phần dư) & Tỷ số $\text{RMSE} / \text{MedAE}$

* **Công thức phần dư:**
  $$e_i = y_i - \hat{y}_i \quad (\text{Thực tế} - \text{Dự đoán})$$
* **Mục đích dùng để làm gì?**
  * Kiểm định chẩn đoán các giả định thống kê của mô hình:
    1. **Tính không chệch (Unbiased):** Trung vị và kỳ vọng của phần dư phải xấp xỉ bằng 0 ($\mathbb{E}[e] \approx 0$).
    2. **Tính phương sai không đổi (Homoscedasticity):** Độ phân tán của phần dư phải đồng đều, không bị phình to hình phễu khi giá trị dự đoán tăng lên.
    3. **Tính độc lập:** Các phần dư không được có tương quan chuỗi với nhau.
* **Tỷ số chẩn đoán $\text{RMSE} / \text{MedAE}$:**
  * Nếu phần dư có phân phối chuẩn lý thuyết: Tỷ số $\text{RMSE} / \text{MedAE} \approx 1.48$.
  * Nếu tỷ số vọt lên rất cao ($> 4\times, 6\times$): Đó là bằng chứng toán học chứng minh phân phối sai số có đuôi cực dày (Heavy-tailed), mô hình dự đoán rất chuẩn cho số đông nhưng có những lỗi cá biệt cực lớn.

---

# 2. NHÓM ĐÁNH GIÁ ĐỘ KHỚP (GOODNESS OF FIT)

---

### 2.1. $R^2$ (Coefficient of Determination - Hệ số xác định)

* **Công thức toán học:**
  $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  *(Trong đó: $\text{SS}_{\text{res}}$ là tổng bình phương phần dư; $\text{SS}_{\text{tot}}$ là tổng phương sai tự nhiên của dữ liệu so với giá trị trung bình $\bar{y}$)*
* **Đơn vị tính:** Không có đơn vị (Unitless), giá trị chuẩn nằm trong khoảng $[0.0, 1.0]$.
* **Mục đích dùng để làm gì?**
  * Định lượng **tỷ lệ phần trăm sự biến thiên (phương sai) của biến mục tiêu được giải thích bởi mô hình**.
  * Cung cấp một thang đo chuẩn hóa không thứ nguyên để so sánh chất lượng mô hình giữa các bài toán có quy mô đơn vị khác nhau.
* **Khi nào thì dùng?**
  * Dùng trong hầu hết các bài toán hồi quy để đo mức độ khớp của mô hình so với mốc cơ sở là việc đoán giá trị trung bình $\bar{y}$.
* **Khi nào KHÔNG nên dùng độc lập?**
  * Trong hồi quy tuyến tính đa biến, $R^2$ luôn luôn tăng lên khi ta thêm biến mới vào mô hình (dù biến đó là rác) $\rightarrow$ Lúc đó bắt buộc phải dùng **Adjusted $R^2$** (có phạt số lượng bậc tự do của các biến).

---

### 2.2. Hiện tượng Âm $R^2$ ($R^2 < 0$)

* **Câu hỏi vấn đáp kinh điển:** *"Hệ số xác định là $R$ bình phương, tại sao lại có thể mang giá trị âm?"*
* **Bản chất toán học:**
  * Ký hiệu $R^2$ ở đây là tên gọi quy ước của chỉ số, **không phải là bình phương của hệ số tương quan Pearson** khi đánh giá ngoài mẫu (out-of-sample).
  * Trong công thức: $R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$. Khi đánh giá trên tập kiểm tra độc lập, nếu mô hình đưa ra dự đoán sai lệch nghiêm trọng, làm cho tổng bình phương sai số của mô hình ($\text{SS}_{\text{res}}$) lớn hơn cả tổng bình phương sai lệch khi chỉ đoán trung bình ($\text{SS}_{\text{tot}}$), thì phân số $\frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} > 1 \rightarrow R^2 < 0$.
* **Khi nào xảy ra hiện tượng này?**
  1. Khi mô hình bị **quá khớp cực độ (Catastrophic Overfitting)** trên tập huấn luyện, khi gặp dữ liệu mới thì dự đoán sai lệch hoàn toàn.
  2. Khi áp dụng mô hình hồi quy tuyến tính không có điều chuẩn vào tập dữ liệu có **đa cộng tuyến nghiêm trọng** hoặc có quan hệ **phi tuyến tính mạnh**.
* **Câu trả lời lý thuyết ngắn gọn:**
  > *"Khi đánh giá ngoài mẫu, $R^2$ bị âm khi mô hình dự báo kém hơn cả một đường thẳng nằm ngang dự báo bằng giá trị trung bình $\bar{y}$ của dữ liệu. Điều này phản ánh mô hình bị overfit nặng hoặc hoàn toàn không phù hợp với cấu trúc dữ liệu."*

---

# 3. NHÓM PHÂN PHỐI DỮ LIỆU & XỬ LÝ TARGET

---

### 3.1. Phân phối lệch (Skewness) & Biến đổi $\text{log1p}$ và $\text{expm1}$

* **Bản chất phân phối lệch phải (Right-skewed):** Dữ liệu tập trung đông đảo ở vùng giá trị nhỏ, nhưng có một dải đuôi dài kéo sang bên phải chứa các giá trị cực lớn ($\text{Mean} > \text{Median}$).
* **Công thức toán học:**
  $$y_{\text{transformed}} = \log(1 + x) \quad (\text{log1p})$$
  $$\hat{y}_{\text{original}} = e^{\hat{y}} - 1 \quad (\text{expm1})$$
* **Mục đích dùng để làm gì?**
  1. Nén dải đuôi dài, biến đổi phân phối lệch về dạng **đối xứng chuẩn (Gaussian-like)**.
  2. Đảm bảo tính **phương sai đồng nhất của phần dư (Homoscedasticity)**, ngăn các giá trị cực đại kéo lệch hàm mất mát bình phương.
  3. Phép tính $\log(1+x)$ triệt tiêu nguy cơ lỗi toán học khi dữ liệu có giá trị bằng 0 ($\log(0) = -\infty$).
  4. Phép biến đổi ngược bằng hàm mũ $\text{expm1}$ đảm bảo kết quả dự báo không bao giờ bị âm.
* **Khi nào thì dùng?**
  * Khi biến mục tiêu $y \ge 0$ và có tính biến thiên cấp số nhân hoặc lệch phải nặng: **thu nhập, giá cả tài sản, doanh thu, thời gian phản hồi, lưu lượng truy cập**.
* **Khi nào KHÔNG nên dùng?**
  * Không dùng khi biến mục tiêu có giá trị âm. Không dùng khi phân phối dữ liệu vốn đã đối xứng chuẩn.

---

# 4. CHỐNG RÒ RỈ DỮ LIỆU & QUY TRÌNH ĐÁNH GIÁ (DATA LEAKAGE & PIPELINES)

---

### 4.1. Data Leakage (Rò rỉ dữ liệu) & Cơ chế Pipeline Scaling

* **Định nghĩa học thuật:** Rò rỉ dữ liệu là hiện tượng thông tin từ tập kiểm định/kiểm thử (hoặc thông tin tương lai) vô tình được đưa vào quá trình tiền xử lý hoặc huấn luyện mô hình, khiến kết quả đánh giá trên giấy tờ cao giả tạo nhưng mô hình sụp đổ khi triển khai thực tế.
* **Lỗi rò rỉ phổ biến nhất (Scaling Leakage):**
  * Thực hiện chuẩn hóa `fit_transform` trên toàn bộ tập dữ liệu trước khi chia Train-Test.
  * **Hậu quả:** Giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ của tập Test đã bị lộ cho tập Train học trước.
* **Giải pháp chuẩn mực:**
  * Luôn sử dụng cấu trúc **Pipeline**: Bất kỳ bước biến đổi nào (StandardScaler, One-Hot Encoder, PCA) đều **chỉ được phép `.fit()` trên tập Train**, sau đó chỉ dùng tham số đã học đó để `.transform()` sang tập Validation và Test.

---

### 4.2. Tại sao quy trình Machine Learning cần cả 3 tập: Train – Validation – Test?

* **🎯 Mục đích phân tách 3 tập độc lập:**
  * **Tập Train (Huấn luyện):** Cho thuật toán học các tham số mô hình (weights, bias, ngưỡng phân nhánh).
  * **Tập Validation (Kiểm định):** Cho **người phát triển** nhìn vào để: so sánh nhiều mô hình, chọn siêu tham số (Hyperparameter Tuning), chọn tập đặc trưng. Tập này được dùng lặp đi lặp lại nhiều lần.
  * **Tập Test (Kiểm thử độc lập - Out-of-sample):** Bị giấu kín hoàn toàn, chỉ được mở đúng **1 lần duy nhất** ở cuối quy trình. Đóng vai trò là bài thi tốt nghiệp độc lập để kiểm tra xem mô hình có bị hiện tượng **"học vẹt gián tiếp" (Overfitting on Validation / Information Bleed)** hay không.

---

# 5. CHIẾN LƯỢC CHIA DỮ LIỆU (VALIDATION STRATEGIES)

---

### 5.1. K-Fold Cross-Validation vs. Temporal Sliding-Window Cross-Validation

* **Random K-Fold Cross-Validation:**
  * **Cơ chế:** Xáo trộn ngẫu nhiên dữ liệu thành $K$ phần bằng nhau.
  * **Khi nào dùng?** Dùng cho dữ liệu cắt ngang (Cross-sectional data), không phụ thuộc vào chuỗi thời gian (ví dụ: phân loại ảnh X-quang, nhận diện chữ viết tay).
  * **Hạn chế:** Nếu áp dụng cho dữ liệu có yếu tố thời gian, Random K-Fold sẽ lấy dữ liệu của tương lai để dự đoán quá khứ $\rightarrow$ **Look-ahead Data Leakage**.
* **Temporal Sliding-Window Cross-Validation (Cửa sổ trượt thời gian):**
  * **Cơ chế:** Sắp xếp dữ liệu theo trình tự thời gian. Giữ cửa sổ huấn luyện có kích thước cố định trượt tịnh tiến về tương lai: Khối quá khứ làm Train $\rightarrow$ Khối thời gian liền kề kế tiếp làm Validation.
  * **Mục đích dùng để làm gì?**
    1. Mô phỏng chính xác kịch bản thực tế: Chỉ dùng dữ liệu lịch sử để dự báo tương lai.
    2. Giữ quy mô mẫu huấn luyện đồng đều giữa các Fold để so sánh công bằng.
    3. Thích ứng với sự thay đổi của môi trường theo thời gian (Concept Drift), giúp mô hình luôn học từ dữ liệu gần nhất.

---

# 6. NHÓM THUẬT TOÁN HỒI QUY (REGRESSION ALGORITHMS)

---

### 6.1. Dummy Baseline Regressor (Mô hình mốc cơ sở)

* **Định nghĩa:** Là mô hình đơn giản nhất, không học mối quan hệ của các biến $X$, chỉ đưa ra một dự đoán hằng số duy nhất cho mọi mẫu.
* **Mục đích dùng để làm gì?**
  * Thiết lập một **mốc sàn tối thiểu (Performance Floor)**. Một mô hình học máy phức tạp bắt buộc phải vượt trội hơn Dummy Baseline thì mới chứng minh được tính hiệu quả thực tế.
* **Khi nào dùng Dummy Mean vs Dummy Median?**
  * **Dummy Mean:** Là nghiệm giải tích tối ưu nếu bài toán đo lường bằng hàm mất mát **MSE/RMSE**.
  * **Dummy Median:** Là nghiệm giải tích tối ưu nếu bài toán đo lường bằng hàm mất mát **MAE**. Khi dữ liệu có phân phối lệch và chứa nhiều ngoại lai, Median là mốc sàn chuẩn mực.

---

### 6.2. Hồi quy Tuyến tính OLS vs. Hồi quy Ridge (Điều chuẩn L2)

* **Hồi quy OLS (Ordinary Least Squares):**
  * Cố gắng tìm vector trọng số $w$ để tối thiểu hóa: $\sum (y_i - X_i w)^2$.
  * **Nhược điểm chí mạng:** Khi các đặc trưng có hiện tượng **đa cộng tuyến (Multicollinearity)**, ma trận $(X^TX)$ gần như suy biến (singular), khiến trọng số $w$ bùng nổ cực lớn, mô hình cực kỳ bất ổn và overfit.
* **Hồi quy Ridge (L2 Regularization / Tikhonov):**
  * Thêm thành phần phạt suy giảm trọng số: $\sum (y_i - X_i w)^2 + \alpha \sum w_j^2$.
* **Mục đích của Ridge:**
  * Giúp ma trận $(X^TX + \alpha I)$ luôn khả nghịch $100\%$.
  * Thu nhỏ (shrink) các trọng số về gần 0 (nhưng không triệt tiêu về 0), kiểm soát phương sai của mô hình, triệt tiêu tác hại của đa cộng tuyến.
* **Khi nào dùng Ridge?**
  * Khi muốn duy trì một mô hình tuyến tính đơn giản, dễ giải thích, nhưng tập dữ liệu có nhiều biến phân loại sau One-Hot Encoding gây ra đa cộng tuyến.

---

# 7. NHÓM HỌC KẾT HỢP CÂY QUYẾT ĐỊNH (ENSEMBLE LEARNING)

---

### 7.1. Random Forest (Bagging) vs. Gradient Boosting (Boosting)

| Tiêu chí so sánh | Random Forest (Bagging) | Gradient Boosting (Boosting) |
| :--- | :--- | :--- |
| **Cơ chế cốt lõi** | Xây dựng hàng trăm cây quyết định **độc lập, song song**. Mỗi cây học trên một tập con ngẫu nhiên (Bootstrap sample). | Xây dựng các cây **tuần tự nối tiếp nhau**. Mỗi cây mới học để dự báo **phần dư (Residuals/Gradient)** của các cây đi trước. |
| **Mục đích toán học** | **Giảm phương sai (Reduce Variance)** cho các cây sâu vốn có xu hướng overfit. | **Giảm độ chệch (Reduce Bias)** bằng cách kết hợp nhiều cây nông (weak learners). |
| **Tốc độ & Song song** | Rất nhanh, có thể song song hóa hoàn toàn trên toàn bộ CPU cores (`n_jobs=-1`). | Phải chạy tuần tự từng cây một (cây sau phải đợi cây trước tính xong phần dư). |
| **Độ nhạy tham số** | Cực kỳ ổn định, ít khi bị overfit khi tăng số lượng cây. Dễ tinh chỉnh. | Rất nhạy cảm với tốc độ học (`learning_rate`) và số vòng lặp; dễ overfit nếu không có cơ chế dừng sớm (Early Stopping). |
| **Khi nào chọn?** | Chọn khi cần mô hình mạnh mẽ, chạy nhanh, dữ liệu có nhiều biến phân loại, không muốn tốn nhiều công tinh chỉnh tham số. | Chọn khi muốn vắt kiệt tối đa độ chính xác (như thi Kaggle) và chấp nhận tốn thời gian tuning. |

---

# 8. NHÓM SIÊU THAM SỐ MÔ HÌNH CÂY (TREE HYPERPARAMETERS)

---

### 8.1. `n_estimators` (Số lượng cây quyết định)
* **Mục đích dùng để làm gì?** Quy định số lượng cây độc lập trong rừng để lấy trung bình kết quả.
* **Khi nào tăng? Khi nào dừng lại?**
  * Theo Luật số lớn, việc tăng `n_estimators` trong Random Forest **không làm tăng nguy cơ Overfitting**. Sai số tổng quát hóa sẽ hội tụ về một giới hạn khi số cây đủ lớn.
  * **Nguyên tắc chọn:** Tăng đến khi hàm mất mát kiểm định bão hòa không giảm thêm thì dừng lại để tối ưu hóa thời gian tính toán và dung lượng bộ nhớ.

---

### 8.2. `max_depth` (Độ sâu tối đa của cây)
* **Mục đích dùng để làm gì?** Khống chế số tầng phân nhánh tối đa tính từ nút gốc đến nút lá.
* **Khi nào tăng? Khi nào giảm?**
  * **Tăng:** Khi mô hình bị Underfitting (cây quá nông, không học được các tương tác phi tuyến phức tạp).
  * **Giảm (hoặc chặn không để `None`):** Khi cây phát triển quá sâu (30-40 tầng) dẫn đến Overfitting (học vẹt dữ liệu nhiễu) và làm phình to kích thước file mô hình (Tree bloat).

---

### 8.3. `min_samples_leaf` (Số lượng mẫu tối thiểu tại nút lá)
* **Mục đích dùng để làm gì?** Quy định một nút lá phải chứa tối thiểu bao nhiêu mẫu dữ liệu thì mới được phép dừng phân nhánh và đưa ra kết luận dự báo.
* **Khi nào đặt = 1? Khi nào tăng lên?**
  * **Đặt = 1:** Khi muốn mô hình học được độ phân giải chi tiết ở các nhóm dữ liệu hiếm có giá trị đặc thù. (Trong Random Forest, việc để leaf=1 vẫn an toàn nhờ cơ chế trung bình của hàng trăm cây).
  * **Tăng lên (ví dụ 5, 10):** Khi dữ liệu chứa nhiều nhiễu hạt, muốn làm mịn (smooth) các dự báo và tăng khả năng tổng quát hóa.

---

### 8.4. `max_features` (Tỷ lệ đặc trưng ngẫu nhiên tại mỗi điểm rẽ nhánh)
* **🎯 ĐÂY LÀ ĐẶC TRƯNG CỐT TỬ CỦA THUẬT TOÁN RANDOM FOREST:**
* **Mục đích dùng để làm gì?**
  * **Phi tương quan hóa các cây quyết định trong rừng (De-correlate the trees).**
* **Nguyên lý toán học:**
  * Nếu để `max_features = 1.0` (100%), những đặc trưng quá mạnh (Dominant Features) sẽ luôn luôn được chọn ở nút gốc của hầu hết mọi cây trong rừng, khiến các cây có cấu trúc phân nhánh gần như giống hệt nhau $\rightarrow$ Các cây bị **tương quan cao (Correlated Trees)**.
  * Trong lý thuyết xác suất, trung bình cộng của các biến ngẫu nhiên có tương quan cao với nhau **không làm giảm được nhiều phương sai**:
    $$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
    *(Nếu hệ số tương quan $\rho$ lớn, dù số cây $B \rightarrow \infty$ thì phương sai vẫn không thể triệt tiêu!)*
  * Bằng cách đặt `max_features < 1.0` (ví dụ 0.7 hoặc $\sqrt{p}$), ta ép buộc thuật toán phải che giấu ngẫu nhiên một phần đặc trưng ở mỗi nút. Cây buộc phải rẽ nhánh dựa trên những biến thứ yếu khác.
  * Điều này làm hệ số tương quan $\rho \rightarrow 0$. Khi đó, việc lấy trung bình cộng hàng trăm cây đa dạng sẽ triệt tiêu tối đa phương sai sai số, giúp mô hình đạt độ chính xác vượt trội!

---

# 9. PHƯƠNG PHÁP ĐO TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE)

---

### 9.1. Permutation Feature Importance (PFI) vs. Impurity-based (MDI)

| Tiêu chí | Permutation Importance (PFI) | Impurity-based Importance (MDI / Gini) |
| :--- | :--- | :--- |
| **Cách đo** | Đo trên tập kiểm tra ngoài mẫu (Out-of-sample): Xáo trộn ngẫu nhiên (shuffle) duy nhất 1 cột đặc trưng để phá vỡ mối quan hệ của nó với biến mục tiêu, rồi đo xem **sai số đánh giá (MAE/RMSE) tăng lên bao nhiêu**. | Tính tổng mức giảm phương sai (Variance reduction) hoặc độ thuần khiết (Gini) mà đặc trưng đó đóng góp tại các nút phân nhánh trên tập Train. |
| **Mục đích** | Đo lường mức độ thiệt hại thực tế của mô hình khi mất đi thông tin của một đặc trưng. | Xem nhanh mức độ phụ thuộc thuật toán ngay sau khi huấn luyện. |
| **Ưu điểm** | **Khách quan tuyệt đối**, không bị thiên lệch, mang đơn vị sai số thực tế dễ hiểu. | Tính toán cực nhanh, có sẵn ngay sau khi fit mô hình. |
| **Nhược điểm** | Tốn thời gian tính toán vì phải chạy dự báo lặp lại cho từng biến. | **Bị thiên vị nặng** cho các biến liên tục hoặc biến phân loại có quá nhiều giá trị duy nhất (High Cardinality). |

---

# 10. NHÓM ĐO LƯỜNG ĐỘ BẤT ĐỊNH (UNCERTAINTY METRICS)

---

### 10.1. Dải sai số thực nghiệm $q90$ (Empirical Quantile Error Band)

* **Mục đích dùng để làm gì?**
  * Thay vì chỉ đưa ra một giá trị dự đoán điểm duy nhất (Point Estimate), mô hình cung cấp một **khoảng bất định an toàn (Prediction Interval)** để quản trị rủi ro.
* **Cách tính:** Lấy phân vị 90% (90th percentile) của chuỗi sai số tuyệt đối trên tập kiểm định thực tế.
* **Khi nào dùng?** Dùng khi triển khai mô hình vào thực tế để người sử dụng biết được biên độ dao động an toàn với độ tin cậy thực tế 90%.

---

### 10.2. ECDF (Empirical Cumulative Distribution Function - Hàm phân phối tích lũy thực nghiệm)

* **Công thức toán học:**
  $$\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(|e_i| \le x)$$
* **Mục đích dùng để làm gì?**
  * Cung cấp bức tranh toàn cảnh về phân phối tích lũy sai số từ 0% đến 100%.
  * Giúp xác định trực quan: Điểm cắt $50\%$ trên đồ thị chính là **MedAE**, điểm cắt $90\%$ chính là dải **$q90$**.
  * Đường cong càng dốc đứng về phía góc trên bên trái chứng minh phần lớn các dự đoán có sai số rất nhỏ.
