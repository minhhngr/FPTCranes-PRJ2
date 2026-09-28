# KỊCH BẢN THUYẾT TRÌNH TRANG 05: CHẨN ĐOÁN MÔ HÌNH TỐT NHẤT & ĐỘ BẤT ĐỊNH
> **Mục tiêu:** Chứng minh cho Giảng viên thấy mô hình **Random Forest** được lựa chọn có khả năng **tổng quát hóa vững chắc trên tập kiểm thử tương lai (Holdout Test)**, giải thích rõ cơ chế hoạt động bên trong (Interpretability), quy trình tinh chỉnh siêu tham số bài bản (Tuning) và định lượng độ bất định an toàn trước khi triển khai thực tế.

---

## 1. THÔNG ĐIỆP CỐT LÕI (NÓI TRONG 20 GIÂY MỞ ĐẦU)

> *"Kính thưa Thầy/Cô, sau khi xác định Random Forest là mô hình chiến thắng ở Trang 04, câu hỏi lớn nhất lúc này là: **Mô hình có thực sự đáng tin cậy khi gặp dữ liệu tương lai chưa từng thấy, và sai số rủi ro tối đa là bao nhiêu?**
> Tại Trang 05, chúng em mở khóa tập kiểm thử độc lập gồm **298 công việc của tháng 03/2026**, thực hiện kiểm định chẩn đoán đa chiều: kiểm tra độ khái quát hóa, phân tích phần dư, giải mã 4 bước tinh chỉnh siêu tham số, phân rã tầm quan trọng của các đặc trưng và thiết lập dải tin cậy thực nghiệm $q90$."*

---

## 2. BƯỚC 1: ĐỌC VÀ CHỈ VÀO 6 THẺ KPI ĐẦU TRANG

Chỉ chuột lần lượt vào 6 thẻ KPI màu sắc nổi bật trên cùng:

| Thẻ Metric | Con số hiển thị | Ý nghĩa kỹ thuật & Lời thoại thuyết trình |
| :--- | :---: | :--- |
| **Saved model** | **Random Forest (full 13)** | *"Mô hình chính thức được lưu trữ đóng gói là Random Forest sử dụng toàn bộ 13 đặc trưng gốc."* |
| **Historical test $R^2$** | **0.813** | *"Trên tập kiểm thử tương lai chưa từng thấy, mô hình vẫn giải thích được **81.3%** sự biến thiên của mức lương."* |
| **Historical test MAE** | **$14,735** | *"Sai số tuyệt đối trung bình chỉ là **$14,735**, thậm chí còn tốt hơn cả mức kiểm định cross-validation ($15,594)."* |
| **Historical test MedAE** | **$4,347** | *"Sai số trung vị cực thấp: 50% số công việc trên thị trường được dự đoán lệch không quá **$4,347**."* |
| **Test - CV $R^2$ gap** | **-0.011** | *"Độ sụt giảm $R^2$ giữa tập phát triển và tập kiểm thử chỉ vỏn vẹn **0.011 (1.1%)**, chứng minh mô hình hoàn toàn không bị Overfitting!"* |
| **Empirical q90** | **±$40,068** | *"Khoảng sai số biên an toàn bao phủ 90% các vị trí tuyển dụng trên thị trường là ±$40,068."* |

---

## 3. BƯỚC 2: PHÂN TÍCH CHI TIẾT 5 TAB CHẨN ĐOÁN

---

### TAB 1: GENERALIZATION (KHẢ NĂNG TỔNG QUÁT HÓA)
* **Mục đích:** Chứng minh mô hình không "học vẹt".

#### 1. Biểu đồ Dollar-Error Comparison (CV vs Test)
* **Hành động:** Chỉ vào các cặp cột so sánh (CV màu xanh/cam vs Test màu đỏ):
  * **MAE:** Giảm từ **$15,594 (CV)** xuống **$14,735 (Test)** $\rightarrow$ **Cải thiện $859 (-5.5%)**.
  * **MedAE:** Giảm từ **$7,280 (CV)** xuống **$4,347 (Test)** $\rightarrow$ **Cải thiện $2,933**.
  * **RMSE:** Từ **$26,747 (CV)** lên **$29,111 (Test)**.
* **Biểu đồ $R^2$ bên cạnh:** Giữ vững từ **0.824 (CV)** sang **0.813 (Test)**.

#### 2. Giải thích câu hỏi của Giảng viên: *"Tại sao Test MAE lại thấp hơn cả CV MAE?"*
* **Lời thoại thuyết phục:**
  > *"Thưa Thầy/Cô, hiện tượng Test MAE tốt hơn CV MAE có 2 nguyên nhân khoa học:
  > 1. Mô hình cuối cùng được huấn luyện trên **toàn bộ 1,201 dòng** của tập Development, do đó nó sở hữu dung lượng tri thức lớn hơn so với lúc chạy CV (mỗi fold chỉ được học trên ~960 dòng).
  > 2. Trong 5 fold CV, Fold 4 gặp phải giai đoạn biến động lương bất thường kéo MAE lên $20k; trong khi tháng test 03/2026 thị trường đi vào quỹ đạo ổn định hơn."*

---

### TAB 2: RESIDUAL DIAGNOSTICS (CHẨN ĐOÁN PHẦN DƯ DỰ BÁO)
* **Mục đích:** Kiểm tra mô hình có bị thiên lệch hệ thống hay vi phạm các giả định thống kê không.

#### 1. Biểu đồ Actual vs. Predicted (Thực tế vs Dự đoán)
* **Hành động:** Chỉ vào đường chéo nét đứt $45^\circ$ ($y = x$).
* **Lời thoại:**
  > *"Các điểm dữ liệu phân tán bám rất sát đường chéo $y=x$. Không hề có hiện tượng cong võng hay tòe loe hình phễu ở dải lương phổ thông (80k - 250k USD). Điều này khẳng định phép biến đổi log1p đã phát huy tối đa tác dụng ổn định phương sai."*

#### 2. Biểu đồ Histogram phần dư (Residual Distribution)
* **Hành động:** Chỉ vào cột đỉnh xung quanh vạch số 0.
* **Lời thoại:**
  > *"Phần dư $y - \hat{y}$ có phân phối chuẩn đẹp đối xứng quanh 0. Trung vị phần dư là **-$940**, nghĩa là mô hình không bị thiên vị (unbiased), không có khuynh hướng luôn đoán quá cao hay luôn đoán quá thấp."*

---

### TAB 3: TUNING (QUY TRÌNH TINH CHỈNH SIÊU THAM SỐ 4 BƯỚC)
* **Mục đích:** Chứng minh bộ tham số không phải là con số bốc ngẫu nhiên mà là kết quả khảo sát khoa học.

Mở rộng phần hướng dẫn phương pháp và giải thích 4 bước khảo sát độ nhạy (Sensitivity Sweeps):

```
Quy trình: Khảo sát đơn tham số tuần tự (Coordinate Descent / 1D Sweeps)
Bộ tham số tối ưu: n_estimators=200 | max_depth=20 | min_samples_leaf=1 | max_features=0.7
```

* **Bước 1: `n_estimators = 200`**
  * Khảo sát từ 100 đến 400 cây. Tại 200 cây, sai số CV MAE giảm chạm đáy. Tăng lên 300 hay 400 cây thì sai số không giảm thêm nhưng làm tăng gấp đôi độ trễ tính toán.
* **Bước 2: `max_depth = 20`**
  * Khảo sát từ 10 đến `None`. Nếu độ sâu nhỏ hơn 15 thì mô hình bị thiếu khớp (underfit), nếu để thả nổi `None` cây sẽ sâu tới 40 tầng và ghi nhớ nhiễu. Độ sâu 20 là điểm cân bằng lý tưởng.
* **Bước 3: `min_samples_leaf = 1`**
  * Khảo sát 1, 2, 4, 8. Chọn 1 giúp mô hình phân biệt được các công việc chuyên biệt có số lượng ít mà không sợ overfit nhờ cơ chế trung bình của rừng cây.
* **Bước 4: `max_features = 0.7` (ĐIỂM NHẤN CẦN NHẤN MẠNH MẠNH MẼ)**
  * **Lời thoại:**
    > *"Thưa Thầy/Cô, việc đặt `max_features = 0.7` là mấu chốt kỹ thuật: Nếu để 1.0 (100%), biến cực mạnh `job_category` sẽ luôn được chọn ở nút gốc của cả 200 cây, khiến các cây bị tương quan cao (correlated trees). Bằng cách ép mỗi nút phân nhánh chỉ được xem xét ngẫu nhiên 70% số đặc trưng, chúng em buộc các cây phải phân nhánh theo kinh nghiệm, quy mô công ty, hình thức làm việc... Điều này giúp **phi tương quan hóa các cây (de-correlate trees)**, phát huy tối đa sức mạnh giảm phương sai của thuật toán Bagging."*

---

### TAB 4: FEATURE RELIANCE & ABLATION (GIẢI THÍCH MÔ HÌNH & TINH GIẢM ĐẶC TRƯNG)

#### 1. Biểu đồ Permutation Importance (Độ quan trọng xáo trộn trên Raw Data)
* **Hành động:** Chỉ vào 2 cột dài nhất:
  * **`job_category`:** Khi xáo trộn ngẫu nhiên, MAE tăng vọt thêm **+$49,598/job**!
  * **`years_of_experience`:** Khi xáo trộn, MAE tăng vọt thêm **+$12,379/job**!
  * Các biến còn lại (`company_size`, `work_setting`, `geographic_region`): Đóng góp tinh chỉnh mức chênh lệch nhỏ hơn.
* **Kết luận:** Vị trí chức danh và số năm kinh nghiệm quyết định hơn 80% giá trị gói thu nhập.

#### 2. Thử nghiệm Tinh giản (Ablation Study): Full 13 vs. Top-2 Features
* Chỉ vào bảng so sánh giữa mô hình Full (13 biến) và Top-2 (chỉ dùng `job_category` + `years_of_experience`):
  * **Full Model (13 biến):** Test MAE = **$14,735** | Test $R^2$ = **0.813**.
  * **Top-2 Model (2 biến):** Test MAE = **$13,829** | Test $R^2$ = **0.820**.
* **Giải thích góc nhìn chuyên gia:**
  > *"Mô hình Top-2 thậm chí có sai số kiểm thử thấp hơn nhờ loại bỏ các nhiễu của các biến phụ. Chúng em giữ mô hình Top-2 làm công cụ ước lượng nhanh (Quick Benchmark) cho người dùng chỉ muốn nhập 2 thông tin cơ bản; trong khi mô hình Full 13 biến được dùng khi ứng viên muốn đánh giá chi tiết theo quy mô công ty và chính sách Remote."*

---

### TAB 5: TRUST AND AUDIT (ĐỘ TIN CẬY & MINH CHỨNG THỰC TẾ)

#### 1. Định lượng rủi ro với Khoảng tin cậy thực nghiệm $q90$ ($\pm \$40,068$)
* **Hành động:** Chỉ vào đường cong ECDF tích lũy và vạch đứt nét $q90$.
* **Lời thoại:**
  > *"Trong triển khai thực tế, không có mô hình nào đoán đúng tuyệt đối 100%. Vì vậy chúng em cung cấp **Dải tin cậy thực nghiệm $q90$ là $\pm \$40,068$**. Đường cong ECDF chứng minh chính xác **89.9%** các công việc trên tập kiểm thử có sai số thực tế nằm gọn trong dải này. Điều này giúp doanh nghiệp có một biên độ dự phòng ngân sách minh bạch."*

#### 2. Tỷ số $\text{RMSE} / \text{MedAE} = 6.7\times$
* Giải thích cho Hội đồng: Tỷ số này cao ($29.1k / $4.3k) vì sai số bị chi phối bởi một nhóm nhỏ các hồ sơ lương khủng, trong khi đa số công việc đều có độ chuẩn xác cực cao.

#### 3. Bảng 3 ví dụ kiểm chứng thực tế (Traceable Benchmark Examples)
* Chỉ vào 3 dòng ví dụ cụ thể (ví dụ: AI Compliance Manager, Senior Data Scientist):
  * Đối chiếu Lương thực tế vs Lương dự đoán Full vs Lương dự đoán Top-2.
  * Minh chứng rằng hệ thống hoàn toàn minh bạch, có thể truy xuất nguồn gốc (audit trail) đến từng bản ghi cụ thể.

---

## 4. BỘ CÂU HỎI PHẢN BIỆN CỦA HỘI ĐỒNG & CÁCH TRẢ LỜI

### Câu 1: "Tại sao nhóm không dùng SHAP Values mà dùng Permutation Importance?"
* **Trả lời:**
  > *"Dạ thưa Thầy/Cô, Permutation Feature Importance có ưu điểm vượt trội là được tính toán trực tiếp trên sự gia tăng của sai số tiền tệ thực tế (MAE tăng bao nhiêu USD) khi một đặc trưng bị vô hiệu hóa, giúp các bên liên quan phi kỹ thuật (HR, Giám đốc tài chính) hiểu ngay tác động tài chính. Ngoài ra, Permutation Importance trên tập out-of-sample phản ánh chính xác giá trị thực tế của đặc trưng trên dữ liệu mới mà không bị thiên lệch bởi số lượng nhánh cây như Impurity (MDI)."*

### Câu 2: "Khoảng $q90 = \pm \$40,068$ có quá rộng không đối với một dự báo lương?"
* **Trả lời:**
  > *"Dạ thưa Thầy/Cô, khoảng $q90$ đại diện cho **phân vị 90%** (nghĩa là bao phủ hầu như toàn bộ các trường hợp khó nhất, bao gồm cả các vị trí cấp cao). Nếu xét ở mức độ điển hình, **MedAE của chúng em chỉ là $4,347** (tức 50% hồ sơ chỉ lệch dưới $4,300), và MAE là $14,735 (chỉ chiếm ~7-8% của mức lương trung bình 180k-200k). Việc đưa ra $q90$ rộng là thể hiện tính trung thực khoa học, cảnh báo cho người dùng về rủi ro ở các trường hợp biên."*

### Câu 3: "Việc thử nghiệm tinh chỉnh 4 siêu tham số này có phải là GridSearch đầy đủ không?"
* **Trả lời:**
  > *"Dạ thưa Thầy/Cô, đây là phương pháp khảo sát độ nhạy tuần tự (Sequential 1D Sensitivity Sweeps) sau khi đã thiết lập mỏ neo bằng một lưới ban đầu. Cách tiếp cận này giúp tiết kiệm thời gian huấn luyện gấp nhiều lần so với Full GridSearch đa chiều (vốn đòi hỏi hàng nghìn lượt fit) nhưng vẫn bảo đảm tìm được vùng tối ưu địa phương vững chắc cho từng tham số chính."*
