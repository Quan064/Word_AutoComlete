# TOPIC-AWARE AUTOCOMPLETE

# I. Giới thiệu

### Bối cảnh và mục đích bài toán

Trong các hệ thống soạn thảo văn bản hiện đại, chức năng tự động gợi ý từ (autocomplete) đóng vai trò quan trọng trong việc:

- Tăng tốc độ nhập liệu
- Giảm lỗi chính tả
- Hỗ trợ người dùng diễn đạt ý tưởng hiệu quả hơn

Tuy nhiên, các phương pháp autocomplete truyền thống thường chỉ dựa trên:

- Tiền tố (prefix)
- Tần suất xuất hiện của từ

Do đó, chúng gặp hạn chế lớn:

- Không hiểu ngữ cảnh của câu đang viết
- Không phân biệt được các từ cùng tiền tố nhưng khác nghĩa


### Vấn đề đặt ra

Xét ví dụ:

```
machine learning is used for image p...

Một hệ thống truyền thống có thể gợi ý:
- people
- police
- party

Trong khi từ đúng theo ngữ cảnh là:
- processing
- prediction
```

Điều này cho thấy cần một hệ thống có khả năng:

- hiểu ngữ cảnh ngữ nghĩa (semantic context)
- kết hợp với tìm kiếm nhanh theo tiền tố


### Mục tiêu của đề tài

Đề tài hướng đến việc xây dựng một hệ thống autocomplete cho văn bản tiếng Anh có khả năng:

- Gợi ý từ dựa trên tiền tố người dùng nhập
- Đồng thời phù hợp với chủ đề của văn bản

Thông qua việc kết hợp:

- **Trie** → Tìm kiếm nhanh theo tiền tố
- **LDA & CGS** → Mô hình hóa chủ đề của văn bản

Các thuật toán này được chọn vì chúng có độ phức tạp thời gian nhỏ, đơn giản và dễ triển khai.


# II. Tìm hiểu thuật toán

## 1. Cấu trúc dữ liệu Trie

### Khái niệm

**Trie** (Prefix Tree) là một cấu trúc dữ liệu dạng cây được sử dụng để lưu trữ và truy vấn tập hợp các chuỗi. Mỗi nút trong Trie đại diện cho một ký tự.


### Cách cài đặt

Ví dụ với các từ:

```
cat, car, dog
```

Trie sẽ có dạng:

```
(root)
 ├── c
 │    ├── a
 │         ├── t
 │         └── r
 └── d
      └── o
           └── g
```


### Tìm kiếm theo tiền tố

Khi người dùng nhập: `ca`

Trie sẽ:

- Đi theo nhánh `c` → `a`
- Duyệt toàn bộ các từ phía dưới

→ trả về:

```
cat, car
```


### Sắp xếp kết quả theo tần suất (Heap Sort)

#### Vấn đề

Trie chỉ trả về danh sách các từ khớp với prefix, nhưng **không sắp xếp theo mức độ phổ biến**. Người dùng thường mong muốn những từ **dùng thường xuyên nhất** được gợi ý trước.

#### Giải pháp: Heap Sort by Frequency

**Ý tưởng:**

- Mỗi nút từ trong Trie lưu trữ thêm **tần suất xuất hiện** của từ đó
- Khi tìm kiếm, sau khi duyệt nhánh trie, sử dụng **Heap (Min-Heap)** để sắp xếp các từ theo tần suất giảm dần
- Trả về top-K kết quả có tần suất cao nhất

#### Cấu trúc dữ liệu nút Trie

```
Node {
  char: character,
  children: Map<char, Node>,
  frequency: int,        // ← Thêm tần suất
  is_end_of_word: bool
}
```

#### Ví dụ

Với từ khóa `ca`:

```
Trie trả về: cat, car, can, call, case

Tần suất:
- cat:  100 (phổ biến)
- car:  450 (rất phổ biến)
- can:  320 (khá phổ biến)
- call: 150 (ít phổ biến)
- case:  80 (rất ít)

Sau Heap Sort (top-3):
1. car  (450)
2. can  (320)
3. cat  (100)
```

### Ưu điểm và hạn chế

- Ưu điểm:

    - Tìm kiếm nhanh theo prefix
    - Phù hợp autocomplete

- Hạn chế:

    - Không hiểu nghĩa của từ được gợi ý
    - Không xét ngữ cảnh


## 2. Mô hình LDA (Latent Dirichlet Allocation)

### Ý tưởng chính

LDA là một mô hình xác suất dùng để khám phá các “chủ đề ẩn” trong tập văn bản.


### Giả định của LDA

LDA giả định:

- Mỗi tài liệu là sự kết hợp của nhiều chủ đề
- Mỗi chủ đề là một phân bố xác suất trên các từ


### Ví dụ trực quan

Một document có thể bao gồm:

- `technology` : 70%
- `education` : 30%

Topic `technology` có thể gồm:

- `code`, `AI`, `data`, `algorithm`, $\dots$


### Biểu diễn xác suất

Từ các giả định này, LDA lập ra hai phân phối xác suất:

1. Phân bố từ theo chủ đề: $P(\text{word} \mid \text{topic})$

2. Phân bố chủ đề theo tài liệu: $P(\text{topic} \mid \text{document})$


### Ứng dụng trong bài toán

Trong hệ thống autocomplete:

- Phần văn bản đã nhập (context) → xem như một “document”
- LDA sử dụng một thuật toán là **CGS** để suy diễn $P(\text{topic} | \text{context})$, nhờ đó xác định được ngữ cảnh hiện tại


## 3. Thuật toán CGS (Collapsed Gibbs Sampling)

### Ý tưởng

CGS là phương pháp suy diễn trong LDA dựa trên phương pháp lấy mẫu tuần tự, cập nhật nhãn chủ đề cho từng từ trong tài liệu tùy theo số lượng chủ đề được yêu cầu.

CGS có 2 giả định:

 - Một tài liệu thường có một chủ đề độc tôn
 - Một từ thường có một nghĩa độc tôn


### Quá trình hoạt động của CGS

Cho CSG một tập tài liệu, số lượng chủ đề cần lọc $(K)$, $\alpha$ và $\beta$ (đôi lúc bị thay bằng $\eta$).

1. Ngẫu nhiên gán một chủ đề cho từng từ trong mọi tài liệu.
(Những từ giống nhau có thể có chủ đề khác nhau)

Xét một tài liệu $d$ cụ thể:

1. Đếm số lần mỗi chủ đề $k$ xuất hiện trong tài liệu $d$.
2. Đếm số lần mỗi từ xuất hiện trong từng chủ đề trên toàn tập dữ liệu.
3. Trong tài liệu $d$, bỏ gán một từ $w_{d,n}$ khỏi chủ đề của nó.
4. Gán một chủ đề mới cho $w_{d,n}$ dựa trên:
  - Tài liệu $d$ thường nói về chủ đề nào (lấy từ bước 2).
  - Tần xuất  $w_{d,n}$ xuất hiện trong từng chủ đề (lấy từ bước 3).
  
  Công thức:
  
  $$
  \frac{n_{d,k}+\alpha}{\sum^K_in_{d,i}+\alpha K}\times\frac{m_{w,k}+\beta}{\sum^V_im_{i,k}+\beta V}
  $$
  
  Trong đó:
  
  - $n_{d,k}$ : Số từ thuộc chủ đề $k$ xuất hiện trong tài liệu $d$
  - $\sum^K_in_{d,i}$ : Số từ trong tài liệu $d$
  - $m_{w,k}$ : Số lần $w_{d,n}$ xuất hiện trong chủ đề $k$
  - $\sum^V_im_{i,k}$ : Số lượng từ vựng không lặp lại trong chủ đề $k$
  - $\alpha \in [0, 1]$ : Mức độ phân tán chủ đề trung bình của toàn bộ tài liệu 
  - $\beta \in [0, 1]$ : Mức độ phân tán nghĩa trung bình của toàn bộ từ
  
  Kết quả của công thức là tỷ lệ gán nhãn $k$ cho $w_{d,n}$ (gán ngẫu nhiên có tỷ lệ).

↻ Lặp lại bước 2-5 cho đến khi nào thấy đủ thì dừng lại.


### Kết quả

Sau nhiều lần lặp, CGS thu được:
- Phân phối từ trong topic - $\phi$, với $\phi_{k,w}​=P(\text{word}=w \mid \text{topic}=k)$
- Phân phối topic trong document - $\theta$, với $\theta_{d,k}​=P(\text{topic}=k \mid \text{document}=d)$


### Ưu điểm và hạn chế

#### Ưu điểm

- Đơn giản, dễ triển khai
- Dễ giải thích cách hoạt động: hiểu được cách một chủ đề được gán cho một từ

#### Hạn chế

- Chậm do cần nhiều vòng lặp (không phù hợp trực tiếp cho realtime nếu không tối ưu)
- Chỉ coi văn bản là túi từ nên không hiểu được quan hệ vị trí giữa các từ với nhau.


## 4. Kết hợp Trie và LDA

### 4.1. Chuẩn bị dữ liệu (Offline)

#### Bước 1 - Thu thập và tiền xử lý văn bản

- **LDA**
  - **Lowercase**: viết thường toàn bộ văn bản
  - **Tokenization**: chuyển đổi chuỗi văn bản thành các đơn vị rời rạc (thường là từ) gọi là token.
  - **Non-alphabetic tokens filtering**: lọc những token không phải chữ cái
  - **Punctuation removal**: loại bỏ dấu câu
  - **Space removal**: loại bỏ khoảng trắng
  - **Stopword removal**: loại bỏ từ dừng (những từ ít đóng góp nghĩa cho chủ đề văn bản)
  - **POS filtering**: chỉ giữ lại những từ loại mang nhiều thông tin (danh từ, động từ, tính từ)
  - **Extreme words filtering**: loại bỏ những từ hiếm xuất hiện hoặc quá phổ biến
  - **Lemmatization**: đưa các dạng khác nhau của một từ về một dạng chuẩn duy nhất.
- **Trie**
  - **Tokenization**
  - **Non-alphabetic tokens filtering**
  - **Punctuation removal**
  - **Space removal**

Việc LDA có thêm Stopword removal, POS filtering, Extreme words filtering và Lemmatization giúp giảm bớt chi phí cho việc tính chủ đề của những từ tần xuất cao nhưng vô nghĩa, đôi khi còn gây nhiễu cho mô hình.

#### Bước 2 - Huấn luyện LDA

Dùng tập văn bản để huấn luyện LDA và thu được:

1. **Topic → Word distribution**

  $$
  P(\text{word} | \text{topic})
  $$

  Ví dụ

  | Chủ đề | WordId_0: machine | WordId_1: learning | WordId_2: cat | WordId_3: fish |
  | --- | --- | --- | --- | --- |
  | Topic_1 (Technology) | 0.45 | 0.45 | 0.05 | 0.05 |
  | Topic_2 (Animal) | 0.02 | 0.03 | 0.60 | 0.35 |

2. **Word → Topic distribution**

  $$
  P(\text{topic} | \text{word})
  $$

  **Chuyển đổi**

  | Từ | Phân phối chủ đề [Technology, Animal] |
  | --- | :---: |
  | WordId_0 (machine) | [0.45, 0.02] |
  | WordId_1 (learning) | [0.45, 0.03] |
  | WordId_2 (cat) | [0.05, 0.60] |
  | WordId_3 (fish) | [0.05, 0.35] |

  **Tác dụng**: xác định được tỷ lệ thuộc chủ đề bất kỳ của mỗi từ.

#### Bước 3 - Xây dựng Trie

Từ toàn bộ vocabulary, xây **Trie index**.

Mỗi từ trong Trie lưu thêm:

- frequency
- topic distribution

Ví dụ

```
machine
- frequency: 3200
- topic:
  + Technology: 0.45
  + Animal: 0.02
```

**Tác dụng**

Trie không chỉ lưu chữ mà còn lưu **thông tin chủ đề**.

---

### 4.2. Quy trình gợi ý khi người dùng đang nhập (Online)

Giả sử người dùng đang viết:

```
machine learning is very po
```

Ta tách thành hai phần.

```
context = "machine learning is very"
prefix  = "po"
```

---

### 4.3. Phân tích ngữ cảnh bằng LDA

#### Bước 1 — Tokenize phần nội dung đã viết (Stop Word Removal & POS Tagging & Lemmatization)

(~~"is"~~, ~~"very"~~)
```
["machine", "learning"]
```

#### Bước 2 — Suy diễn topic distribution

Ta xác định được topic distribution của ngữ cảnh bằng trung bình cộng topic distribution của các token (Linearized LDA):

$$\begin{bmatrix} 0.45 \\ 0.02 \end{bmatrix} + \begin{bmatrix} 0.45 \\ 0.03 \end{bmatrix} = \begin{bmatrix} 0.90 \\ 0.05 \end{bmatrix} \xrightarrow{Normalize} \begin{bmatrix} 0.947 \\ 0.053 \end{bmatrix}$$

Kết quả ngữ cảnh:

```
Technology: 0.947
Animal: 0.053
```

Cách tính trung bình này nhanh hơn nhiều so với việc hội tụ CGS gốc nhưng đồng thời cũng đánh đổi độ chính xác.

---

### 4.4. Lấy candidate từ Trie

Tìm các từ bắt đầu bằng prefix.

```
prefix = "po"
```

Trie trả về:

```
power
point
policy
politics
polynomial
```

---

### 4.5. Tính điểm gợi ý có điều kiện theo chủ đề

Mỗi từ có phân bố topic:

```
power:
- freq: 6800
- topic:
  + AI: 0.40
  + Physics: 0.45
  + ...

policy:
- freq: 5000
- topic:
  + Politics: 0.80
  + ...
```

Ta tính **độ phù hợp với context** của một từ bằng một hàm Heuristic ($\alpha$ là hyperparameters):

$$\text{score}(w) = \frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)} \times \left(1 + \alpha \cdot \text{cosineˍsim}(\text{topic}(w), \text{topic}(\text{context}))^2\right)$$

**Ý nghĩa**: Từ nào có score cao hơn thì sẽ được ưu tiên gợi ý.

---

### 4.6. Xếp hạng kết quả

Ví dụ Autocomplete hiển thị:

| word | score |
| --- | --- |
| power | 0.62 |
| polynomial | 0.55 |
| point | 0.32 |
| policy | 0.05 |

---

### 4.7. Phân tích độ phức tạp thời gian (Online)

#### Tổng quan quy trình

Quy trình gợi ý online gồm các bước chính:

1. **Tokenize & tiền xử lý context**
2. **Suy diễn topic distribution của context (Linearized LDA)**
3. **Tìm từ khớp prefix trong Trie**
4. **Tính điểm cho mỗi từ**
5. **Lấy top-K kết quả**

#### Phân tích từng bước

| Bước | Độ phức tạp | Mô tả |
| --- | --- | --- |
| **1. Tokenize & xử lý context** | $O(C)$ | $C$ = độ dài context (ký tự) |
| **2. POS tagging & Lemmatization** | $O(n)$ hoặc $O(n \log V)$ | $n$ = số token ~ $C/5$; $V$ = vocabulary size |
| **3. Lấy topic distribution** | $O(n \cdot K)$ | $n$ = số token; $K$ = số topics (~15-30) |
| **4. Normalize topic vector** | $O(K)$ | Chia vector cho tổng |
| **5. Tìm nhánh prefix trong Trie** | $O(L)$ | $L$ = độ dài prefix (~3-5 ký tự) |
| **6. Duyệt toàn bộ từ khớp** | $O(M)$ | $M$ = số ký tự trong nhánh (khác L) |
| **7. Heap sort top-K** | $O(N_m \log K)$ | $N_m$ = số từ khớp prefix (~10-100) |
| **8. Tính điểm (scoring)** | $O(N_m \cdot K)$ | Cosine similarity cho mỗi từ |
| **9. Lấy top-K cuối cùng** | $O(N_m \log K)$ | QuickSelect hoặc HeapSort |

#### Độ phức tạp tổng thể

$$T(n) = O(C + n \cdot K + L + N_m \cdot K + N_m \log K)$$

**Đơn giản hóa** (bước LDA và scoring chiếm thời gian chủ yếu):

$$T(n) \approx O(n \cdot K + N_m \cdot K)$$

#### Phân tích trường hợp thực tế

Với các giá trị điển hình cho hệ thống:

- $C$ ~100-500 ký tự (context)
- $n$ ~10-30 token (sau tokenize, loại bỏ stopword)
- $K$ ~15-30 topics (từ LDA)
- $L$ ~3-5 ký tự (prefix)
- $N_m$ ~10-100 từ khớp (phụ thuộc prefix)

| Trường hợp | $n$ | $N_m$ | $K$ | Thời gian (ms) | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| **Tốt nhất** | 10 | 10 | 15 | ~2-3 | Prefix đặc thù, context ngắn |
| **Trung bình** | 20 | 50 | 20 | ~10-15 | Prefix phổ biến, context trung bình |
| **Xấu nhất** | 30 | 100 | 30 | ~30-50 | Prefix chung (ví dụ "a"), context dài |

#### Kết luận

**Độ phức tạp chủ yếu:** $O((n + N_m) \cdot K)$

- Thường dưới 50ms (phù hợp realtime)
- Bottleneck chính: Tính điểm ($scoring step$) với $O(N_m \cdot K)$

---

### 4.7. Kiến trúc tổng thể

```
                 Dataset
                    │
        ┌───────────┴───────────┐
        │                       │
   Train LDA                Build Trie
        │                       │
Topic-word matrix         Prefix index
        │                       │
        └──────────┬────────────┘
                   │
             User typing
                   │
        split(context, prefix)
                   │
        ┌──────────┴──────────┐
        │                     │
  LDA inference           Trie search
  P(topic|context)        prefix list
        │                     │
        └──────────┬──────────┘
                   │
           Topic-weighted ranking
                   │
              Autocomplete
```


# III. Dữ liệu kiểm thử thuật toán

## 1. Kích cỡ bộ dữ liệu

Bộ dữ liệu gồm **90,000 tài liệu** (lấy từ *CNN News Corpus*):

- **Tập huấn luyện (Training set):** 54,000 tài liệu (60%)
- **Tập xác thực (Validation set):** 18,000 tài liệu (20%)
- **Tập kiểm thử (Test set):** 18,000 tài liệu (20%)

## 2. Quy trình cài đặt

### 2.1. Chọn các giá trị siêu tham số

Trước khi huấn luyện, xác định các cấu hình mô hình cần so sánh:

- Số chủ đề: $K = 15, 20, 25, 30$
- Các tham số Dirichlet: $\alpha$, $\beta$

Mỗi cấu hình tương ứng với một mô hình LDA khác nhau.

### 2.2. Huấn luyện mô hình trên tập huấn luyện

Với mỗi cấu hình:

1. Sử dụng **54,000 tài liệu** từ tập huấn luyện
2. Chạy **Gibbs Sampling** qua nhiều vòng lặp
3. Sau khi hội tụ, thu được:
   - Phân phối từ của mỗi chủ đề: $\phi_k$
   - Phân phối chủ đề của tài liệu: $\theta_d$

Các tham số này được giữ cố định sau khi huấn luyện.

### 2.3. Suy diễn chủ đề cho tập xác thực

Áp dụng mô hình đã huấn luyện lên **18,000 tài liệu xác thực**:

- Khởi tạo chủ đề cho các từ một cách ngẫu nhiên

Trong quá trình này:

- **Không cập nhật** phân phối chủ đề – từ $\phi_k$
- Chỉ chạy **Linearized LDA** để suy diễn $\theta_d$ cho tập xác thực

### 2.4. Đánh giá mô hình trên tập xác thực

Sau khi suy diễn:

> Đánh giá các cấu hình thông qua cảm nhận chủ quan của con người. Phương pháp này được lựa chọn thay vì dùng một chỉ số đánh giá (ví dụ như Coherence) là vì hiện vẫn còn nhiều tranh cãi về tính đúng đắn của những chỉ số này vì chúng đôi khi đưa ra kết quả không tương đồng so với chất lượng của topic model.

→ So sánh các cấu hình để chọn mô hình tốt nhất.

Tiếp tục dùng cấu hình tốt nhất này để tìm một hàm Heristic (kết hợp Trie và LDA) cho ra điểm số cao nhất khi tính điểm **Hit@K** (tỷ lệ truy vấn có kết quả đúng nằm trong top K) trên tập Validation.

### 2.5. Đánh giá tổng quát

Sử dụng **18,000 tài liệu kiểm thử** (hoàn toàn độc lập):

- So sánh:
  - Mô hình kết hợp **Trie Freq + LDA** (cấu hình tốt nhất + Hàm Heuristic đã tìm được)
  - Mô hình dùng **Trie Freq**
  - Mô hình chỉ dùng **Trie**

- Thước đo:
  - **Hit@K** (tỷ lệ truy vấn có kết quả đúng nằm trong top K)

## 3. Ý nghĩa của cách chia dữ liệu

Cách chia này giúp:

- Hạn chế **overfitting**
- Đảm bảo đánh giá khách quan trên dữ liệu chưa từng thấy
- Phản ánh đúng hiệu suất tổng quát của mô hình


# IV. Kiểm nghiệm

## 1. Phân tích tập huấn luyện cho LDA

Tập huấn luyện cho LDA bao gồm:
- Trước khi tiền xử lý:
  - Số tài liệu: 53 928
  - Tổng lượng token: 34 707 979
  - Kích thước tập từ vựng: 189 878
  - Trung bình độ dài tài liệu: 643.60
  - Độ lệch chuẩn độ dài tài liệu: 337.28
- Sau khi tiền xử lý:
  - Số tài liệu: 53 926
  - Tổng lượng token: 13 435 668
  - Kích thước tập từ vựng: 25 185
  - Trung bình độ dài tài liệu: 249.15
  - Độ lệch chuẩn độ dài tài liệu: 131.98

**Bảng phân tích**

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Histogram%20of%20Document%20Lengths.png?raw=true" alt="Not found">
  <figcaption>Phổ độ dài tài liệu tập huấn luyện của LDA sau khi tiền xử lý</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Total%20Tokens.png?raw=true" alt="Not found">
  <figcaption>Kích thước tập từ vựng trước và sau khi tiền xử lý</figcaption>
</figure>

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Dataset/Vocabulary%20Size.png?raw=true" alt="Not found">
  <figcaption>Tổng lượng token trước và sau khi tiền xử lý</figcaption>
</figure>

**Nhận xét**

Hiệu quả tiền xử lý rất tốt:
- Vocabulary giảm từ 189,878 → 25,185 (cắt giảm ~87%)
- Tổng tokens giảm từ 34.7M → 13.4M (cắt giảm ~61%)
- Chỉ loại bỏ noise, mà còn giữ lại được phần lớn tài liệu (53,928 → 53,926): gần như không mất dữ liệu

→ Tập huấn luyện được tiền xử lý rất kỹ, lượng dữ liệu đủ, phân bố hợp lý → điều kiện tốt để huấn luyện LDA với kết quả chất lượng cao

## 2. Lựa chọn siêu tham số và hàm Heuristic

### 2.1. Chọn số lượng chủ đề (K) cho LDA

#### Thử nghiệm với các giá trị K khác nhau

Chúng tôi thử nghiệm với các mô hình LDA sử dụng các giá trị K (số lượng chủ đề) khác nhau: $K ∈ \set{5, 10, 15, 20, 25, 30, ..., 400}$

#### Phương thức đánh giá

1. **Định tính dựa trên cảm nhận của con người**
  Cảm thấy K nào có sự phân bổ chủ đề hợp lý hơn thì chọn K đó.

2. **Topic Coherence** ($C_v$ : đánh giá mức độ các từ trong cùng một topic xuất hiện cùng nhau)\
Mặc dù trước đó đã nói rằng phương thức này không đảm bảo tính đúng đắn nhưng chúng tôi vẫn sẽ chọn một K bằng phương thức này phòng trường hợp này phòng trường hợp nó có thể ra kết quả Hit tốt hơn cho hàm Heuristic.

#### Kết quả

1. **Định tính dựa trên cảm nhận của con người**: K = 20

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/LDA_CGS/Topic_Distribution_K20.png?raw=true" alt="Not found">
  <figcaption>Phân bổ chủ để khi số lượng chủ đề là 20</figcaption>
</figure>

2. **Topic Coherence** ($C_v$): K = 170

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/LDA_CGS/Coherence_K170.png?raw=true" alt="Not found">
  <figcaption>Điểm Coherence trên tập Validation</figcaption>
</figure>

### 2.2. Chọn hàm Heuristic để kết hợp Trie và LDA

#### Hàm Heuristic tìm được

$$\text{score}(w) = \frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)} \times \left(1 + \alpha \cdot \text{cosineˍsim}(\text{topic}(w), \text{topic}(\text{context}))^2\right)$$

Trong đó:
- $w$ : từ được gợi ý
- $\text{freq}(w)$ : tần suất từ $w$ trong tập huấn luyện
- $\text{maxˍfreq}$ : tần suất cao nhất trong vocabulary
- $\text{topic}(w)$ : phân bố chủ đề của từ $w$ (vector K chiều)
- $\text{topic}(\text{context})$ : phân bố chủ đề của ngữ cảnh hiện tại
- $\alpha$ : hyperparameter điều chỉnh trọng số của "topic relevance" (thường $\alpha \in [0.5, 2.0]$)

#### Phân tích từng thành phần

**1. Phần tần suất: $\frac{\log(\text{freq}(w) + 1)}{\log(\text{maxˍfreq} + 1)}$**

| Đặc tính | Giải thích |
| --- | --- |
| **Tại sao logarit?** | Tần suất thô có dạng power-law (một vài từ rất phổ biến) → log giúp chuẩn hóa |
| **Tại sao +1?** | Tránh $\log(0)$ khi freq=0 |
| **Phạm vi giá trị** | $[0, 1]$ → dễ kết hợp với phần khác |
| **Ý nghĩa** | Tránh từ rất phổ biến làm lấn át những từ có ý nghĩa topic khác |

**2. Phần chủ đề: $1 + \alpha \cdot \text{cosineˍsim}(...)^2$**

| Đặc tính | Giải thích |
| --- | --- |
| **Cosine similarity** | Đo góc giữa hai vector topic → [0, 1] |
| **Bình phương (^2)** | Khuếch đại sự khác biệt (amplify difference) → topic relevance rõ ràng hơn |
| **$1 +$ phía trước** | Đảm bảo từ hoàn toàn không liên quan (sim=0) vẫn có score > 0 |
| **Hyperparameter $\alpha$** | Điều chỉnh mức độ "topic sensitivity" |

#### Tại sao nhân hai phần lại với nhau?

$$\text{score}(w) = \underbrace{\frac{\log(\text{freq})}{\log(\text{maxˍfreq})}}_\text{phần A: popularity} \times \underbrace{(1 + \alpha \cdot \text{sim}^2)}_\text{phần B: relevance}$$

**Ý tưởng:**
- Từ phủ hợp ngữ cảnh nhưng hiếm → score vừa phải
- Từ phổ biến nhưng không phù hợp → score thấp
- Từ vừa phổ biến vừa phù hợp → score cao

#### Lưu ý

Việc thử nghiệm hàm Heuristic và hyperparameter α được thực hiện trên tập Validation.

### 2.3 Kết luận

- K tốt nhất: 20
- Alpha tốt nhất theo với độ dài prefix từ 1 đến 6: [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

## 3. Kết quả đánh giá tổng thể

Đánh giá tổng thể mô hình được thực hiện trên tập Test (18 000 tài liệu), với mỗi tài liệu chọn 10 token  ngẫu nhiên làm query.

<figure align="center">
  <img src="https://github.com/Quan064/Word_AutoComlete/blob/main/Analysis/Model_comp.png?raw=true" alt="Not found">
  <figcaption>Bảng so sánh tổng thể mô hình autocomplete trên tập Test (query each prefix length ~1.8 triệu)</figcaption>
</figure>

### Nhận xét

Bảng so sánh mô hình cho thấy kêt quả của Trie + LDA không có tính vượt trội rõ ràng so với Trie Freq. Mặt khác, nếu tập test có độ loãng chủ đề của ngữ cảnh lớn hơn, điểm của Trie + LDA và Trie Freq khả năng cao là sẽ hoàn toàn trùng khớp.

Dưới đây là một số nguyên nhân mà chúng tôi nghi ngờ là đang gây ra tình trạng này:

1. **Từ trong tài liệu đa số không nhiều ý nghĩa hoặc không theo chủ đề tổng**: Tần suất từ từ Trie Freq đã đủ để xác định gợi ý, thông tin chủ đề từ LDA không cung cấp thêm lợi thế đáng kể.

2. **Linearized LDA mất thông tin**: Sử dụng trung bình cộng đơn giản thay vì Collapsed Gibbs Sampling đầy đủ, dẫn tới giảm độ chính xác.

3. **Hàm Heuristic chưa tối ưu**: Phần tần suất chiếm trọng lực lớn, "lấn át" phần chủ đề. Hyperparameter $\alpha$ có thể cần điều chỉnh thêm.

4. **CGS chưa đủ mạnh để capture semantic relationships**: CGS dựa trên co-occurrence statistics, không thực sự hiểu được semantic similarity giữa các từ. Ví dụ, từ "neural" và "deep" có thể không được coi là gần nhau về mặt ngữ nghĩa vì chúng xuất hiện ở vị trí khác nhau trong corpus. Word embeddings (Word2Vec, FastText, BERT) có khả năng capture được các mối quan hệ ngữ nghĩa sâu hơn, giúp cải thiện chất lượng gợi ý.

# V. Cách chạy chương trình

# VI. Kết luận