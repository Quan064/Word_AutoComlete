﻿# TOPIC-AWARE AUTOCOMPLETE

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


### Độ phức tạp

Liệt kê các gợi ý:

$$
O(L+M)
$$

Trong đó:

- $L$ : độ dài prefix
- $M$ : số lượng ký tự có trong các nhánh con bên dưới nút tiền tố cuối cùng

→ Rất nhanh, phù hợp realtime


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

$$\text{score}(\text{word.freq, word.topic, context.topic}) = \frac{\log(\text{word.freq})}{\log(\text{maxˍfreq})} \cdot \left(1+\alpha\cdot\text{cosineˍsim}(\text{word.topic}, \text{context.topic})^2\right)$$

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

- Tính chỉ số **Topic Coherence** $C_v$
  (đánh giá mức độ các từ trong cùng một topic xuất hiện cùng nhau)

Các chỉ số này phản ánh khả năng mô hình giải thích dữ liệu mới.

→ So sánh các cấu hình và chọn mô hình tốt nhất.

### 2.5. Đánh giá tổng quát

Sử dụng **18,000 tài liệu kiểm thử** (hoàn toàn độc lập):

- So sánh:
  - Mô hình kết hợp **Trie Freq + LDA** (cấu hình tốt nhất)
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

## 2. Lựa chọn siêu tham số và hàm Heuristic

## 3. Kết quả đánh giá tổng thể


# V. Ứng dụng

# VI. Kết luận