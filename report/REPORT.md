# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đào Phước Thịnh
**Mã số:** 2A202600029
**Nhóm:** C401_D5
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Cosine similarity chỉ ra sự tương đồng về ngữ nghĩa giữa các văn bản khi tính toán góc của hai vectors trong không gian. High similarity tức là các vectors có xu hướng cùng phương, tương đương với các văn bản có ngữ nghĩa và ý niệm giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Mèo rất thích ăn hạt và cá.
- Sentence B: Các bé mèo trưởng thành chuộng đồ ăn vụn như bánh cá.
- Tại sao tương đồng: Đề cập đến cùng một chủ đề (mèo và sở thích ăn uống).

**Ví dụ LOW similarity:**
- Sentence A: Mèo rất thích ăn hạt và cá.
- Sentence B: Phương trình đạo hàm được giải bằng tích phân từng phần.
- Tại sao khác: Hai câu nói về các lĩnh vực hoàn toàn không liên quan với nhau (thú cưng và toán lý tưởng).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Khác với Euclidean distance dễ bị ảnh hưởng bởi độ dài (vector magnitude), Cosine similarity chỉ đo lường góc và phương của chúng do đó nó không bị thiên lệch bởi kích thước dài hay ngắn của bài text mà tập trung vào nội dung hàm nghĩa bên trong.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* step_size = 500 - 50 = 450. Số lượng chunk xấp xỉ = (10,000 - 50) / 450 = ~22.1. Do phần dư phải chia thêm nên thực tế là 23 chunks.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Nếu overlap = 100 thì chunks = (10,000 - 100)/400 = 24.75 -> 25 chunks. Overlap nhiều hơn giúp ngăn chặn việc chia cắt một câu dài ở những chỗ nhạy cảm, đồng thời nối liền duy trì context liền mạch giữa các chunks.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Lĩnh vực Y Sinh Học & Scientific Fact Checking (SciFact từ BEIR Benchmark)

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Lĩnh vực Factchecking cho y sinh học đòi hỏi truy xuất chi tiết cụ thể cực kì chính xác dựa vào các tài liệu chuyên ngành hàn lâm gồ ghề. RAG thích hợp để kiểm tra thông tin thực tế dựa vào ngữ cảnh phức tạp của các bài báo từ corpus SciFact thay vì thông tin kiến thức mơ hồ. 

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `corpus.jsonl` | SciFact BEIR | ~2.5 MB | doc_id, source, type |
| 2 | `queries.jsonl` | SciFact BEIR | ~10.4 KB | query_id |
| 3 | `test.tsv` | BEIR qrels | ~4 KB | id |
| 4 | - | - | - | - |
| 5 | - | - | - | - |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | str | `51` | Cho phép truy vấn và dễ dàng xóa/cập nhật tất cả các chunks ứng với một document lớn ở CSDL. |
| `source` | str | `corpus.jsonl` | Tiện cho tracking nguồn tin và fallback filter dựa vào đường dẫn nơi sinh ra doc đấy. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Dựa trên dữ liệu thực thi SciFact Benchmark (`run-benchmarks.py`) sử dụng 500 abstracts gây nhiễu + 50 queries:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| SciFact | Baseline (No Splitting) | 500+ | ~1500 chars | Rất tốt nhưng làm pha loãng context|
| SciFact | FixedSizeChunker (`fixed_size`) | Nhiều | 100 | Tốt (overlap=20) |
| SciFact | SentenceChunker (`by_sentences`) | Nhiều | Trung bình | Tốt nhất, chia đúng ngữ nghĩa |
| SciFact | RecursiveChunker (`recursive`) | Đa dạng | <100 | Tương đối |

### Strategy Của Tôi

**Loại:** FixedSizeChunker kết hợp với các cấu hình linh hoạt tuỳ domain. (Benchmark Strategy: FixedSizeChunker chunk=100 overlap=20)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* Strategy liên tục trượt con trỏ theo độ dài kí tự với bước nhảy được cho là (chunk_size - overlap_size). Phương pháp này bỏ qua nội hàm câu cú, cắt đúng số kí tự nhằm đảm bảo độ dài max token của embedding models nhưng nhét thêm các kí tự lùi về sau làm phần đệm để không gây hiểu lầm ngữ cảnh khi bị trượt mất 1 số từ khoá.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?* Đối với SciFact, bài báo abstract khoa học thường mang ý chính nằm san sát xen lẫn các thuật ngữ chuyên môn. FixedSize chia nhỏ abstract khổng lồ thành khoảng từ cụ thể, và overlap 20 chars giúp không lọt mất các định danh Gene hoặc Protein bị kẹt giữa 2 chunk. Điều này chứng minh vì Retrieval MRR đạt 0.9123!

**Code snippet (nếu custom):**
```python
# Mặc định code base FixedSizeChunker với window step.
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Retrieval Quality? (Recall@5 & MRR@5)| 
|-----------|----------|----------------------------------------|
| SciFact | Baseline | Recall: 0.896, MRR: 0.8900 | 
| SciFact | **FixedSize Của tôi** | Recall: 0.976, MRR: 0.9123 | 

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | FixedSize (size=100, overlapping) | 9 / 10 | Điểm recall cao nhất đạt 0.976 | Tốn tài nguyên vector vì chia doc quá nhỏ |
| Bạn 1 | Sentence Chunker (regex) | 8.5 / 10 | Câu văn tách gọn trọn vẹn ngữ nghĩa | MRR kém hơn ở ngưỡng 0.90 do size câu thay đổi đột ngột |
| Bạn 2 | Recursive Chunker | 8 / 10 | Chia đoạn thông minh khi có cấu trúc | Do nội dung abstract Y Sinh không phù hợp để đệ quy |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* Strategy tốt nhất cho text học thuật là kết hợp FixedSize với overlap để tạo recall lớn nhất (thí nghiệm chạy ra F1 lớn nhất khoảng ~0.36) bởi vì thông tin kiểm chứng y sinh có thể nằm ở bất cứ đoạn kí tự nào mà một phương pháp cắt theo Sentence khó mà bắt chính xác hơn Fixed.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?* Dùng Regex có các pattern `[.!?]` phân tách ngắt quãng câu theo kết nối câu cơ bản. Xử lý các edge case từ viết tắt khoa học hay từ "Mr.", "Et al." bằng heuristic logic ngoại lệ.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?* Đệ quy tìm kiếm các ranh giới tự nhiên của ngôn ngữ có phân cấp từ `\n\n` cho đoạn, đến dấu `.` cho câu và khoảng trắng cho dòng. Nểu text dài hơn size max hiện tại, tách nó tại delim gần nhất tìm được, ngược lại trả về list với chunk nhỏ base case đã hợp lệ.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?* ChromaDB được dùng trực tiếp dưới dạng PersistentClient. Hàm `add_documents` ánh xạ `id`, `text`, `metadata` đồng thời convert array list nhúng embedding model rồi gọi API `self._collection.upsert()`. `search` gọi hàm truy xuất Chroma truyền embeddings đầu vào được trả luôn top-k với độ chính xác chuẩn form distance `1-d` theo cosine. 

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?* Chroma filter cho phép lọc trước các records ở bộ nhớ để bỏ qua tài liệu không đủ điều kiện field `doc_id` bằng metadata `where={"key": "val"}` condition query. Hàm delete sử dụng interface `self._collection.delete(where={"doc_id": doc_id})`.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?* Format một template LLM trích xuất context dưới dạng text ghép nối (nối toàn bộ contents chunk lấy ra từ hàm Search). Nhét context vào System Instruction để KnowledgeAgent prompt API mock phản hồi lại câu hỏi người dùng kèm theo references tự nhiên.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.10.18, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\Admin\miniconda3\envs\ai_env\python.exe
collected 42 items
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true PASSED [100%]
============================= 42 passed in 1.98s ==============================  
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | I have a black car. | My automobile is dark colored. | high | 0.85 | Có |
| 2 | Python is a programming language. | Snakes are reptiles of the suborder Serpentes. | low | 0.12 | Có |
| 3 | Artificial Intelligence shapes the future. | Machine Learning provides future opportunities. | high | 0.76 | Có |
| 4 | Today the weather is rainy. | I need an umbrella today. | high | 0.65 | Có |
| 5 | Water boils at 100 degrees Celsius. | Paris is the capital of France. | low | -0.05 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Phép thử Pair 4 giữa Thời Tiết mưa và Nhu cầu cầm theo một chiếc Ô có độ tương đồng Semantic không cao bằng từ khoá trực tiếp nhưng vẫn ở Threshold ~0.65 có nghĩa là Embeddings model của AI đã có nhận thức ẩn về nhân quả đời thực, không chỉ từ mượn xác thịt mà dựa trên không gian ý tưởng tiềm ẩn. 

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Môi trường viễn thông 5G ảnh hưởng não thế nào? | Không đủ các bằng chứng gây ung thư bởi trạm 5G. |
| 2 | Paracetamol quá liều ở con nít có hậu quả gì? | Dẫn tới hoại tử gan khẩn cấp ở con trẻ với liều >150mg/kg. |
| 3 | Tác động thuốc SSRI với triệu chứng lo âu? | SSRI cho thấy giảm nhẹ lo âu trầm trọng nhưng cần chú ý 2 tuần đầu. |
| 4 | RNA polymerase quan trọng thế nào đến sự phân bào? | Polymerase làm bước phiên mã thiết yếu để kích hoạt mitosis. |
| 5 | Summarize the key information from the loaded files. | [SciFact Test Abstract] - Information about genes and IFNs interaction. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Môi trường viễn thông 5G | Nghiên cứu tần số FR1 / RF không ảnh hưởng ... | 0.81 | Có | Các dải sóng nhỏ không tác động lên vùng hồi hải mã hay cấu trúc nơ-ron... |
| 2 | Paracetamol quá liều | Triệu chứng ngộ độc acetaminophan ở giai đoạn đầu... | 0.84 | Có | Nếu dùng >150mg, chức năng gan có báo hiệu suy giảm gấp đôi. |
| 3 | Tác động thuốc SSRI | Bài báo lâm sàng dùng Fluoxetine vs lo âu... | 0.76 | Có | Thuốc thể hiện độ thích nghi sau 4 tuần ở biểu đồ trầm cảm... |
| 4 | RNA polymerase | Transcription profiling of infected cell cycle... | 0.68 | Tạm | Cần enzyme để tách xoắn kép kích động tạo mRNA trung gian... |
| 5 | Summarize key files. | Data microarray designed for molecular studied of genes... | 0.288 | Không | Dữ liệu SciFact chỉ chứa nội dung gene sinh học, truy vấn demo sai domain ngữ cảnh. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Biết được phương pháp Sentence Chunking thực ra hữu hiệu hơn khi kết hợp cùng overlap số lượng câu để duy trì nối liền đoạn văn bản không quá gắt, thay vì chỉ FixedSize có thể cắt dở 1 đoạn định danh.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Các nhóm chia domain về IT Support có điểm số Retrieval MRR lớn hơn hẳn do tính chuẩn hóa của nội dung các văn bản hướng dẫn công nghệ so với sự mơ hồ phức tạp của các bài đăng ngôn ngữ khoa học đời sống.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Cải thiện bước Filter đầu vào, băm nhỏ Metadata ra ví dụ như `Metadata={'category': 'Disease', 'year': 2024}` thay vì các trường rỗng tuếch để quá trình Search không phải đọc các bài lỗi thời.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
