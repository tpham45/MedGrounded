# `src/retrieval/retrieve.py`

## Mục đích

File này cung cấp hàm `retrieve()` — điểm vào duy nhất để "tìm các đoạn văn bản liên quan nhất tới một câu hỏi" từ dữ liệu đã được chunk + index ở bước trước (`chunk_and_embed.py`). Đây là hàm được các module downstream (`generation/generate.py`, `eval/evaluate_faithfulness.py`) import và gọi trực tiếp — nó là "trái tim" của phần retrieval trong pipeline RAG.

## Input / Output

- **Input khi import module**: file `data/processed/faiss_index/index.faiss` (FAISS index đã build) và `data/processed/chunks.jsonl` (text + metadata của từng chunk) — cả hai được load một lần vào bộ nhớ ngay khi module được import, không phải mỗi lần gọi hàm.
- **Input của hàm `retrieve(query, top_k)`**: một câu hỏi dạng string, và số lượng kết quả muốn lấy (`top_k`, mặc định 5).
- **Output của `retrieve()`**: một list gồm `top_k` dict, mỗi dict có `text` (nội dung chunk), `question` (câu hỏi gốc trong dataset mà chunk này thuộc về), `answer` (nhãn yes/no/maybe gốc), `score` (khoảng cách L2 — càng nhỏ càng giống).
- Khi chạy trực tiếp (`python retrieve.py`): in ra console top-5 kết quả cho 3 câu hỏi mẫu, để kiểm tra retrieval có hợp lý không.

## Giải thích từng hàm

### Code cấp module (chạy khi import)

```python
_index = faiss.read_index(str(FAISS_INDEX_PATH))
_model = SentenceTransformer(EMBEDDING_MODEL)
with CHUNKS_PATH.open(...) as f:
    _chunks = [json.loads(line) for line in f]
```

Ba biến `_index`, `_model`, `_chunks` được load **một lần duy nhất** ngay khi file này được import (không nằm trong hàm nào). Lý do: load FAISS index và đặc biệt là load model `SentenceTransformer` (nặng, tốn vài trăm ms tới vài giây) là thao tác tốn thời gian — nếu load lại mỗi lần gọi `retrieve()` thì sẽ rất chậm khi gọi hàm này nhiều lần (ví dụ 50 lần trong `evaluate_faithfulness.py`). Load một lần ở module scope giúp mọi lời gọi `retrieve()` sau đó chỉ tốn thời gian cho phần tính toán thực sự (embed câu hỏi + tìm kiếm), không tốn thời gian load lại tài nguyên.

`_chunks` được load thành một **list Python theo đúng thứ tự** ghi trong `chunks.jsonl` — thứ tự này bắt buộc phải khớp với thứ tự vector trong FAISS index, vì FAISS chỉ trả về **chỉ số (index)** chứ không trả về nội dung — `_chunks[idx]` chính là cách duy nhất để tra ngược từ chỉ số đó ra text + metadata gốc.

### `retrieve(query, top_k=5)`

- `query`: câu hỏi cần tìm context liên quan (string).
- `top_k`: số lượng chunk gần nhất muốn lấy về.

Logic:
1. `_model.encode([query], convert_to_numpy=True)` — chuyển câu hỏi thành 1 vector embedding 384 chiều (dùng đúng model `all-MiniLM-L6-v2` như lúc build index, để embedding nằm cùng không gian vector).
2. `_index.search(query_embedding, top_k)` — FAISS trả về 2 mảng: `distances` (khoảng cách L2 từ query tới từng kết quả) và `indices` (vị trí của từng kết quả trong index, tương ứng vị trí trong `_chunks`).
3. Với mỗi cặp `(score, idx)`, tra `_chunks[idx]` để lấy lại `text`, `question`, `answer` gốc, đóng gói cùng `score` thành 1 dict kết quả.
4. Trả về list các dict này, đã được FAISS sắp xếp sẵn theo độ liên quan giảm dần (khoảng cách tăng dần).

### Khối `if __name__ == "__main__"`

Chạy thử `retrieve()` với 3 câu hỏi mẫu thuộc các chủ đề khác nhau (mitochondria/plant cell death, exercise/diabetes, aspirin/heart attacks), in ra top-5 kết quả mỗi câu (rút gọn text còn ~150 ký tự để dễ đọc). Đây là bước kiểm tra thủ công (sanity check) — không phải unit test tự động — giúp người đọc/người chạy nhìn trực quan xem retrieval có trả về nội dung hợp lý về mặt ngữ nghĩa hay không.

## Quyết định thiết kế và lý do

- **Load model/index/chunks ở module scope (biến global có tiền tố `_`) thay vì trong hàm hoặc trong một class**: đơn giản hoá việc sử dụng — chỗ khác chỉ cần `from retrieval.retrieve import retrieve` rồi gọi thẳng, không cần khởi tạo object hay quản lý lifecycle. Tiền tố `_` báo hiệu đây là chi tiết nội bộ, không nên bị import/sửa từ ngoài.
- **Dùng đúng `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` giống hệt `chunk_and_embed.py`**: bắt buộc phải khớp model giữa lúc build index và lúc query, vì embedding của 2 model khác nhau nằm ở không gian vector khác nhau hoàn toàn — dùng sai model sẽ cho ra kết quả tìm kiếm vô nghĩa dù code chạy không lỗi.
- **`score` là khoảng cách L2 (càng nhỏ càng tốt), không phải cosine similarity (càng lớn càng tốt)**: vì `IndexFlatL2` (chọn ở bước build index) đo bằng khoảng cách Euclidean. Người đọc code ở các file khác cần nhớ điều này khi diễn giải giá trị `score` — số thấp nghĩa là gần/giống, không phải ngược lại.
- **Không cache kết quả `retrieve()` theo từng query**: với quy mô hiện tại (encode 1 câu hỏi ngắn + search trong ~4486 vector) thao tác này chỉ mất vài chục mili-giây, nên không cần thêm cơ chế cache — giữ code đơn giản.

## Lỗi/vấn đề đã gặp

Không có lỗi khi viết file này ban đầu. Tuy nhiên, ở bước sau (`evaluate_faithfulness.py`) khi cố tình chạy `retrieve()` **song song trên nhiều thread** (qua `asyncio.to_thread`) để tăng tốc, chương trình bị **segfault**. Nguyên nhân: `SentenceTransformer.encode()` (dùng PyTorch + HuggingFace tokenizers bên dưới) không an toàn khi bị gọi đồng thời từ nhiều OS thread khác nhau trong cùng một process — thư viện tokenizers có cơ chế song song hoá nội tại (Rust thread pool) dễ xung đột khi bị gọi lồng từ Python threading. Cách khắc phục: **không** chạy `retrieve()` song song; ở `evaluate_faithfulness.py`, mọi lời gọi `retrieve()` được thực hiện **tuần tự** (vòng for thông thường) trước khi bước vào phần xử lý song song (gọi LLM). Vì retrieval là tác vụ local/CPU, không gọi mạng, nên chạy tuần tự vẫn rất nhanh (không phải bottleneck), khác với các lời gọi LLM (I/O-bound, chờ mạng) mới thực sự cần chạy song song để tiết kiệm thời gian.
