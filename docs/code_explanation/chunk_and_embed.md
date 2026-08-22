# `src/retrieval/chunk_and_embed.py`

## Mục đích

File này biến dataset thô (`pubmedqa_labeled.jsonl`) thành một **vector index có thể tìm kiếm được**: cắt nhỏ (chunk) từng đoạn context trong dataset, tạo embedding cho từng chunk bằng sentence-transformers, rồi build một FAISS index từ các embedding đó. Đây là bước chuẩn bị dữ liệu (offline, chạy một lần) cho retrieval ở các bước sau.

## Input / Output

- **Input**: `data/raw/pubmedqa_labeled.jsonl` (tạo bởi `load_data.py`) — mỗi dòng có `pubid`, `question`, `context.contexts` (list đoạn văn), `final_decision`.
- **Output**: 3 file trong `data/processed/`:
  - `chunks.jsonl`: mỗi dòng là 1 chunk văn bản kèm metadata (`text`, `sample_id`, `question`, `answer`).
  - `embeddings.npy`: ma trận NumPy shape `(số_chunk, 384)`, mỗi hàng là vector embedding của chunk tương ứng theo đúng thứ tự trong `chunks.jsonl`.
  - `faiss_index/index.faiss`: FAISS index đã build sẵn từ các embedding, dùng để tìm kiếm ngữ nghĩa (semantic search) nhanh.
- Ngoài ra in ra console: số lượng chunk, shape của embedding matrix, và tổng số vector trong FAISS index (để xác nhận số vector khớp số chunk).

## Giải thích từng hàm

### `load_samples(path)`

- `path`: đường dẫn tới `pubmedqa_labeled.jsonl`.

Đọc toàn bộ file, parse từng dòng JSON, trả về list các dict (mỗi dict là 1 mẫu gốc trong dataset). Đơn giản là bước load ngược lại những gì `load_data.py` đã ghi ra.

### `chunk_samples(samples)`

- `samples`: list các mẫu (dict) từ `load_samples()`.

Đây là hàm cốt lõi của việc chunking. Với mỗi `sample`:
1. Lấy `sample_id` (dùng `pubid` nếu có, nếu không thì dùng `idx` — chỉ số thứ tự trong list — làm fallback).
2. Lấy `question` và `answer` (từ field `final_decision`) — đây là metadata sẽ được gắn vào **mọi chunk** sinh ra từ mẫu này.
3. Lấy `contexts` — list các đoạn văn (một câu hỏi PubMedQA thường có nhiều đoạn context, ví dụ BACKGROUND, RESULTS, v.v.).
4. Với mỗi đoạn context, dùng `RecursiveCharacterTextSplitter` (từ `langchain_text_splitters`) để cắt thành các chunk nhỏ hơn (`chunk_size=500`, `chunk_overlap=50`).
5. Mỗi chunk được đóng gói thành 1 dict: `{"text": ..., "sample_id": ..., "question": ..., "answer": ...}` rồi append vào list `chunks` kết quả.

Kết quả: 1000 mẫu ban đầu → 4486 chunk (số liệu thực tế khi chạy), vì mỗi mẫu có nhiều đoạn context và mỗi đoạn dài có thể bị cắt thành nhiều chunk.

### `embed_chunks(chunks, model_name=EMBEDDING_MODEL)`

- `chunks`: list chunk từ `chunk_samples()`.
- `model_name`: tên model sentence-transformers, mặc định `"all-MiniLM-L6-v2"`.

Load model `SentenceTransformer(model_name)`, trích riêng field `"text"` của từng chunk ra một list string, rồi gọi `model.encode(texts, show_progress_bar=True, convert_to_numpy=True)` để tạo embedding hàng loạt (batch). Trả về một NumPy array shape `(n_chunks, 384)` — 384 là số chiều embedding của model `all-MiniLM-L6-v2`.

### `save_chunks(chunks, path)`

Tương tự `save_as_jsonl` bên `load_data.py`: tạo thư mục cha nếu chưa có, ghi từng chunk ra một dòng JSON trong file `.jsonl`. Thứ tự dòng trong file này **phải khớp chính xác** với thứ tự hàng trong `embeddings.npy` và thứ tự vector trong FAISS index — vì ở bước retrieval, chỉ số (index) trả về từ FAISS sẽ được dùng để tra ngược lại đúng dòng trong file này.

### `build_faiss_index(embeddings)`

- `embeddings`: NumPy array từ `embed_chunks()`.

Lấy `dimension = embeddings.shape[1]` (384), tạo `faiss.IndexFlatL2(dimension)`, gọi `index.add(embeddings)` để nạp toàn bộ vector vào index, rồi trả về index đó.

### `main()`

Điều phối toàn bộ pipeline theo thứ tự: load samples → chunk → embed → lưu chunks + embeddings ra file → build FAISS index → lưu index ra file, kèm in log ở mỗi bước để theo dõi tiến trình.

## Quyết định thiết kế và lý do

- **`IndexFlatL2` thay vì các loại FAISS index khác (ví dụ `IndexIVFFlat`, `IndexHNSWFlat`)**: `IndexFlatL2` làm **brute-force exact search** (so khớp tuyến tính với mọi vector, không xấp xỉ). Với quy mô dữ liệu ở đây (~4486 vector, 384 chiều — rất nhỏ so với các ứng dụng production), brute-force vẫn đủ nhanh (mili-giây mỗi query) mà lại cho kết quả **chính xác tuyệt đối** (không có sai số approximate như HNSW/IVF). Các loại index xấp xỉ (approximate) chỉ thực sự cần thiết khi có hàng triệu vector trở lên — ở quy mô prototype này, đánh đổi tốc độ lấy độ chính xác gần như không có nghĩa lý, nên chọn `IndexFlatL2` là lựa chọn đơn giản và an toàn nhất.
- **`chunk_size=500, chunk_overlap=50`**: 500 ký tự tương đương khoảng 1-2 câu văn khoa học — đủ nhỏ để mỗi chunk mang một ý tương đối trọn vẹn (không quá loãng thông tin khi embed), nhưng đủ lớn để không cắt vụn câu giữa chừng. `overlap=50` giúp giảm rủi ro mất ngữ cảnh ở ranh giới giữa 2 chunk liền kề (một câu bị cắt đôi vẫn còn xuất hiện đủ trong ít nhất 1 trong 2 chunk).
- **`RecursiveCharacterTextSplitter` thay vì cắt theo số ký tự cố định (fixed-size splitting)**: splitter này ưu tiên cắt tại ranh giới tự nhiên (đoạn văn → câu → từ) trước khi buộc phải cắt cứng giữa chừng, giúp chunk giữ được ngữ nghĩa trọn vẹn hơn so với cắt mù theo index ký tự.
- **Metadata (`question`, `answer`, `sample_id`) được gắn vào MỌI chunk** thay vì lưu riêng ở một bảng map: đơn giản hoá việc tra cứu ở bước sau — khi retrieve được 1 chunk, có ngay đầy đủ thông tin nó thuộc câu hỏi nào mà không cần join thêm với bảng khác. Đánh đổi là dữ liệu bị lặp lại (question/answer lặp ở mọi chunk cùng 1 sample), nhưng với quy mô dữ liệu nhỏ ở đây, sự đơn giản trong code quan trọng hơn việc tối ưu dung lượng lưu trữ.
- **`pubid` làm `sample_id`, fallback về `idx`**: `pubid` là ID gốc từ PubMed, đáng tin cậy hơn để tra cứu chéo về sau (ví dụ nếu cần đối chiếu lại với nguồn gốc bài báo); `idx` chỉ là phương án dự phòng nếu vì lý do nào đó dataset thiếu field này.

## Lỗi/vấn đề đã gặp

Không gặp lỗi runtime khi viết file này. Chạy thành công ngay từ lần đầu: 1000 mẫu → 4486 chunk → embedding shape `(4486, 384)` → FAISS index với đúng 4486 vector, khớp số chunk.
