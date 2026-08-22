# `src/retrieval/load_data.py`

## Mục đích

File này tải dataset PubMedQA (bản `pqa_labeled` — có nhãn ground truth yes/no/maybe) từ HuggingFace Hub về máy, và lưu lại thành file `.jsonl` ở local để các bước sau (chunking, embedding, v.v.) không cần tải lại mỗi lần chạy.

## Input / Output

- **Input**: không có input từ người dùng. Dataset được tải trực tiếp từ HuggingFace bằng ID `qiaojin/PubMedQA`, config `pqa_labeled`, split `train` (1000 mẫu).
- **Output**: file `data/raw/pubmedqa_labeled.jsonl` — mỗi dòng là một JSON object tương ứng 1 mẫu, với các field gốc từ dataset: `pubid`, `question`, `context` (dict chứa `contexts` là list đoạn văn, cùng `labels`, `meshes`...), `long_answer`, `final_decision` (yes/no/maybe).
- Ngoài ra in ra console: tổng số mẫu và 1 ví dụ (question/context/answer) để kiểm tra nhanh dữ liệu load đúng, không cần mở file lên xem.

## Giải thích từng hàm

### `load_pubmedqa_labeled()`

Không nhận tham số. Gọi `datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")` và trả về đối tượng `Dataset` của HuggingFace. Đây chỉ là một wrapper mỏng quanh `load_dataset` — tách thành hàm riêng để dễ mock/test hoặc đổi nguồn dataset sau này mà không phải sửa `main()`.

### `save_as_jsonl(dataset, output_path)`

- `dataset`: đối tượng `Dataset` trả về từ `load_pubmedqa_labeled()`.
- `output_path`: đường dẫn `Path` tới file `.jsonl` muốn ghi ra.

Logic: tạo thư mục cha nếu chưa tồn tại (`mkdir(parents=True, exist_ok=True)`), sau đó lặp qua từng `row` trong dataset, convert row đó (vốn là dict) sang JSON string bằng `json.dumps(row, ensure_ascii=False)` rồi ghi ra file, mỗi row một dòng. `ensure_ascii=False` để giữ nguyên ký tự Unicode (nếu có) thay vì escape thành `\uXXXX`, giúp file dễ đọc hơn khi mở trực tiếp.

### `main()`

Hàm điều phối chính, chạy tuần tự:
1. Gọi `load_pubmedqa_labeled()` để tải dataset.
2. In `len(dataset)` — số lượng mẫu — để xác nhận tải đủ 1000 mẫu.
3. Lấy `dataset[0]` làm ví dụ mẫu, in ra `question`, `context`, `final_decision` — đây là bước sanity-check thủ công, giúp người chạy nhìn ngay dữ liệu có đúng cấu trúc mong đợi không (ví dụ: `context` phải là dict có key `contexts` là list các đoạn văn, không phải string đơn).
4. Gọi `save_as_jsonl()` để ghi ra `data/raw/pubmedqa_labeled.jsonl`.
5. In đường dẫn file đã lưu.

## Quyết định thiết kế và lý do

- **Dùng config `pqa_labeled` thay vì `pqa_artificial` hoặc `pqa_unlabeled`**: `pqa_labeled` là bản có nhãn `final_decision` (yes/no/maybe) do con người gán, tức có ground truth thật. Việc này quan trọng vì mục tiêu cuối của project là đánh giá faithfulness — cần có câu trả lời "đúng" để có cơ sở so sánh, dù các bước eval hiện tại (RAGAS faithfulness) chưa trực tiếp dùng `final_decision`, nhưng để mở khả năng mở rộng đánh giá correctness sau này thì bản có nhãn là lựa chọn đúng ngay từ đầu.
- **Lưu ra `.jsonl` thay vì giữ nguyên dataset HuggingFace hoặc dùng `.csv`/`.parquet`**: `.jsonl` (JSON Lines) cho phép đọc từng dòng một cách streaming, dễ debug bằng mắt thường (`cat file.jsonl | head`), và tương thích tự nhiên với cấu trúc lồng nhau của field `context` (dict chứa list) — CSV sẽ phải serialize field này thành string phức tạp, mất tính trực quan.
- **`OUTPUT_PATH` được tính bằng `Path(__file__).resolve().parents[2]`**: đảm bảo đường dẫn ra `data/raw/` luôn đúng bất kể người dùng chạy script từ thư mục nào (root project hay từ trong `src/retrieval/`), thay vì dùng đường dẫn tương đối dễ vỡ khi đổi working directory.

## Lỗi/vấn đề đã gặp

Không có lỗi đáng kể khi viết file này — chạy đúng ngay ở lần đầu (1000 mẫu tải về, ví dụ in ra đúng cấu trúc mong đợi). Điểm cần lưu ý duy nhất: khi chạy không có `HF_TOKEN`, HuggingFace Hub in ra warning "unauthenticated requests" — không phải lỗi, chỉ là cảnh báo về rate limit thấp hơn, không ảnh hưởng tới việc tải dataset 1000 mẫu (khá nhỏ).
