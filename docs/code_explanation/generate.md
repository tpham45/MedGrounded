# `src/generation/generate.py`

## Mục đích

File này thực hiện phần "generation" trong RAG: lấy context liên quan (qua `retrieve()`), ghép vào một prompt có ràng buộc chặt, rồi gọi Claude để sinh câu trả lời **chỉ dựa trên context được cung cấp**. Đây là nơi định nghĩa hợp đồng "grounding" cốt lõi của cả project: LLM không được bịa, và phải thừa nhận khi không đủ bằng chứng.

## Input / Output

- **Input**: một câu hỏi (string), cộng `top_k` (số chunk context muốn lấy, mặc định 5).
- **Output của `generate_answer()`**: một dict gồm `query` (câu hỏi gốc), `retrieved_chunks` (list chunk lấy được từ `retrieve()`, đầy đủ metadata), `generated_answer` (text câu trả lời của Claude).
- **Phụ thuộc bên ngoài**: cần biến môi trường `ANTHROPIC_API_KEY` (load qua `.env` bằng `python-dotenv`) vì có gọi API Claude thật.
- Khi chạy trực tiếp (`python generate.py`): in ra câu trả lời cho 2 câu hỏi mẫu, để kiểm tra grounding hoạt động đúng.

## Giải thích từng hàm

### Code cấp module

```python
_llm = ChatAnthropic(model=LLM_MODEL, temperature=0)
```

Tạo sẵn 1 instance `ChatAnthropic` (từ `langchain_anthropic`) ở scope module, dùng chung cho mọi lời gọi `generate_answer()` — tương tự lý do load model 1 lần trong `retrieve.py`: tránh khởi tạo lại client mỗi lần gọi hàm. Biến `_llm` này còn được **import trực tiếp** bởi `evaluate_faithfulness.py` để tái sử dụng cùng một client (và cùng `PROMPT_TEMPLATE`) khi sinh câu trả lời RAG trong bước đánh giá.

`PROMPT_TEMPLATE` là một template string cố định, có 2 chỗ trống `{context}` và `{query}` được điền vào lúc gọi.

### `generate_answer(query, top_k=5)`

- `query`: câu hỏi cần trả lời.
- `top_k`: số chunk context muốn lấy từ retrieval.

Logic:
1. Gọi `retrieve(query, top_k=top_k)` để lấy các chunk liên quan nhất (xem chi tiết ở `retrieve.md`).
2. Ghép nội dung `text` của các chunk lại thành 1 khối context duy nhất, các chunk cách nhau bằng dòng trống (`"\n\n".join(...)`).
3. Điền `context` và `query` vào `PROMPT_TEMPLATE` để tạo prompt hoàn chỉnh.
4. Gọi `_llm.invoke(prompt)` — gửi prompt tới Claude, chờ phản hồi đồng bộ (blocking).
5. Trả về dict gồm câu hỏi gốc, các chunk đã retrieve (giữ lại để có thể trace/debug/dùng lại ở bước sau — ví dụ `evaluate_faithfulness.py` cần chính các chunk này để chấm faithfulness), và nội dung câu trả lời (`response.content`).

### Khối `if __name__ == "__main__"`

Chạy thử `generate_answer()` với 2 câu hỏi mẫu — 1 câu có context phù hợp trong dataset (mitochondria/plant cell death), 1 câu không có context phù hợp (exercise/diabetes) — để kiểm tra trực quan cả 2 nhánh hành vi: trả lời có căn cứ, và từ chối trả lời khi thiếu bằng chứng.

## Quyết định thiết kế và lý do

- **Model `claude-haiku-4-5`**: đây là model nhỏ/rẻ/nhanh trong họ Claude, phù hợp cho việc prototype một pipeline cần gọi LLM nhiều lần (mỗi câu hỏi ít nhất 1 lần sinh câu trả lời, và sau này còn nhân đôi vì có thêm no-RAG baseline). Với tác vụ "đọc context rồi trả lời trong phạm vi context" — không đòi hỏi reasoning phức tạp — model nhỏ đã đủ chất lượng, không cần model lớn/đắt hơn.
- **`temperature=0`**: ép model trả lời càng nhất quán/quyết định càng tốt (ít ngẫu nhiên), phù hợp cho tác vụ trả lời câu hỏi y khoa dựa trên bằng chứng — không muốn model "sáng tạo" cách diễn đạt khác nhau mỗi lần chạy, vì mục tiêu là đo faithfulness một cách ổn định, có thể lặp lại.
- **Prompt ép buộc "chỉ dùng context, nếu không đủ thì nói "insufficient evidence" thay vì đoán"**: đây là ràng buộc quan trọng nhất của cả hệ thống — nó trực tiếp encode chống lại hành vi hallucination. Việc yêu cầu một câu trả lời cố định ("insufficient evidence") thay vì để model tự diễn đạt việc "không biết" theo nhiều cách, giúp dễ phát hiện/lọc các trường hợp này sau này nếu cần (ví dụ đếm tỷ lệ câu hỏi mà context không đủ).
- **`_llm.invoke()` (đồng bộ) trong file này, thay vì async ngay từ đầu**: ở giai đoạn viết file này, mục tiêu chỉ là kiểm tra logic generate hoạt động đúng cho từng câu hỏi riêng lẻ — chưa cần tối ưu tốc độ hàng loạt. Việc chuyển sang gọi async (`ainvoke`) chỉ thực sự cần thiết ở `evaluate_faithfulness.py`, nơi phải chạy nhiều câu hỏi song song — nên file `generate.py` được giữ nguyên dạng sync để đơn giản, còn phần eval tự viết logic async riêng dựa trên cùng `_llm` và `PROMPT_TEMPLATE` (import trực tiếp từ đây thay vì viết lại).
- **Trả về cả `retrieved_chunks` trong kết quả**, không chỉ `generated_answer`: để bước sau (đặc biệt là chấm faithfulness bằng RAGAS) có ngay context đã dùng để sinh câu trả lời, không phải gọi lại `retrieve()` lần nữa — đảm bảo tính nhất quán (câu trả lời được chấm điểm đúng với context đã thực sự dùng để sinh ra nó).

## Lỗi/vấn đề đã gặp

Không gặp lỗi khi viết file này — chạy đúng ngay từ đầu, cả 2 nhánh hành vi (trả lời có căn cứ / "insufficient evidence") đều hoạt động như mong đợi khi test bằng 2 câu hỏi mẫu.
