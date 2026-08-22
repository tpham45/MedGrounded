# `src/eval/evaluate_faithfulness.py`

## Mục đích

File này là bước đánh giá cuối cùng của cả pipeline: với một tập câu hỏi cố định lấy từ dataset, sinh **cả câu trả lời RAG (có context) lẫn câu trả lời no-RAG (không có context)** cho mỗi câu, chấm điểm **faithfulness** (mức độ câu trả lời được support bởi bằng chứng) cho cả hai bằng RAGAS, rồi so sánh trung bình 2 nhóm. Mục tiêu là định lượng: RAG thực sự giúp câu trả lời bám sát bằng chứng hơn bao nhiêu so với việc không có retrieval — đây chính là con số trả lời cho câu hỏi cốt lõi của cả project MedGrounded.

## Input / Output

- **Input**: `data/raw/pubmedqa_labeled.jsonl` (dataset gốc) — lấy ngẫu nhiên 50 câu hỏi (seed=42, để chạy lại vẫn ra đúng 50 câu đó — reproducible).
- **Output**: `data/processed/eval_results.jsonl` — mỗi dòng gồm `question`, `generated_answer_rag`, `generated_answer_no_rag`, `faithfulness_score_rag`, `faithfulness_score_no_rag`.
- In ra console: điểm faithfulness trung bình của nhóm RAG, nhóm no-RAG, và chênh lệch giữa 2 nhóm.
- **Phụ thuộc bên ngoài**: `ANTHROPIC_API_KEY` (dùng 2 lần — 1 lần cho việc sinh câu trả lời qua `_llm` import từ `generate.py`, 1 lần cho LLM chấm điểm faithfulness của RAGAS, tạo riêng qua `AsyncAnthropic`).

## Giải thích từng hàm

### Đoạn stub `langchain_community.chat_models.vertexai` ở đầu file

```python
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAI: pass
_vertexai_stub.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)
```

Đây **không phải logic nghiệp vụ** — đây là một workaround cho bug tương thích thư viện (giải thích chi tiết ở mục "Lỗi đã gặp" bên dưới). Đoạn này phải nằm **trước** mọi import liên quan tới `ragas`, vì nó đăng ký một module giả vào `sys.modules` để chặn lỗi `ModuleNotFoundError` xảy ra khi `ragas` cố import module thật (không tồn tại trong phiên bản `langchain-community` đã cài).

### `load_questions(path, n, seed)`

- `path`: đường dẫn `pubmedqa_labeled.jsonl`.
- `n`: số câu hỏi muốn lấy (50).
- `seed`: seed cho random shuffle, để kết quả có thể lặp lại giống hệt giữa các lần chạy.

Đọc toàn bộ dataset, dùng `random.Random(seed).shuffle(samples)` (tạo một instance `Random` riêng với seed cố định, không đụng tới global random state — an toàn hơn `random.shuffle` trực tiếp), rồi lấy `n` câu hỏi đầu tiên sau khi xáo trộn. Dùng `Random(seed)` thay vì `random.seed(seed)` toàn cục để tránh ảnh hưởng tới các phần khác của chương trình có thể cũng dùng module `random`.

### `build_faithfulness_metric()`

Khởi tạo metric `Faithfulness` của RAGAS — cụ thể:
1. Tạo `AsyncAnthropic` client (client bất đồng bộ chính thức của Anthropic SDK — **không phải** client đồng bộ, xem lý do ở mục thiết kế).
2. `llm_factory(JUDGE_MODEL, provider="anthropic", client=client, max_tokens=4096)` — đây là API hiện tại (RAGAS 0.4.x) để tạo một LLM wrapper dùng làm "judge" (người chấm điểm) cho các metric của RAGAS.
3. `del llm.model_args["top_p"]` — xoá thủ công tham số `top_p` khỏi cấu hình mặc định của RAGAS (xem lý do ở mục "Lỗi đã gặp").
4. Trả về `Faithfulness(llm=llm)` — đối tượng metric sẵn sàng để gọi `.ascore(...)`.

### `generate_rag_answer(query, context_texts)` *(async)*

- `query`: câu hỏi.
- `context_texts`: list các đoạn text context (đã được retrieve từ trước, truyền vào chứ không tự gọi `retrieve()` bên trong — xem lý do thiết kế).

Ghép `context_texts` thành 1 khối (giống hệt logic trong `generate_answer()` của `generate.py`), điền vào `PROMPT_TEMPLATE` (import trực tiếp từ `generate.py` để đảm bảo dùng đúng 1 prompt cho cả 2 nơi, tránh lệch pha), gọi `await _llm.ainvoke(prompt)` — bản bất đồng bộ của `.invoke()` — rồi trả về `response.content`.

### `generate_no_rag_answer(query)` *(async)*

Tương tự nhưng dùng `NO_RAG_PROMPT_TEMPLATE` (không có context, chỉ hỏi thẳng) — đây là baseline: Claude trả lời bằng kiến thức nội tại của nó, không được cung cấp bất kỳ đoạn văn bản nào từ dataset.

### `process_question(question, context_texts, faithfulness, semaphore)` *(async)*

Đây là hàm xử lý **1 câu hỏi hoàn chỉnh**, chạy như một task độc lập (được gọi 50 lần song song, giới hạn bởi `semaphore`). Logic bên trong `async with semaphore:` (chỉ tối đa `MAX_CONCURRENCY` task được vào khối này cùng lúc):

1. `asyncio.gather(generate_rag_answer(...), generate_no_rag_answer(...))` — sinh **đồng thời** cả 2 câu trả lời (RAG và no-RAG) cho cùng 1 câu hỏi, vì 2 việc này độc lập với nhau, không cần chờ tuần tự.
2. `asyncio.gather(faithfulness.ascore(...), faithfulness.ascore(...))` — chấm điểm **đồng thời** cả 2 câu trả lời đó. Điểm quan trọng: **cả hai đều được chấm với cùng `context_texts`** (context mà RAG đã dùng) — kể cả câu trả lời no-RAG (vốn không hề "nhìn thấy" context này khi sinh ra). Đây là cách để đo: "nếu so với đúng tập bằng chứng mà một hệ RAG sẽ dùng, câu trả lời không dùng RAG khớp được bao nhiêu phần trăm?" — câu trả lời gần như chắc chắn là thấp, và đó chính là điều muốn chứng minh.
3. Trả về 1 dict chứa đầy đủ câu hỏi, 2 câu trả lời, 2 điểm số — sẵn sàng để ghi ra file.

### `save_results(results, path)`

Giống hệt pattern `save_as_jsonl`/`save_chunks` ở các file trước: tạo thư mục nếu chưa có, ghi mỗi kết quả ra 1 dòng JSON.

### `main()` *(async)*

Điều phối toàn bộ:
1. Lấy 50 câu hỏi (`load_questions`).
2. Khởi tạo `faithfulness` metric và `semaphore = asyncio.Semaphore(MAX_CONCURRENCY)`.
3. **Retrieve context cho cả 50 câu hỏi TRƯỚC, chạy TUẦN TỰ** bằng list comprehension thường (không async, không song song) — đây là điểm thiết kế quan trọng, giải thích chi tiết bên dưới.
4. `asyncio.gather(*(process_question(...) for ...))` — chạy 50 task `process_question` "song song" (thực chất bị giới hạn còn tối đa 5 task hoạt động cùng lúc nhờ semaphore bên trong mỗi task).
5. Lưu kết quả, tính trung bình từng nhóm, in ra chênh lệch.

## Quyết định thiết kế và lý do

### Tại sao retrieval chạy tuần tự còn generation/scoring chạy song song?

Đây là quyết định thiết kế quan trọng nhất của file này, và nó dựa trên 2 đặc điểm khác nhau hoàn toàn của 2 loại tác vụ:

- **Retrieval** (`retrieve()` — gồm `SentenceTransformer.encode()` và `FAISS.search()`) là tác vụ **CPU-bound, chạy local, không gọi mạng**. Với ~4486 vector, mỗi lần retrieve chỉ mất vài chục mili-giây — 50 lần retrieve tuần tự tốn chưa tới 2 giây tổng cộng. Chạy song song không đem lại lợi ích đáng kể về tốc độ, và (quan trọng hơn) **không an toàn**: gọi `SentenceTransformer.encode()` đồng thời từ nhiều thread (qua `asyncio.to_thread`) đã gây ra **segfault** thực tế khi thử nghiệm (xem mục lỗi bên dưới).
- **Generation và faithfulness scoring** đều là tác vụ **I/O-bound, phải chờ mạng** (gọi API Claude, mỗi lần mất vài giây tới hơn một phút tùy độ dài). Đây là nơi chạy song song thực sự có lợi: trong lúc chờ phản hồi của request này, chương trình có thể gửi request khác đi, tận dụng thời gian chờ thay vì đứng im.

Vì vậy pipeline được chia làm 2 pha rõ rệt: pha 1 (retrieval) chạy hết tuần tự trước để tránh rủi ro crash, pha 2 (mọi thứ liên quan LLM) mới bắt đầu chạy song song.

### Tại sao dùng `asyncio.Semaphore(MAX_CONCURRENCY=5)` thay vì chạy tất cả 50 câu cùng lúc?

`asyncio.gather()` nếu không có gì giới hạn sẽ cố gửi toàn bộ 50×4 = 200 request cùng lúc. Điều này rất dễ chạm rate limit của Anthropic API (dẫn tới lỗi 429 hoặc bị throttle), và cũng khó kiểm soát tài nguyên (bộ nhớ, số connection mở). `Semaphore(5)` giới hạn số câu hỏi được xử lý đồng thời xuống 5, cân bằng giữa tốc độ (vẫn nhanh hơn nhiều so với tuần tự) và độ ổn định (không dồn dập request tới mức bị giới hạn hoặc gây lỗi khó chẩn đoán).

### Tại sao mọi lời gọi LLM trong file này dùng `ainvoke()`/`ascore()` (async thuần), không dùng `asyncio.to_thread()` để bọc hàm sync?

Ban đầu, cách tiếp cận đầu tiên là tái sử dụng thẳng `generate_answer()` (hàm sync có sẵn trong `generate.py`, dùng `_llm.invoke()`) bằng cách bọc nó trong `asyncio.to_thread()`. Cách này gây ra **hiện tượng treo** (process chạy CPU 300%, mọi kết nối mạng ở trạng thái `CLOSE_WAIT` mà không tiến triển) — nguyên nhân nhiều khả năng là do gọi `.invoke()` (sync) từ một thread pool trong khi cùng lúc gọi `.ainvoke()` (async) từ event loop chính trên **cùng một instance `_llm`**, khiến connection pool nội bộ (dùng chung giữa 2 client sync/async của `ChatAnthropic`) bị race condition. Cách khắc phục: chỉ dùng `ainvoke()` (native async) ở mọi nơi trong file này, không bao giờ trộn với `.invoke()` sync trên cùng object `_llm` cùng lúc.

### Tại sao `context_texts` được truyền vào `process_question()` thay vì để mỗi task tự gọi `retrieve()`?

Đây là hệ quả trực tiếp của quyết định "retrieval phải tuần tự": nếu để mỗi `process_question()` tự gọi `retrieve()` bên trong, retrieval sẽ lại bị chạy song song (vì `process_question` chạy song song) — quay về đúng bug segfault đã gặp. Bằng cách retrieve trước, tuần tự, rồi truyền `context_texts` (đã là list string thuần, không còn phụ thuộc gì vào model/FAISS) vào từng task, đảm bảo phần chạy song song hoàn toàn không đụng tới `SentenceTransformer`/FAISS.

### Tại sao model chấm điểm (judge) cũng là `claude-haiku-4-5`, giống model sinh câu trả lời?

Đơn giản hoá: dùng cùng 1 model cho cả sinh và chấm giúp không phải quản lý thêm API key/cấu hình cho model khác, và với vai trò "judge" (đánh giá xem 1 câu có được support bởi context hay không) — một tác vụ suy luận tương đối trực tiếp — model nhỏ như haiku vẫn đủ khả năng đảm nhiệm, không cần model lớn hơn.

## Lỗi/vấn đề đã gặp và cách sửa

Đây là file gặp nhiều vấn đề kỹ thuật nhất trong cả project, vì nó phải tích hợp nhiều thư viện (RAGAS, Anthropic SDK, asyncio) đồng thời — chi tiết từng lỗi:

### 1. `ModuleNotFoundError: langchain_community.chat_models.vertexai`

`ragas==0.4.3` import cứng `from langchain_community.chat_models.vertexai import ChatVertexAI` ngay ở top-level của `ragas/llms/base.py` — dù project này không hề dùng Google Vertex AI. Module đó đã bị xoá khỏi phiên bản `langchain-community==0.4.2` đang cài (thư viện đang trong quá trình "sunset"/deprecate theo cảnh báo chính thức của nó). Đây là lỗi **không tương thích giữa 2 phiên bản thư viện**, không phải lỗi code của project.

**Cách sửa**: đăng ký một module giả (`types.ModuleType`) chứa 1 class `ChatVertexAI` rỗng vào `sys.modules` trước khi import `ragas`, để câu lệnh `import` bên trong `ragas` tìm thấy "module" này và không raise lỗi. Vì `ChatVertexAI` chỉ được dùng ở đó như 1 phần tử trong một list kiểm tra kiểu (`isinstance` check), một class rỗng là đủ để không phá vỡ logic thật của RAGAS.

### 2. API của `ragas.metrics.Faithfulness` (cách cũ) đã bị deprecate

Ban đầu định dùng `from ragas.metrics import Faithfulness` theo tài liệu/thói quen cũ (dựa trên `SingleTurnSample`, `LangchainLLMWrapper`), nhưng RAGAS 0.4.x đã tái cấu trúc hoàn toàn: API cũ vẫn tồn tại nhưng chỉ để tương thích ngược (deprecated, cảnh báo trỏ sang API mới), còn implementation thực sự đã chuyển sang `ragas.metrics.collections.Faithfulness`, dùng `llm_factory()` với một client SDK gốc (`AsyncAnthropic`) thay vì bọc qua LangChain.

**Cách sửa**: dùng đúng API mới (`ragas.metrics.collections.Faithfulness` + `ragas.llms.llm_factory`), gọi `.ascore(user_input=..., response=..., retrieved_contexts=...)` — xác định bằng cách đọc trực tiếp source code đã cài trong venv (`inspect.signature`) thay vì đoán theo tài liệu cũ, vì hành vi thực tế của phiên bản đã cài mới là nguồn sự thật đáng tin nhất.

### 3. Model `claude-haiku-4-5` từ chối request có cả `temperature` và `top_p`

`llm_factory()` mặc định luôn set cả `temperature=0.01` và `top_p=0.1` trong cấu hình gửi lên API. Model `claude-haiku-4-5` trả lỗi 400 `"temperature and top_p cannot both be specified for this model"`.

**Cách sửa**: sau khi tạo `llm`, xoá thủ công key `top_p` khỏi `llm.model_args` (`del llm.model_args["top_p"]`) — dict cấu hình nội bộ mà RAGAS dùng để build request. Đã thử truyền `top_p=None` qua `llm_factory(..., top_p=None)` trước, nhưng cách đó khiến API nhận được `top_p: null` tường minh và vẫn báo lỗi khác ("top_p: Input should be a valid number") — chỉ có việc xoá hẳn key đó khỏi dict mới thực sự loại `top_p` ra khỏi request.

### 4. `IncompleteOutputException: max_tokens length limit`

Sau khi sửa lỗi #3, gặp tiếp lỗi RAGAS không sinh đủ output vì `max_tokens` mặc định của `InstructorModelArgs` chỉ là 1024 — không đủ cho bước "statement generation" của Faithfulness (tách câu trả lời dài thành các statement rời để kiểm tra từng cái).

**Cách sửa**: truyền `max_tokens=4096` khi gọi `llm_factory(...)`.

### 5. Segfault khi chạy `retrieve()` song song qua `asyncio.to_thread`

Thiết kế song song ban đầu gọi `retrieve()` (chứa `SentenceTransformer.encode()`) đồng thời từ nhiều thread khác nhau (tối đa 5, theo semaphore). Chương trình bị **segfault** (exit code 139) không có traceback (vì crash ở tầng native code, Python exception handler không bắt được). Kèm theo cảnh báo `resource_tracker: There appear to be 1 leaked semaphore objects` — dấu hiệu cho thấy xung đột trong cơ chế song song hoá nội bộ của tokenizers/PyTorch khi bị gọi từ nhiều OS thread cùng lúc.

**Cách sửa**: tách retrieval ra khỏi phần chạy song song hoàn toàn — retrieve tất cả context cần thiết tuần tự trong `main()` trước, chỉ truyền kết quả (list string thuần) vào các task chạy song song. Xem chi tiết lý do ở mục thiết kế phía trên.

### 6. Treo (hang) khi trộn lẫn `.invoke()` sync (qua thread) và `.ainvoke()` async trên cùng client

Sau khi sửa lỗi #5 bằng cách gọi `generate_answer()` (hàm sync có sẵn) qua `asyncio.to_thread`, chương trình không crash nhưng **treo vô thời hạn**: CPU dùng gần 300% liên tục, mọi kết nối TCP ở trạng thái `CLOSE_WAIT` (server đã đóng kết nối nhưng chương trình không xử lý tiếp), không có tiến triển dù đợi hơn 14 phút cho chỉ 3 câu hỏi.

**Cách sửa**: viết lại để không bao giờ gọi `_llm.invoke()` (sync) đồng thời với `_llm.ainvoke()` (async) trên cùng 1 instance client. Thay vào đó, viết riêng `generate_rag_answer()` dùng `ainvoke()` thuần async ngay trong file này (nhận `context_texts` đã được retrieve sẵn làm tham số, thay vì gọi lại `generate_answer()` sync).

### Tóm tắt bài học rút ra

Cả 3 lỗi #5 và #6 đều xuất phát từ cùng một sai lầm gốc: cố "song song hoá" một thứ vốn không được thiết kế để dùng đa luồng an toàn (model ML nội bộ dùng thread pool riêng, hoặc 1 client HTTP có state chia sẻ). Bài học: khi thêm concurrency vào một pipeline có sẵn, phải xác định rõ **từng phần** là CPU-bound hay I/O-bound, và chỉ song song hoá phần I/O-bound (gọi mạng) bằng cơ chế native của asyncio (`ainvoke`, `ascore`) — không dùng `to_thread` để "ép" song song cho những thứ vốn đã nhanh và có thể không thread-safe.
