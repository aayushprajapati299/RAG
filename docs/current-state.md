# Current State of the RAG Application

This document outlines the current architecture, implementation details, operational risks, and key behaviors preserved across refactoring of the codebase as of November 2026.

---

## 1. Current Stack and How the App Starts

### Tech Stack
- **Language**: Python 3
- **Web Framework**: Flask (`flask`, `werkzeug`)
- **Template Engine**: Jinja2 (`templates/index.html` with vanilla JavaScript and CSS)
- **Document Processing**:
  - `PyMuPDF` (`fitz`): PDF text extraction and page rendering.
  - `Pillow` (`PIL`): Image handling for rasterized PDF pages.
  - `pytesseract`: OCR engine wrapper for image-to-text extraction.
  - Native Python file I/O for `.txt` documents.
- **Search & Retrieval**: Pure-Python BM25 lexical ranking implementation (`math`, `re`). No vector database or external search engine is used.
- **Generation / LLM Client**: `requests` sending HTTP requests directly to Groq's OpenAI-compatible chat completions API (`https://api.groq.com/openai/v1/chat/completions`).
- **Configuration & Environment**: `dotenv` (`load_dotenv()`) for loading `.env` variables (e.g., `GROQ_API_KEY`).
- **Dependencies in `requirements.txt`**:
  - `flask`, `pypdf` (unused in code), `requests`, `pymupdf`, `pytesseract`, `pillow`, `werkzeug`, `gunicorn` (unused in startup scripts), `dotenv` (note: PyPI package is `dotenv` rather than `python-dotenv`).

### How the App Starts
- **Local Execution**:
  - Running `python app.py` executes the block `if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=True)`.
  - Runs on port 5000 bound to all network interfaces with Flask debug mode enabled.
- **Deployment Execution (`render.yaml`)**:
  - Declares a web service named `vectorless-rag` in Python environment.
  - Build command: `pip install -r requirements.txt`.
  - Start command: `python app.py` (starts the built-in Flask development server directly, rather than using `gunicorn`).

---

## 2. Existing Routes and Request/Response Behavior

### `GET /`
- **Purpose**: Serves the single-page document Q&A interface.
- **Handler**: `index()` in `app.py`.
- **Response**: Returns rendered `templates/index.html` (HTTP 200).

### `POST /upload`
- **Purpose**: Uploads and indexes a document file.
- **Handler**: `upload_file()` in `app.py`.
- **Input**: `multipart/form-data` with form field `file`.
- **Validation**:
  - Returns HTTP 400 `{"error": "No file part in the request"}` if `'file'` is absent from `request.files`.
  - Returns HTTP 400 `{"error": "No file selected for uploading"}` if `file.filename` is empty.
- **Processing**:
  - Sanitizes filename via `secure_filename(file.filename)`.
  - Saves file to `uploads/<filename>`.
  - Invokes `index_document(filepath)`.
- **Response**:
  - Success: HTTP 200 `{"status": "ok", "pages": <int>}`.
  - Error: HTTP 500 `{"error": "Failed to index document: <exception_message>"}`.

### `POST /ask`
- **Purpose**: Answers user queries using the indexed document and ongoing conversation context, or clears history.
- **Handler**: `ask_question()` in `app.py`.
- **Input**: JSON payload `{"question": "<str>", "reset": <bool, optional>}`.
- **Validation**:
  - Returns HTTP 400 `{"error": "Missing 'question' field"}` if body is missing or lacks `'question'`.
- **Reset Branch**:
  - If `data.get('reset')` evaluates to true:
    - Clears the global `chat_history = []`.
    - Returns HTTP 200 `{"answer": "Chat history cleared.", "sources": []}`.
- **Query Branch**:
  - Retrieves top pages via `retrieve_pages(question)`.
  - Generates answer via `generate_answer(question, pages, chat_history)`.
  - Appends `{"question": question, "answer": result["answer"]}` to global `chat_history`.
  - Enforces a sliding window keeping at most 6 recent turns (`chat_history.pop(0)` if length exceeds 6).
- **Response**:
  - Success: HTTP 200 `{"answer": "<str>", "sources": [<int>, ...]}`.
  - Error: HTTP 500 `{"error": "Failed: <exception_message>"}`.

---

## 3. Extraction, OCR, Retrieval, and Generation Mechanics

### Document Extraction and In-Memory Indexing (`indexer.py`)
- **Storage**: Global in-memory dictionary `PAGE_INDEX: dict[int, str]`, mapping 1-based page numbers to extracted text.
- **Reset on Upload**: Calling `index_document(filepath)` triggers `PAGE_INDEX.clear()`, wiping prior indexed data.
- **PDF Extraction**:
  - Opens PDF with PyMuPDF (`fitz.open(filepath)`).
  - Iterates through pages with `enumerate(doc)`.
  - Extracts text using `page.get_text().strip()`.
  - **OCR Fallback**: If extracted text has fewer than 20 characters (`len(text) < 20`), calls `extract_text_with_ocr(page)`.
  - If resulting text is non-empty, stores it in `PAGE_INDEX[i + 1]`. Pages with no text (or failed OCR) are skipped.
- **OCR Execution**:
  - Points to hardcoded Windows binary: `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`.
  - Renders page at 200 DPI via `page.get_pixmap(dpi=200)`.
  - Converts PNG byte buffer to a PIL Image (`Image.open(io.BytesIO(img_bytes))`).
  - Runs `pytesseract.image_to_string(img)` and strips whitespace.
- **TXT Extraction**:
  - Reads entire file with `encoding="utf-8", errors="ignore"`.
  - Splits text by whitespace (`content.split()`).
  - Partitions into 500-word chunks (`chunk_size = 500`).
  - Maps chunks to consecutive integers starting at 1 (`PAGE_INDEX[i + 1] = " ".join(chunk)`).
- **Unsupported Formats**: If file extension is not `.pdf` or `.txt`, no pages are indexed; returns `0`.

### Retrieval Mechanics (`retriever.py`)
- **Tokenization (`tokenize`)**:
  - Lowercases text and strips non-alphanumeric characters via regex `re.sub(r'[^a-z0-9\s]', '', text)`.
  - Splits on whitespace into words.
  - Filters against a static 20-word stopword list (`{"the","is","in","at","of","a","an","and","to","it","this","that","was","for","on","are","with","as","be","by"}`).
- **BM25 Scoring (`bm25_score`)**:
  - Parameters: $k_1 = 1.5$, $b = 0.75$.
  - Calculates average document length $avgdl = \frac{1}{N} \sum \text{len}(\text{tokenize}(t))$ on the fly across all entries in `PAGE_INDEX`.
  - For each unique query token:
    - Calculates term frequency ($tf$) in the candidate page tokens.
    - Calculates document frequency ($df$) by scanning and re-tokenizing every page in `PAGE_INDEX`: `sum(1 for page in PAGE_INDEX.values() if token in tokenize(page))`.
    - Computes IDF: $idf = \ln\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)$.
    - Normalizes TF: $tf\_norm = \frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \left(1 - b + b \cdot \frac{\text{len}(page\_tokens)}{avgdl}\right)}$.
    - Accumulates $score += idf \cdot tf\_norm$.
- **Page Retrieval (`retrieve_pages`)**:
  - Tokenizes the query.
  - Iterates through every page in `PAGE_INDEX`, tokenizes page text, and computes BM25 score.
  - Sorts pages descending by score and returns the top $k$ items (default `top_k=5`).
  - Unconditional return: Does not filter out pages with score `0.0`.

### Groq Generation (`generator.py`)
- **API Configuration**:
  - Endpoint: `https://api.groq.com/openai/v1/chat/completions`.
  - API Key: `GROQ_API_KEY` read from environment variables.
  - Model Name: `"openai/gpt-oss-20b"`.
  - Hyperparameters: `temperature=0.4`, `max_tokens=1024`.
- **Prompt Structure (`build_prompt`)**:
  - System instruction header: Instructs model to rely on context, use chat history, reason step-by-step for comparisons/calculations, answer `"I could not find this in the document"` if missing, and avoid hallucinations.
  - Context insertion: For each retrieved page, appends `[Page {page_num}]: {text}\n\n`. Text is hard-truncated to 800 characters (`text[:800] + "..."`).
  - Conversation history: Formatted as sequential pairs `User: ...\nAssistant: ...\n\n`.
  - Final query: `LATEST QUESTION:\n{question}\n\nANSWER (think step by step):\n`.
- **Response Handling**:
  - Extracts message content from `response.json()["choices"][0]["message"]["content"]`.
  - Returns `{"answer": <str>, "sources": [<page_numbers>]}`.
  - Source list is extracted directly from the input page list (`[page.get("page_number") for page in pages]`) regardless of relevance score or model citation.
  - Catches exceptions and returns `{"answer": f"Error: {str(e)}", "sources": []}`.

---

## 4. Important Known Risks Visible in the Code

1. **Shared In-Memory Global State (No Concurrency / Multi-Tenancy)**:
   - `indexer.PAGE_INDEX` is a single module-level dictionary. When any user uploads a document, `PAGE_INDEX.clear()` deletes the document for all concurrent users.
   - `app.chat_history` is a single global list. Chat history is shared across all clients, causing cross-user conversation leakage and race conditions.
   - All state is ephemeral and lost upon process restart. If scaled to multi-worker WSGI processes (e.g., Gunicorn), worker memory diverges.

2. **Hardcoded Platform Dependency for OCR**:
   - `pytesseract.pytesseract.tesseract_cmd` is hardcoded to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - Any execution on Linux (including Render deployment), macOS, or Windows with custom install locations will raise an unhandled `FileNotFoundError` / `TesseractNotFoundError` during OCR fallback.

3. **Deployment Configuration Inconsistencies**:
   - `render.yaml` runs `python app.py` instead of a production WSGI server (`gunicorn`), running Flask in `debug=True` on a public port (`0.0.0.0:5000`).
   - `render.yaml` uses a standard Python native environment without installing system Tesseract packages or providing the binary path.
   - `requirements.txt` specifies `dotenv` instead of `python-dotenv`, which installs a different/deprecated package on PyPI.
   - `pypdf` is listed in `requirements.txt` but never imported.

4. **Severe Algorithmic Inefficiency in BM25 Scoring**:
   - `bm25_score()` re-tokenizes the entire corpus twice per candidate page: once for `avgdl` and once inside the token loop to calculate $df$.
   - For $N$ pages and $Q$ query tokens, full corpus tokenization occurs $N \cdot (1 + Q)$ times per query ($O(N^2 \cdot Q)$). For medium or large documents, `/ask` requests will experience extreme latency or request timeouts.
   - No tokenization cache or inverted index is maintained.

5. **LLM Model Identifier & API Failure Risk**:
   - `generator.py` specifies `"model": "openai/gpt-oss-20b"`. This identifier is not a standard supported model on Groq's API and will return 4xx API errors.
   - Context is aggressively truncated to 800 characters per page (`text[:800]`), causing silent loss of relevant information on dense pages.
   - Sources returned to the UI reflect all retrieved candidates rather than pages with non-zero relevance scores or actual citations used in the answer.

6. **Input Validation and Security Risks**:
   - No file size limits (`MAX_CONTENT_LENGTH`) configured on Flask, exposing the service to memory exhaustion via large file uploads.
   - File extension check only occurs downstream in `indexer.py`. Unsupported file types are uploaded and written to disk without rejection, returning 0 pages indexed.
   - Files saved to `uploads/<filename>` can collide and overwrite each other when multiple uploads share the same filename.

---

## 5. Existing Behaviors to Preserve During Refactoring

The following functional capabilities represent the existing contract of the application and should remain operational:

1. **Web Interface**:
   - Single-page interface providing document upload controls, indexing feedback, conversation stream, and a question input box.
2. **Document Ingestion**:
   - File upload via `POST /upload` accepting `.pdf` and `.txt` files.
   - Preservation of 1-based page numbering for PDFs.
   - Virtual page chunking for `.txt` files based on fixed word counts.
   - Automatic OCR fallback for scanned or image-heavy PDF pages with sparse text (< 20 characters).
3. **Retrieval**:
   - Lexical BM25 search matching user questions against document pages.
   - Ranking of top-relevant pages provided as context for query generation.
4. **Chat & Conversation Continuity**:
   - Conversational question answering via `POST /ask` incorporating document context and prior conversational turns.
   - Multi-turn conversation memory with a sliding window of recent interactions.
   - Ability to clear/reset chat history via the UI and via `{"reset": true}` flag on `POST /ask`.
5. **Attribution & Status Feedback**:
   - Reporting total indexed page counts to the client on upload completion.
   - Returning source page references corresponding to retrieved context alongside generated answers.
6. **Error Handling**:
   - Consistent JSON error response contracts on failed uploads and generation issues.
