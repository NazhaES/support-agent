# support-agent

An IT support assistant that answers employee questions from the company's own documents.

A LangGraph agent routes each question, searches the knowledge base in Pinecone, and checks
whether the retrieved text really answers it. If it does, the answer comes only from the documents,
with sources. If not, it searches the web, rewrites the question and retries, or says it doesn't know
instead of guessing.

- **Hybrid search**: meaning (embeddings) plus exact words (BM25), so codes and product names are found
- **Re-ranking**: a cross-encoder picks the best chunks out of 20 candidates
- **Evidence check**: an LLM grades the retrieved text before any answer is written
- **Web fallback**: Tavily search when the documents don't cover the question
- **Conversation memory**: follow-up questions are understood in context, saved per chat in PostgreSQL
- **Streaming chat**: answers appear word by word, with each step of the agent shown live
- **Chat history**: past chats in the sidebar, each one its own thread
- **Uploads**: add PDF, DOCX, TXT or MD files from the web page (admin key required)
- **Audit log**: every question, answer, route and step saved in PostgreSQL
- **Evaluation**: RAGAS and DeepEval on a fixed test set

## Demo video

[Watch the demo video](docs/demo-small.mp4)

## Screenshots

**Chat and agent trace**: past chats on the left, the answer from the company documents in the middle, and each step the agent took on the right.

![Chat and agent trace](docs/chat-and-agent-trace.png)

**Document upload**: add a PDF from the web page. It is split by section and indexed, with page numbers kept for citations.

![Document upload](docs/document-upload.png)

**LangSmith thread**: one conversation traced turn by turn, with every step, model call, latency and token cost.

![LangSmith thread](docs/langsmith-thread-trace.png)

**Evaluation**: DeepEval results on 10 test questions. Answers are correct, grounded in the documents, and follow the security and escalation rules.

![DeepEval results](docs/deepeval-results.png)

**RAGAS results** on the same questions: retrieval finds every needed document (context precision and recall 1.00). RAGAS faithfulness is stricter than DeepEval's and also counts details not stated in the documents.

![RAGAS results](docs/ragas-results.png)

## How it works

```
question → follow-up rewrite → router → hybrid search + re-rank → evidence grade
                                                  ├─ good → answer from documents
                                                  └─ weak → web search → grade
                                                                ├─ good → answer from web
                                                                └─ weak → rewrite and retry, or "not enough evidence"
```

## Run with Docker

```bash
docker compose up --build
```

Open http://localhost:8088. PostgreSQL starts with it.

## Run without Docker

PostgreSQL still runs in Docker:

```bash
docker compose up -d postgres
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py
```

## Settings

Copy `.env.example` to `.env` and fill in the keys:

```
GROQ_API_KEY=...
PINECONE_API_KEY=...
TAVILY_API_KEY=...
HF_TOKEN=...
PINECONE_INDEX_NAME=industry-agentic-rag-kb
PINECONE_NAMESPACE=langgraph-agentic-rag
TOP_K=2
ADMIN_API_KEY=...
POSTGRES_USER=support
POSTGRES_PASSWORD=...
POSTGRES_DB=support_agent
DATABASE_URL=postgresql://support:...@localhost:5432/support_agent
```

`DATABASE_URL` is for running without Docker. In Docker, the app connects to the `postgres` service automatically.

Optional: `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANK_MODEL`, `RERANK_CANDIDATES`, `MAX_RETRIES`, `JUDGE_MODEL`.
Defaults are in `app/core/config.py`.

## Add documents

From the web page: **Upload** a file and enter the admin key.

Or index everything in `data/`:

```bash
python ingest_sample_kb.py
```

## API

- `POST /api/chat`: the full answer as JSON
- `POST /api/chat/stream`: the answer streamed, one JSON event per line
- `POST /api/ingest`: upload a document (`X-Admin-Key` header)
- `GET /api/health`

## Evaluate

```bash
python eval_deepeval.py
python eval_ragas.py
```

Questions and expected answers are in `data/eval_dataset.json`.
DeepEval checks faithfulness, answer relevancy, contextual precision, recall and relevancy,
plus correctness, security and escalation rules.

## Folders

- `app/`: FastAPI backend, LangGraph workflow, retrieval and ingestion
- `data/`: knowledge base documents and the evaluation set
- `uploads/`: files uploaded from the web page
- `.cache/models/`: embedding and re-ranker, downloaded on first use

## Database

PostgreSQL holds the conversation memory (LangGraph checkpoint tables) and the `query_audit` table.

```bash
docker compose exec postgres psql -U support -d support_agent \
  -c "SELECT created_at, thread_id, source_used, question FROM query_audit ORDER BY id DESC LIMIT 20;"
```
