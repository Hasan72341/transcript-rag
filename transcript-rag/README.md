# Transcript-RAG — Retrieval API

The Python service runs the notebook's retrieval pipeline without opening Jupyter:

1. Rewrite follow-up questions using the supplied chat history.
2. Retrieve transcript summaries from FAISS and rerank them with ColBERT.
3. Build a RAPTOR summary tree and execute a ReWOO retrieval plan.
4. Generate an answer and select supporting turns from the original transcripts.

Models are loaded once at startup. Graph state belongs to each request, so one chat does not inherit another chat's context. One inference request runs at a time per process; run Uvicorn with one worker.

## Start with Docker

Restore the artifacts listed below, then run from the project root:

```bash
docker compose up --build
```

Compose starts vLLM on port 8000 and the API on localhost port 8001. It waits for vLLM's health check before starting the API.

```bash
curl http://localhost:8001/health
curl http://localhost:8001/ready
```

`/health` checks the API process. `/ready` reports whether initialization succeeded; it does not continuously probe vLLM. If initialization fails, inspect `docker compose logs rag-api`, correct the configuration or artifacts, and run `docker compose restart rag-api`.

To run Jupyter as well:

```bash
docker compose --profile notebook up transcript-rag
```

Jupyter is available on port 8888. Its existing configuration disables token authentication; keep that port private.

## Required artifacts

Paths are relative to this directory unless configured otherwise:

| Path | Contents |
| --- | --- |
| `const/faiss_index/index.faiss` | FAISS vectors |
| `const/faiss_index/index.pkl` | Trusted LangChain document store and index mapping |
| `const/summaries-20k.json` | Transcript records with `transcript_id` and a `conversation` list |

Each conversation turn needs `text` and may include `speaker`. Indexed summary text must contain `[Id]: <transcript UUID>` matching a transcript record. The index must use the same embedding model and normalization settings as the service. The default is normalized `BAAI/bge-large-en-v1.5` embeddings.

These artifacts are not included in this checkout. The API reports HTTP 503 when they are absent. It never substitutes sample answers.

## Run without Docker

Use Python 3.11 and an environment suitable for the selected models:

```bash
cd transcript-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-api.txt
export VLLM_BASE_URL=http://localhost:8000/v1
export OMP_NUM_THREADS=1
uvicorn api:app --host 127.0.0.1 --port 8001 --workers 1
```

`OMP_NUM_THREADS=1` prevents nested FAISS/OpenMP worker pools; it is also set in the Docker image.

vLLM must already be serving the configured model. Starting with only `requirements-api.txt` installed is sufficient to inspect missing-artifact responses, but not to run retrieval.

| Environment variable | Default |
| --- | --- |
| `RAG_DATA_DIR` | `const/` beside `settings.py` |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` |
| `VLLM_MODEL` | `/model-gptoss` |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` |
| `EMBEDDING_DEVICE` | `cuda` |
| `COLBERT_MODEL` | `colbert-ir/colbertv2.0` |

Compose sets the vLLM address to `http://vllm:8000/v1`. Model caches are mounted from the root `huggingface/` directory. Changing the embedding model requires rebuilding the index.

## HTTP contract

Interactive API documentation is available at `http://localhost:8001/docs`.

```bash
curl http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Why are hotel bookings canceled?","history":[]}'
```

`message` accepts 1–4,000 characters after trimming. Optional `history` contains up to 12 `{ "role": "user" | "assistant", "text": "..." }` records, each up to 8,000 characters. History contains prior messages only; the current question is supplied separately. Only the last 12,000 characters of formatted history are used for question rewriting.

The response contains `bot_answer` (Markdown answer, elapsed seconds, retrieved node IDs, and text records) and `evidence` (transcript IDs, selected turn indices, selected turns, and full turns). Turn indices are zero-based. Full conversations in `bot_answer.leave_text` are keyed by transcript ID for compatibility with the UI.

| Status | Meaning |
| --- | --- |
| `200` | Answer generated |
| `422` | Invalid request |
| `429` | Another inference request is running |
| `502` | Retrieval or model processing failed |
| `503` | Data or model initialization is incomplete |
| `504` | A model request timed out |

The Next.js proxy translates invalid input to HTTP 400 and exposes error messages as `{ "error": "..." }`. It waits up to ten minutes; individual model requests have a 200-second timeout. A proxy timeout does not stop Python inference already in progress, and the API remains busy until that work ends.

The service has no authentication and is intended for local use. Compose binds its port to loopback.

## Tests

With the runtime dependencies installed:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

API-only tests require no models or GPU:

```bash
python -m pytest tests/test_api.py tests/test_evidence.py -q
```

The graph test uses real LangGraph and FAISS execution with deterministic model outputs. It checks orchestration and evidence mapping, not pretrained-model quality. A full inference smoke test still requires the external artifacts and model server.
