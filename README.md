# Transcript-RAG — Conversational Intelligence System

Transcript-RAG answers questions about customer-support conversations and returns the transcript turns that support each answer. It combines FAISS retrieval, ColBERT reranking, RAPTOR summary trees, and a ReWOO planning graph.

The project includes a Python API, a Next.js chat interface, the original research notebook, and query-generation experiments.

## How it works

```text
Chat UI → Next.js /api/chat → Python /chat
                                  │
                         FAISS → ColBERT → RAPTOR/ReWOO
                                  │
                         vLLM answer + transcript evidence
```

The backend loads models once and creates separate graph state for each request. The frontend shows the returned answer and lets you inspect selected turns or full conversations.

## Run the application

Full inference requires an NVIDIA GPU, Docker Compose with NVIDIA Container Toolkit, model weights, and the transcript artifacts below. The weights and artifacts are not included in this checkout.

| Project-root path | Required contents |
| --- | --- |
| `model-gptoss/` | Model weights served by vLLM |
| `huggingface/` | Embedding and ColBERT model cache; models may download on first use |
| `transcript-rag/const/faiss_index/` | `index.faiss` and trusted `index.pkl` |
| `transcript-rag/const/summaries-20k.json` | Original transcript records and summaries |

The original model artifacts were stored at `gs://recor-rag/model-gptoss` and `gs://recor-rag/huggingface`. Their availability and access permissions have not been verified. Obtain the matching transcript data and index from the project artifacts separately.

After restoring the files, start the backend from the project root:

```bash
docker compose up --build
```

Check readiness in another terminal:

```bash
curl http://localhost:8001/ready
```

Then start the frontend with Node.js 22.18+ and Yarn 1:

```bash
cd website
yarn install --frozen-lockfile
cp .env.example .env.local
yarn dev
```

Open <http://localhost:3000> and ask a question such as “Why are hotel bookings canceled?” The interface uses real API requests. Missing artifacts or unavailable models produce an error; they do not trigger a sample answer.

| Service | Address |
| --- | --- |
| Web interface | `http://localhost:3000` |
| Retrieval API and interactive docs | `http://localhost:8001/docs` |
| vLLM | `http://localhost:8000` |

See [backend setup and API contract](transcript-rag/README.md) and [frontend setup and tests](website/README.md) for configuration and troubleshooting.

## Research notebook

To start Jupyter alongside the API:

```bash
docker compose --profile notebook up transcript-rag
```

Open `http://localhost:8888`, then run `answer-generation.ipynb` in order. Its experiments are separate from the API process and load their own models. The existing Jupyter configuration disables token authentication; keep port 8888 private.

## Project structure

| Path | Contents |
| --- | --- |
| [`transcript-rag/`](transcript-rag/) | Retrieval API, model adapter, RAPTOR/ReWOO pipeline, tests, and research notebook |
| [`website/`](website/) | Chat interface, Next.js API proxy, and frontend tests |
| [`queries-generation/method-1/`](queries-generation/method-1/) | Domain/intent-based query generation and evaluation notebooks |
| [`queries-generation/method-2/`](queries-generation/method-2/) | Self-Instruct generation and Auto Evol-Instruct refinement |
| [`queries-generation/method-3/`](queries-generation/method-3/) | Query generation from transcripts |
| [`simulated-real-query-system/`](simulated-real-query-system/) | Task 1 and task 2 query spreadsheets |
| [`technical-report/Technical_Report.pdf`](technical-report/Technical_Report.pdf) | Technical report |

## Query-generation experiments

Each method has separate inputs and dependencies:

- [Method 1 generation](queries-generation/method-1/query-generation/README.md) and [evaluation](queries-generation/method-1/query-evaluation/README.md)
- [Method 2: Self-Instruct and Auto Evol-Instruct](queries-generation/method-2/README.md)
- [Method 3: transcript-based generation](queries-generation/method-3/README.md)

These scripts generate evaluation questions; they are not required to serve chat requests.
