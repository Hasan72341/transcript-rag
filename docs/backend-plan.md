# Backend integration

The browser posts the current message and up to 12 prior messages to Next.js `/api/chat`. That route validates the request, forwards it to the Python `/chat` endpoint, and returns the answer and evidence. The Python service loads BGE embeddings, the FAISS index, ColBERT, and the notebook's RAPTOR/ReWOO graph once per process. Each request creates its own graph state.

The API exposes `/health` for process liveness and `/ready` for model/data initialization. Missing artifacts leave the process available for diagnostics and cause chat requests to return 503. Inference runs in a worker thread, with one request admitted at a time to protect GPU memory. Other requests receive 429. Model failures return 502; model timeouts return 504. The proxy waits up to ten minutes.

The response retains `bot_answer` and `evidence`. Full transcript turns are supplied alongside selected turns so multiline text and turn indices survive rendering. No demo response is used as an error fallback.

Implementation:
1. Add contract and error-path tests for the API and proxy.
2. Extract reusable notebook code into Python modules and add configuration and evidence normalization.
3. Add the API process to Compose and connect the chat UI, including pending, failure, and retry behavior.
4. Resolve the frontend dependency mismatch and update setup instructions.
5. Run backend and proxy tests, frontend lint/type/build checks, and HTTP smoke checks. Full model inference requires external data artifacts and a GPU.
