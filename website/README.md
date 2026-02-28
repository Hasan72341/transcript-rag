# Transcript-RAG — Web Interface

A chat interface for asking questions about customer-support conversations and inspecting the transcript evidence behind each answer.

Messages are sent to the Next.js `/api/chat` route, which forwards them to the Python retrieval API. The UI displays request progress and errors, supports retrying a failed question, and keeps answers associated with the chat that submitted them. There is no sample-response fallback.

## Development

Use Node.js 22.18 or newer and Yarn 1. Run from this directory:

```bash
yarn install --frozen-lockfile
cp .env.example .env.local
yarn dev
```

Open <http://localhost:3000>. The Python backend defaults to `http://127.0.0.1:8001`; change `RAG_API_URL` in `.env.local` if necessary, then restart Next.js. This variable is server-side only. See the [backend setup](../transcript-rag/README.md) for model and data requirements.

The page loads without the backend, but questions require a ready retrieval service. When artifacts are missing, the UI shows the backend's error rather than a generated answer.

## Chat and evidence

Chat history is stored in browser `sessionStorage` under the legacy key `flyperplex_storage` to preserve existing chats. Each request includes the current question and up to 12 prior messages from that chat. Requests are stateless on the backend.

The evidence panel shows selected transcript turns. “Show Full” uses structured `full_turns`, preserving multiline text and original turn indices. Older saved responses can still use conversation text from `bot_answer.leave_text`.

Only one question can be submitted at a time in this page. Switching chats while waiting does not redirect the response. Retrying a failed request does not append the user message again.

## Checks

```bash
yarn test
yarn lint
yarn tsc --noEmit
yarn build
```

Proxy tests use a local HTTP server to check forwarding, validation, backend errors, invalid responses, and timeouts. The production build loads Ubuntu Mono through `next/font/google` and requires access to Google Fonts.

For browser tests:

```bash
yarn playwright install chromium
yarn test:e2e
```

The browser tests run the real UI with controlled API responses. They do not require model weights or a GPU.

To serve the production build:

```bash
yarn start
```

## Source files

| File | Purpose |
| --- | --- |
| [`app/page.tsx`](app/page.tsx) | Chat UI, requests, retry behavior, and evidence sidebar |
| [`app/api/chat/route.ts`](app/api/chat/route.ts) | Next.js route entry point |
| [`app/lib/chat-proxy.ts`](app/lib/chat-proxy.ts) | Request validation, backend forwarding, and timeout handling |
| [`app/lib/response.ts`](app/lib/response.ts) | Runtime response validation |
| [`app/stores/useChatStore.ts`](app/stores/useChatStore.ts) | Shared response types and persisted chat state |

The response contract is documented in the [backend README](../transcript-rag/README.md#http-contract).
