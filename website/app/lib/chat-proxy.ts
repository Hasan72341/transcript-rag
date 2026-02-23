import { isAnalysisResponse } from "./response.ts";

type HistoryMessage = { role: "user" | "assistant"; text: string };
type ChatRequest = { message: string; history: HistoryMessage[] };

const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown, max: number): value is string => typeof value === "string" && value.trim().length > 0 && value.trim().length <= max;
function parseRequest(value: unknown): ChatRequest | null {
    if (!record(value) || !text(value.message, 4000) || Object.keys(value).some((key) => !["message", "history"].includes(key))) return null;
    const history = value.history ?? [];
    if (!Array.isArray(history) || history.length > 12) return null;
    if (!history.every((item) => record(item) && ["user", "assistant"].includes(String(item.role)) && text(item.text, 8000))) return null;
    return { message: value.message.trim(), history: history.map((item) => ({ role: item.role, text: item.text.trim() })) };
}

export async function proxyChat(request: Request, options: { url?: string; timeoutMs?: number } = {}): Promise<Response> {
    let input: ChatRequest | null;
    try {
        input = parseRequest(await request.json());
    } catch {
        return Response.json({ error: "Send a valid JSON request." }, { status: 400 });
    }
    if (!input) return Response.json({ error: "Enter a message of 1–4,000 characters and at most 12 prior messages." }, { status: 400 });

    const url = (options.url ?? process.env.RAG_API_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");
    try {
        const response = await fetch(`${url}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(input),
            cache: "no-store",
            signal: AbortSignal.timeout(options.timeoutMs ?? 600_000),
        });
        const data: unknown = await response.json();
        if (!response.ok) {
            const status = [429, 502, 503, 504].includes(response.status) ? response.status : 502;
            const error = record(data) && typeof data.detail === "string" ? data.detail : "The retrieval service could not answer this request.";
            return Response.json({ error }, { status });
        }
        if (!isAnalysisResponse(data)) return Response.json({ error: "The retrieval service returned an invalid response." }, { status: 502 });
        return Response.json(data);
    } catch (error) {
        const timedOut = error instanceof Error && ["TimeoutError", "AbortError"].includes(error.name);
        return Response.json({ error: timedOut ? "Answer generation timed out. Try again with a shorter question." : "Cannot reach the retrieval service. Check that the backend is running." }, { status: timedOut ? 504 : 502 });
    }
}
