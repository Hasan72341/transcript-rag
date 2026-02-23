import { proxyChat } from "../../lib/chat-proxy";

export const runtime = "nodejs";
export const maxDuration = 600;

export async function POST(request: Request) {
    return proxyChat(request);
}
