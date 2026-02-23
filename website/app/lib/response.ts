import type { AnalysisResponse } from "../stores/useChatStore";

const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
const strings = (value: unknown) => Array.isArray(value) && value.every((item) => typeof item === "string");
const indices = (value: unknown) => Array.isArray(value) && value.every((item) => Number.isInteger(item) && item >= 0);
const turns = (value: unknown) => Array.isArray(value) && value.every((item) => record(item) && Number.isInteger(item.turn_index) && Number(item.turn_index) >= 0 && typeof item.speaker === "string" && typeof item.text === "string");
const textRecords = (value: unknown) => Array.isArray(value) && value.every((item) => record(item) && typeof item.id === "string" && typeof item.text === "string");

export function isAnalysisResponse(value: unknown): value is AnalysisResponse {
    if (!record(value) || !record(value.bot_answer) || !Array.isArray(value.evidence)) return false;
    const answer = value.bot_answer;
    return typeof answer.answer === "string" && answer.answer.trim().length > 0
        && typeof answer.time === "number" && Number.isFinite(answer.time) && answer.time >= 0
        && strings(answer.node_ids) && textRecords(answer.unique_text) && textRecords(answer.leave_text)
        && value.evidence.every((item) => record(item) && typeof item.transcript_id === "string"
            && indices(item.turn_numbers) && turns(item.relevant_turns)
            && (item.full_turns === undefined || turns(item.full_turns)));
}

