"use client";

import React, { useState, useEffect, useRef } from "react";
import { Menu, Plus, Settings, Send, MessageSquare, Trash2, Clock, FileText, X, ChevronRight, User, Bot, ChevronDown, ChevronUp } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { isAnalysisResponse } from "./lib/response";
import { useChatStore, type AnalysisResponse, type RelevantTurn } from "./stores/useChatStore";

const EvidenceSidebar = ({ isOpen, onClose, data }: { isOpen: boolean; onClose: () => void; data: AnalysisResponse | null }) => {
    const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

    if (!data) return null;

    const toggleExpand = (id: string) => {
        setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
    };

    return (
        <>
            {/* Backdrop */}
            {isOpen && <div className="fixed inset-0 bg-black/20 z-40 transition-opacity" onClick={onClose} />}

            {/* Sidebar */}
            <div
                className={`fixed inset-y-0 right-0 z-50 w-full sm:w-[500px] bg-white shadow-2xl transform transition-transform duration-300 ease-in-out ${
                    isOpen ? "translate-x-0" : "translate-x-full"
                }`}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="flex items-center justify-between p-5 border-b border-gray-100">
                        <div>
                            <h2 className="text-lg font-semibold text-gray-800">Evidence Detail</h2>
                            <p className="text-xs text-gray-500 mt-1">Found {data.evidence.length} relevant transcripts</p>
                        </div>
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-500">
                            <X size={20} />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-y-auto p-5 space-y-6 bg-[#f8f9fa]">
                        {data.evidence.map((ev, i) => {
                            const fullTextObj = data.bot_answer.leave_text.find((t) => t.id === ev.transcript_id);
                            const isExpanded = !!expandedIds[ev.transcript_id];
                            let fullConversation: RelevantTurn[] = [];

                            if (fullTextObj) {
                                fullConversation = fullTextObj.text.split("\n").map((line, idx) => {
                                    const colonIndex = line.indexOf(":");
                                    let speaker = "System";
                                    let text = line;
                                    if (colonIndex !== -1) {
                                        speaker = line.substring(0, colonIndex).trim();
                                        text = line.substring(colonIndex + 1).trim();
                                    }
                                    return { turn_index: idx, speaker, text };
                                });
                            }
                            fullConversation = ev.full_turns ?? fullConversation;
                            const displayedTurns = isExpanded && fullConversation.length > 0 ? fullConversation : ev.relevant_turns;

                            return (
                                <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                                    {/* Card Header */}
                                    <div className="bg-[#f0f4f9] px-4 py-3 border-b border-gray-200 flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <FileText size={14} className="text-blue-600" />
                                            <span className="text-xs font-semibold text-gray-700 font-mono">ID: {ev.transcript_id.slice(0, 8)}...</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="text-[10px] bg-white px-2 py-0.5 rounded-full text-gray-500 border border-gray-200">
                                                {isExpanded ? "All Turns" : `${ev.relevant_turns.length} Relevant Turns`}
                                            </span>
                                            {fullConversation.length > 0 && (
                                                <button
                                                    onClick={() => toggleExpand(ev.transcript_id)}
                                                    className="flex items-center gap-1 text-[10px] bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full hover:bg-blue-200 transition-colors"
                                                >
                                                    {isExpanded ? (
                                                        <>
                                                            Show Less <ChevronUp size={12} />
                                                        </>
                                                    ) : (
                                                        <>
                                                            Show Full <ChevronDown size={12} />
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                    </div>

                                    {/* Card Body with Unified View */}
                                    <div className="p-4">
                                        <div className="space-y-4 max-h-[400px] overflow-y-auto pr-1">
                                            {displayedTurns.map((turn, tIdx) => (
                                                <div key={tIdx} className={`flex gap-3 ${turn.speaker === "Customer" ? "flex-row" : "flex-row-reverse"}`}>
                                                    <div
                                                        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                                                            turn.speaker === "Customer" ? "bg-purple-100 text-purple-600" : "bg-blue-100 text-blue-600"
                                                        }`}
                                                    >
                                                        {turn.speaker === "Customer" ? <User size={14} /> : <Bot size={14} />}
                                                    </div>
                                                    <div
                                                        className={`flex-1 p-3 rounded-lg text-sm leading-relaxed ${
                                                            turn.speaker === "Customer" ? "bg-gray-50 text-gray-800 rounded-tl-none" : "bg-blue-50 text-blue-900 rounded-tr-none"
                                                        }`}
                                                    >
                                                        <div className="text-[10px] uppercase font-bold tracking-wider opacity-50 mb-1">{turn.speaker}</div>
                                                        <p className="whitespace-pre-wrap">{turn.text}</p>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </>
    );
};

type PendingRequest = {
    chatId: string;
    message: string;
    history: { role: "user" | "assistant"; text: string }[];
};

export default function Home() {
    const requestInFlight = useRef(false);
    const [pendingChatId, setPendingChatId] = useState<string | null>(null);
    const [failedRequest, setFailedRequest] = useState<(PendingRequest & { error: string }) | null>(null);
    const [inputText, setInputText] = useState("");
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(false);
    const [selectedEvidenceData, setSelectedEvidenceData] = useState<AnalysisResponse | null>(null);
    const [isEvidencePanelOpen, setIsEvidencePanelOpen] = useState(false);
    const { chats, activeChatId, addChat, setActiveChat, appendMessage, deleteChat } = useChatStore();

    const activeChat = chats.find((c) => c.id === activeChatId) || null;
    const messagesEndRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        const checkMobile = () => {
            const mobile = window.innerWidth < 768;
            setIsMobile(mobile);
            setIsSidebarOpen(false);
        };
        checkMobile();
        window.addEventListener("resize", checkMobile);
        return () => window.removeEventListener("resize", checkMobile);
    }, []);
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [activeChat?.messages]);

    const sendRequest = async (request: PendingRequest) => {
        if (requestInFlight.current) return;
        requestInFlight.current = true;
        setPendingChatId(request.chatId);
        setFailedRequest(null);
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: request.message, history: request.history }),
                signal: AbortSignal.timeout(610_000),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : "Could not generate an answer.");
            if (!isAnalysisResponse(data)) throw new Error("The server returned an invalid answer.");
            appendMessage(request.chatId, {
                role: "assistant",
                text: data.bot_answer.answer,
                createdAt: Date.now(),
                data,
            });
        } catch (error) {
            setFailedRequest({ ...request, error: error instanceof Error ? error.message : "Could not reach the server." });
        } finally {
            requestInFlight.current = false;
            setPendingChatId(null);
        }
    };

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault();
        const message = inputText.trim();
        if (!message || message.length > 4000 || requestInFlight.current) return;
        const history = (activeChat?.messages ?? []).slice(-12).map(({ role, text }) => ({ role, text: text.slice(-8000) }));
        const chatId = activeChat?.id ?? addChat(message).id;
        if (activeChat) appendMessage(chatId, { role: "user", text: message, createdAt: Date.now() });
        setInputText("");
        void sendRequest({ chatId, message, history });
    };

    const handleNewChat = () => {
        setActiveChat(null);
        setInputText("");
        if (isMobile) setIsSidebarOpen(false);
    };

    const openEvidence = (data: AnalysisResponse) => {
        setSelectedEvidenceData(data);
        setIsEvidencePanelOpen(true);
    };

    return (
        <div className="flex h-screen text-[#1f1f1f] font-sans overflow-hidden matrix bg-white">
            {/* --- SIDEBAR --- */}
            <aside
                className={`fixed inset-y-0 left-0 z-40 transition-all matrix duration-300 ease-in-out flex flex-col bg-[#f0f4f9]
                    ${isSidebarOpen ? "w-[280px] translate-x-0" : "-translate-x-full md:translate-x-0 md:w-0 md:opacity-0 md:pointer-events-none"}
                `}
            >
                <div className="p-4 flex items-center justify-between">
                    <button onClick={handleNewChat} className="flex items-center gap-3 bg-white shadow-sm hover:shadow-md text-[#444746] px-4 py-3 rounded-xl transition-all w-full sm:w-auto">
                        <Plus size={20} />
                        <span className="text-sm font-medium">New chat</span>
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-4 py-2">
                    <div className="text-xs font-medium text-gray-500 mb-3 px-2">Recent</div>
                    <div className="space-y-1">
                        {chats.map((chat) => (
                            <div
                                key={chat.id}
                                className={`group flex items-center justify-between p-2 rounded-full cursor-pointer text-sm text-[#444746] hover:bg-white transition-colors
                                ${activeChatId === chat.id ? "bg-[#d3e3fd] text-[#001d35] font-medium" : ""}
                                `}
                                onClick={() => {
                                    setActiveChat(chat.id);
                                    if (isMobile) setIsSidebarOpen(false);
                                }}
                            >
                                <div className="flex items-center gap-2 overflow-hidden">
                                    <MessageSquare size={16} className="shrink-0" />
                                    <span className="truncate max-w-[160px]">{chat.title}</span>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        deleteChat(chat.id);
                                    }}
                                    className={`opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity ${activeChatId === chat.id ? "opacity-100" : ""}`}
                                >
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="p-4 mt-auto space-y-2">
                    <button className="flex items-center gap-3 w-full p-2 text-sm text-[#444746] hover:bg-white rounded-lg transition-colors">
                        <Settings size={18} />
                        {isSidebarOpen && <span>Settings</span>}
                    </button>
                </div>
            </aside>

            {/* --- OVERLAY for Mobile --- */}
            {isMobile && isSidebarOpen && <div className="fixed inset-0 bg-black/50 z-30" onClick={() => setIsSidebarOpen(false)} />}

            {/* --- MAIN CONTENT --- */}
            <main className={`flex-1 flex flex-col h-full transition-all duration-300 relative p-2 md:p-5 ${isSidebarOpen && !isMobile ? "ml-[280px]" : "ml-0"}`}>
                <div className="flex-1 flex flex-col h-full transition-all duration-300 relative rounded-2xl bg-white shadow-xl">
                    {/* HEADER */}
                    <header className="flex items-center justify-between px-4 py-3 sticky top-0 z-20">
                        <div className="flex items-center gap-2">
                            <button aria-label="Toggle chat sidebar" onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-gray-100 rounded-full text-gray-600 transition-colors">
                                <Menu size={24} />
                            </button>
                            <div>
                                <span className="text-xl font-medium text-gray-600 tracking-tight">Transcript-RAG</span>
                                <p className="text-xs text-gray-500">Conversational Intelligence System</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium text-sm">F</div>
                        </div>
                    </header>

                    {/* CONTENT AREA */}
                    <div className="flex-1 overflow-y-auto w-full relative">
                        <div className="max-w-[900px] mx-auto w-full min-h-full flex flex-col">
                            {!activeChat ? (
                                <div className="flex-1 flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-500">
                                    <div className="mb-8">
                                        <h1 className="text-5xl sm:text-6xl font-medium text-[#c4c7c5] tracking-tight mb-2">Hello, User</h1>
                                        <h1 className="text-5xl sm:text-6xl font-medium text-[#444746] tracking-tight">How can I help you today?</h1>
                                    </div>
                                    <p className="text-gray-400 mb-8">Try asking: why hotel bookings get cancel</p>
                                </div>
                            ) : (
                                <div className="flex-1 p-4 pb-32 pt-8 space-y-8">
                                    {activeChat.messages.map((m, idx) => (
                                        <div key={idx} className="w-full flex flex-col gap-2">
                                            {m.role === "user" ? (
                                                <div className="flex justify-end">
                                                    <div className="bg-[#f0f4f9] text-[#1f1f1f] max-w-[80%] px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed">{m.text}</div>
                                                </div>
                                            ) : (
                                                /* ASSISTANT MESSAGE */
                                                <div className="flex gap-4 max-w-3xl w-full mx-auto md:ml-0 md:mr-auto">
                                                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 shrink-0 mt-1" />

                                                    <div className="flex-1 overflow-hidden">
                                                        {/* Metadata Header (if structured data exists) */}
                                                        {m.data && (
                                                            <div className="flex items-center gap-2 mb-2 text-xs text-green-700 bg-green-50 w-fit px-2 py-1 rounded-md border border-green-100">
                                                                <Clock size={12} />
                                                                <span>Processed in {(m.data as AnalysisResponse).bot_answer.time.toFixed(2)}s</span>
                                                            </div>
                                                        )}

                                                        {/* Main Content */}
                                                        <div className="prose prose-p:text-[#1f1f1f] prose-headings:text-[#1f1f1f] max-w-none text-sm leading-relaxed">
                                                            <ReactMarkdown>{m.text}</ReactMarkdown>
                                                        </div>

                                                        {/* EVIDENCE PREVIEW CARD */}
                                                        {m.data && (
                                                            <div className="mt-4">
                                                                <button
                                                                    onClick={() => openEvidence(m.data as AnalysisResponse)}
                                                                    className="group flex flex-col sm:flex-row sm:items-center justify-between w-full bg-white border border-gray-200 hover:border-blue-300 hover:shadow-md hover:shadow-blue-50 transition-all rounded-xl p-3 text-left"
                                                                >
                                                                    <div className="flex items-start gap-3">
                                                                        <div className="bg-blue-50 text-blue-600 p-2 rounded-lg group-hover:bg-blue-100 transition-colors">
                                                                            <FileText size={18} />
                                                                        </div>
                                                                        <div>
                                                                            <h4 className="text-sm font-semibold text-gray-800">View Source Evidence</h4>
                                                                            <p className="text-xs text-gray-500 mt-0.5">
                                                                                {(m.data as AnalysisResponse).evidence.length} conversations analyzed for this answer.
                                                                            </p>
                                                                        </div>
                                                                    </div>
                                                                    <div className="mt-2 sm:mt-0 self-end sm:self-center">
                                                                        <ChevronRight size={16} className="text-gray-400 group-hover:text-blue-500" />
                                                                    </div>
                                                                </button>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                    <div ref={messagesEndRef} />
                                </div>
                            )}
                        </div>
                    </div>

                    {/* INPUT AREA */}
                    <div className="p-4 w-full flex justify-center items-center flex-col gap-4">
                        <div className="max-w-[900px] mx-auto w-full relative">
                            {pendingChatId === activeChatId && pendingChatId && (
                                <p role="status" className="mb-3 text-sm text-gray-600">Searching conversations and generating an answer…</p>
                            )}
                            {failedRequest?.chatId === activeChatId && failedRequest && (
                                <div role="alert" className="mb-3 rounded-lg bg-red-50 p-3 text-sm text-red-800">
                                    <p>{failedRequest.error}</p>
                                    <button type="button" disabled={pendingChatId !== null} onClick={() => void sendRequest(failedRequest)} className="mt-2 font-medium underline disabled:opacity-50">Retry question</button>
                                </div>
                            )}
                            <form
                                onSubmit={handleSubmit}
                                className={`bg-[#f0f4f9] rounded-[2rem] transition-all duration-200 border border-transparent focus-within:bg-white focus-within:shadow-lg focus-within:border-gray-200 relative
                                    ${activeChat ? "min-h-[60px]" : "min-h-[60px]"}
                                `}
                            >
                                <div className="flex items-center px-4 py-3 gap-3">
                                    <button type="button" className="p-2 bg-[#d3e3fd] hover:bg-[#c4d7fc] text-[#001d35] rounded-full transition-colors shrink-0">
                                        <Plus size={20} />
                                    </button>
                                    <textarea
                                        aria-label="Ask about customer conversations"
                                        maxLength={4000}
                                        disabled={pendingChatId !== null}
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter" && !e.shiftKey) {
                                                e.preventDefault();
                                                handleSubmit(e);
                                            }
                                        }}
                                        placeholder="Ask about hotel cancellations..."
                                        className="w-full bg-transparent border-none outline-none text-[#1f1f1f] placeholder:text-[#5e5e5e] text-base resize-none max-h-32 py-2"
                                        rows={1}
                                        style={{ minHeight: "24px" }}
                                    />
                                    <div className="flex items-center gap-1 shrink-0">
                                        <button
                                            type="submit"
                                            aria-label="Send message"
                                            disabled={!inputText.trim() || pendingChatId !== null}
                                            className={`p-2 ${
                                                inputText.trim() ? "bg-blue-600 hover:bg-blue-700" : "bg-gray-200"
                                            } rounded-full text-white transition-all animate-in zoom-in duration-200 relative`}
                                        >
                                            <Send className="size-5 -translate-x-px translate-y-px" />
                                        </button>
                                    </div>
                                </div>
                            </form>
                        </div>

                        <div className="text-center mt-3">
                            <p className="text-xs text-[#5e5e5e]">Check the source evidence before relying on an answer.</p>
                        </div>
                    </div>
                </div>
            </main>

            {/* --- SLIDING EVIDENCE SIDEBAR --- */}
            <EvidenceSidebar isOpen={isEvidencePanelOpen} onClose={() => setIsEvidencePanelOpen(false)} data={selectedEvidenceData} />
        </div>
    );
}
