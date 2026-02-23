import { v4 as uuidv4 } from "uuid";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

export type RelevantTurn = {
    turn_index: number;
    speaker: string;
    text: string;
};

export type Evidence = {
    transcript_id: string;
    turn_numbers: number[];
    relevant_turns: RelevantTurn[];
    full_turns?: RelevantTurn[];
};

type ConversationText = {
    text: string;
    id: string;
};

export type BotAnswer = {
    answer: string;
    time: number;
    node_ids: string[];
    unique_text: ConversationText[];
    leave_text: ConversationText[];
};

export type AnalysisResponse = {
    bot_answer: BotAnswer;
    evidence: Evidence[];
};

export type Message = {
    role: "user" | "assistant";
    text: string;
    createdAt: number | string;
    thinking?: string;
    data?: AnalysisResponse;
};

export type Chat = {
    id: string;
    title: string;
    messages: Message[];
    createdAt: number;
};

type ChatState = {
    chats: Chat[];
    activeChatId?: string | null;
    addChat: (initialMessage: string) => Chat;
    setActiveChat: (chatId: string | null) => void;
    appendMessage: (chatId: string, msg: Message) => void;
    updateTitle: (chatId: string, title: string) => void;
    deleteChat: (chatId: string) => void;
    clear: () => void;
};

export const useChatStore = create<ChatState>()(
    persist(
        (set) => ({
            chats: [],
            activeChatId: null,

            addChat: (initialMessage: string) => {
                const id = uuidv4();
                const chat: Chat = {
                    id,
                    title: (initialMessage || "New Chat").slice(0, 40),
                    messages: initialMessage ? [{ role: "user", text: initialMessage, createdAt: Date.now() }] : [],
                    createdAt: Date.now(),
                };
                set((state) => ({
                    chats: [chat, ...state.chats],
                    activeChatId: id,
                }));
                return chat;
            },

            setActiveChat: (chatId: string | null) => set(() => ({ activeChatId: chatId })),

            appendMessage: (chatId: string, msg: Message) => {
                set((state) => ({
                    chats: state.chats.map((c) => (c.id === chatId ? { ...c, messages: [...c.messages, msg] } : c)),
                }));
            },

            updateTitle: (chatId: string, title: string) => {
                set((state) => ({
                    chats: state.chats.map((c) => (c.id === chatId ? { ...c, title } : c)),
                }));
            },

            deleteChat: (chatId: string) => {
                set((state) => {
                    const newChats = state.chats.filter((c) => c.id !== chatId);
                    const isActive = state.activeChatId === chatId;
                    return {
                        chats: newChats,
                        activeChatId: isActive ? (newChats[0]?.id ?? null) : state.activeChatId,
                    };
                });
            },

            clear: () => set({ chats: [], activeChatId: null }),
        }),
        {
            name: "flyperplex_storage",
            storage: createJSONStorage(() => sessionStorage),
        }
    )
);

export default useChatStore;
