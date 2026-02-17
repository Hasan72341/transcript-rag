import type { Metadata } from "next";
import { Ubuntu_Mono } from "next/font/google";
import { ReactNode } from "react";
import "./globals.css";

const ubuntuMono = Ubuntu_Mono({
    subsets: ["latin"],
    weight: "400",
    variable: "--font-ubuntu-mono",
});

export const metadata: Metadata = {
    title: "Transcript-RAG — Conversational Intelligence System",
    description: "Analyze customer conversations with retrieval-based answers and supporting transcript evidence.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
    return (
        <html lang="en">
            <body className={`${ubuntuMono.className}`}>{children}</body>
        </html>
    );
}
