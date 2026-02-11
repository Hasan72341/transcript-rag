"""Reusable RAPTOR/ReWOO pipeline; the notebook remains the research entry point."""

import asyncio
import json
import re
import sys
import time
import types
from typing import Annotated, TypedDict

import httpx
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from evidence import collect_evidence
from llm import VLLMWrapper
from prompts import PLANNER_PROMPT, SOLVER_PROMPT, query_rewrite_prompt
from raptor import Raptor


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    raptor: object
    plan_string: str
    evidence: dict[str, str]
    node_ids: list[str]


def leaf_ids(node_ids, adjacency):
    visited, leaves = set(), set()
    pending = list(node_ids)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        children = adjacency.get(node, [])
        if children:
            pending.extend(children)
        else:
            leaves.add(node)
    return sorted(leaves)


class RetrievalPipeline:
    def __init__(self, settings):
        settings.validate_artifacts()
        response = httpx.get(f"{settings.vllm_url}/models", timeout=10)
        response.raise_for_status()
        if settings.model not in {model["id"] for model in response.json()["data"]}:
            raise ValueError("VLLM_MODEL does not match a model served by vLLM")
        self.model = VLLMWrapper(server_url=settings.vllm_url, model_name=settings.model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        store = FAISS.load_local(str(settings.data_dir / "faiss_index"), self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = store.as_retriever(search_kwargs={"k": 1000})
        # RAGatouille 0.0.9 still imports the pre-LangChain-1 compressor path.
        compatibility = types.ModuleType("langchain.retrievers.document_compressors.base")
        compatibility.BaseDocumentCompressor = BaseDocumentCompressor
        sys.modules.setdefault("langchain.retrievers.document_compressors.base", compatibility)
        from ragatouille import RAGPretrainedModel
        self.reranker = RAGPretrainedModel.from_pretrained(settings.colbert_model)
        with (settings.data_dir / "summaries-20k.json").open() as file:
            records = json.load(file)
        self.transcripts = {record["transcript_id"].lower(): record for record in records}
        if not self.transcripts or any(not isinstance(record.get("conversation"), list) for record in records):
            raise ValueError("Transcript records must include a conversation list")

        self.graph = self._compile_graph()

    def _compile_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("rewrite", self.summary_gen)
        graph.add_node("retrieve", self.make_raptor)
        graph.add_node("planner", self.planner_node)
        graph.add_node("worker", self.worker_node)
        graph.add_node("solver", self.solver_node)
        for origin, target in zip([START, "rewrite", "retrieve", "planner", "worker", "solver"], ["rewrite", "retrieve", "planner", "worker", "solver", END]):
            graph.add_edge(origin, target)
        return graph.compile()

    def answer(self, message, history):
        started = time.monotonic()
        # Context belongs to this request; there is no shared ChatBot.summary.
        summary = "\n".join(f"{item.role}: {item.text}" for item in history)[-12000:]
        result = self.graph.invoke({"messages": [HumanMessage(content=message)], "summary": summary})
        tree = result["raptor"]
        nodes = result["node_ids"]
        leaves = [tree.ALL_NODES[node] for node in leaf_ids(nodes, tree.adjacency)]
        evidence, conversations = asyncio.run(collect_evidence(result["messages"][0].content, leaves, self.transcripts, self.model))
        return {
            "bot_answer": {
                "answer": result["messages"][-1].content,
                "time": time.monotonic() - started,
                "node_ids": nodes,
                "unique_text": [tree.ALL_NODES[node] for node in nodes],
                "leave_text": conversations,
            },
            "evidence": evidence,
        }

    def make_raptor(self, state):
        query = state["messages"][-1].content.strip()
        documents = [document.page_content for document in self.retriever.invoke(query)]
        if not documents:
            raise ValueError("The retrieval index returned no documents")
        reranked = self.reranker.rerank(query=query, documents=documents, k=min(100, len(documents)))
        return {"raptor": Raptor([item["content"] for item in reranked], self.model, self.embeddings)}

    def planner_node(self, state: ChatState):
        question = state["messages"][-1].content
        summary = state["summary"]

        chain = PLANNER_PROMPT | self.model
        result = chain.invoke({"question": question,"summary": summary})
        return {"plan_string": result}

    def solver_node(self, state: ChatState):
        plan_string = state["plan_string"]
        evidence = state["evidence"]
        question = state["messages"][-1].content
        full_context = ""
        regex_pattern = r"(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        for line in plan_string.split('\n'):
            if "=" in line:
                match = re.search(regex_pattern, line)
                if match:
                    step_id = match.group(1)
                    if step_id in evidence:
                        full_context += f"{line}\nRESULT: {evidence[step_id]}\n---\n"
                    else:
                        full_context += f"{line}\nRESULT: No evidence found.\n---\n"
            else:
                full_context += f"{line}\n"

        chain = SOLVER_PROMPT | self.model

        final_answer = chain.invoke({"question": question, "plan_evidence": full_context, "summary": state["summary"]})
        return {"messages": [AIMessage(content=final_answer)]}

    def worker_node(self, state: ChatState):
        plan_string = state["plan_string"]
        raptor = state["raptor"]
        summary = state['summary']
        evidence = {}
        regex_pattern = r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        matches = re.findall(regex_pattern, plan_string)
        node_ids = set()
        for match in matches:
            description, step_id, tool, tool_input = match
            resolved_input = tool_input
            context_dependency_exists = False

            for prev_id, prev_text in evidence.items():
                if prev_id in resolved_input:
                    resolved_input = resolved_input.replace(prev_id, prev_text)
                    context_dependency_exists = True
            if tool == "Raptor":
                search_query = resolved_input
                if context_dependency_exists:
                    extraction_prompt = f"""
                    You are a Search Query Optimizer for a retrieval system.
                    Your goal is to extract the key entities and intent from the 'Context' to answer the 'Task'.

                    Rules:
                    1. Prioritize SPECIFIC ENTITIES (Names, IDs, Promo Codes, Error Messages).
                    2. Discard conversational filler (e.g., "The agent said", "The customer asked").
                    3. The query must be short (5-10 words) and optimized for vector search.
                    4. Output ONLY the query string.

                    ---
                    Example 1:
                    Task: Search for the terms of #E1
                    Context: "The customer Lily Kelly mentioned she is using the Summer20 promo code for her booking..."
                    Distilled Query: Summer20 promo code terms and conditions

                    Example 2:
                    Task: Identify the resolution for #E1
                    Context: "Issue: Double booking error for Jack Bell (bk-12345). The agent offered a refund..."
                    Distilled Query: Jack Bell double booking resolution refund
                    ---

                    Current Task: {description}
                    Context: {resolved_input[:3000]}...

                    Distilled Query:"""
                    query_msg = self.model.invoke(extraction_prompt)
                    search_query = query_msg.strip().replace('"', '')
                nodes = raptor.retrieve_collapsed(search_query, top_k=10)
                node_ids_match = [node["id"] for node in nodes]
                node_ids.update(node_ids_match)
                context_text = "\n".join([f"- {n['text']}" for n in nodes])
                evidence[step_id] = context_text

            elif tool == "LLM":
                msg = f"{description}\n\nInput Context:\n{resolved_input} More Context:\n{summary}"
                response = self.model.invoke(msg)
                evidence[step_id] = response

        if not node_ids:
            raise ValueError("The planner did not retrieve evidence")
        return {"evidence": evidence, "node_ids": sorted(node_ids)}

    def summary_gen(self, state: ChatState):
        summary = state.get("summary", "")
        query = str(state["messages"][-1].content).strip()
        if not summary:
            new_query = query
        else:
            prompt = query_rewrite_prompt.format(summary=summary, query=query)
            new_query = self.model.invoke(prompt).strip()
            
        return {"messages": [HumanMessage(content=new_query, id=state["messages"][-1].id)]}

