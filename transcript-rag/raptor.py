"""RAPTOR clustering and retrieval extracted from the research notebook."""

RANDOM_SEED = 42
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate # type: ignore
import asyncio
import faiss
import uuid
import umap
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class Raptor:
    def __init__(self, docs, summariser_model, embed_model):
        self.doc = docs
        self.NODE_IDS = {}
        self.EDGES = []
        self.summariser_model = summariser_model
        self.embed_model = embed_model
        self.results = self.recursive_embed_cluster_summarize(docs, level=1, n_levels=3)
        self.ALL_NODES = {v: {"text": k, "id": v} for k, v in self.NODE_IDS.items()}
        self.adjacency, self.roots = self.build_parent_child_adjacency(self.NODE_IDS, self.EDGES)
        self.levels = self.compute_node_levels(self.adjacency)
        self.levels = self.invert_levels(self.levels)
        self.node_embeddings_map = self._create_flat_index()
        self.all_embeddings = np.array(
            [self.node_embeddings_map[nid] for nid in list(self.node_embeddings_map.keys())],
            dtype="float32"
        )
        dim = self.all_embeddings.shape[1]
        self.node_id_to_int = {nid: i for i, nid in enumerate(list(self.node_embeddings_map.keys()))}
        self.int_to_node_id = {i: nid for i, nid in enumerate(list(self.node_embeddings_map.keys()))}

        int_ids = np.array(list(self.node_id_to_int.values())).astype("int64")
        base_index = faiss.IndexFlatL2(dim)  # exact L2 search
        self.index = faiss.IndexIDMap(base_index)
        self.index.add_with_ids(self.all_embeddings, int_ids)

        print(f"FAISS index built")

    def global_cluster_embeddings(self, embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

    def local_cluster_embeddings(self, embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
        return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

    def get_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED) -> int:
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    def perform_clustering(self, embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
        import time
        start = time.time()
        if len(embeddings) <= dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        global_clusters, n_global_clusters = self.GMM_cluster(reduced_embeddings_global, threshold)
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0
        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]
            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )
            total_clusters += n_local_clusters
        end = time.time()
        print(f"Clustering took {end - start} seconds")
        return all_local_clusters

    def embed(self, texts):
        text_embeddings = self.embed_model.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np

    def embed_cluster_texts(self, texts):
        text_embeddings_np = self.embed(texts)
        cluster_labels = self.perform_clustering(text_embeddings_np, 10, 0.1)
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        return df

    def fmt_txt(self, df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize_texts(self, texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clusters = self.embed_cluster_texts(texts)
        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append({"text": row["text"], "embd": row["embd"], "cluster": cluster})
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()
        print(f"Generated Clusters: {len(all_clusters)}")
        template ="""Analyze the provided sub-set of conversational data transcripts. Your task is to generate a concise, highly-abstracted summary (max 300 words) that focuses only on the following:

        1.  *Causal Motifs:* What recurring conversational patterns, agent behaviors, or dialogue segments are described as leading to specific business events (e.g., escalations, refunds, churn)?
        2.  *Dialogue Dynamics:* Summarize the key structural elements of the conversation (e.g., branching flows, repeated queries, silences) that complicate analysis.
        3.  *Core Entities:* Identify and list the main operational entities, business events, or outcomes mentioned (e.g., 'refund requests,' 'escalation to supervisors').

        The goal is to provide a high-level, actionable abstraction of the data's causal content, not a detailed restatement of the transcripts.

        {context}"""
        prompt = ChatPromptTemplate.from_template(template)
        all_formatted_txts = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            prompt_txt = prompt.format(context=formatted_txt)
            all_formatted_txts.append(prompt_txt)
        summaries = asyncio.run(self.summariser_model.batch_acall(all_formatted_txts))

        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })
        return df_clusters, df_summary

    def get_id_for_text(self, text: str):
        if text not in self.NODE_IDS:
            self.NODE_IDS[text] = str(uuid.uuid4())
        return self.NODE_IDS[text]

    def recursive_embed_cluster_summarize(self, texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        results = {}
        if level == 1:
            for t in texts:
                self.get_id_for_text(t)
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)
        for _, row in df_summary.iterrows():
            summary_text = row["summaries"]
            summary_cluster = row["cluster"]
            summary_id = self.get_id_for_text(summary_text)
            children_df = df_clusters[df_clusters["cluster"].apply(lambda labels: summary_cluster in labels)]
            for child_text in children_df["text"].tolist():
                child_id = self.get_id_for_text(child_text)
                self.EDGES.append((child_id, summary_id))
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            next_results = self.recursive_embed_cluster_summarize(new_texts, level + 1, n_levels)
            results.update(next_results)
        return results

    def build_parent_child_adjacency(self, NODE_IDS=None, EDGES=None):
        if NODE_IDS is None:
            NODE_IDS = self.NODE_IDS
        if EDGES is None:
            EDGES = self.EDGES
        adjacency = defaultdict(list)
        all_children = set()
        all_parents = set()
        for child, parent in EDGES:
            adjacency[parent].append(child)
            all_children.add(child)
            all_parents.add(parent)
        root_nodes = list(all_parents - all_children)
        return adjacency, root_nodes

    def compute_node_levels(self, adjacency=None):
        if adjacency is None:
            adjacency, _ = self.build_parent_child_adjacency()
        nodes = set(adjacency.keys())
        for children in adjacency.values():
            nodes.update(children)
        in_degree = {n: 0 for n in nodes}
        for parent, children in adjacency.items():
            for c in children:
                in_degree[c] += 1
        leaves = [n for n in nodes if n not in adjacency]
        levels = {leaf: 0 for leaf in leaves}
        reverse_adj = defaultdict(list)
        for parent, children in adjacency.items():
            for c in children:
                reverse_adj[c].append(parent)
        queue = deque(leaves)
        while queue:
            node = queue.popleft()
            node_level = levels[node]
            for parent in reverse_adj[node]:
                if parent not in levels:
                    levels[parent] = node_level + 1
                else:
                    levels[parent] = max(levels[parent], node_level + 1)
                queue.append(parent)
        return levels

    def invert_levels(self, levels: dict) -> dict:
        level_to_nodes = {}
        for node, level in levels.items():
            if level not in level_to_nodes:
                level_to_nodes[level] = []
            level_to_nodes[level].append(node)
        return level_to_nodes

    def get_leaf_nodes(self) -> List[str]:
        """Return all leaf nodes (nodes with no children)."""

        all_nodes = set(self.NODE_IDS.values())
        parent_nodes = set(self.adjacency.keys())
        leaf_nodes = all_nodes - parent_nodes

        return list(leaf_nodes)

    def filter_leaf_nodes(self, node_ids: List[str]) -> List[str]:
        """Filter only those IDs that are leaves."""
        return [
            node_id
            for node_id in node_ids
            if node_id not in self.adjacency  # no children
        ]

    def retrieve_collapsed(self, query: str, top_k: int = 10):
        query_vec = self.embed_model.embed_documents([query])
        query_vec = np.array(query_vec, dtype="float32")
        distances, int_ids = self.index.search(query_vec, top_k)

        results = []
        for internal_id in int_ids[0]:
            if internal_id == -1:
                continue  # FAISS returns -1 if no result

            node_id = self.int_to_node_id[internal_id]
            node_data = self.ALL_NODES[node_id]

            results.append({
                "id": node_id,
                "text": node_data["text"]
            })

        return results

    def _create_flat_index(self):
        """
        Internal helper: Flattens the tree by aggregating embeddings from all levels
        into a single lookup dictionary {node_id: embedding_vector}.
        """
        node_embeddings = {}
    
        for level, (df_clusters, _) in self.results.items():
            for _, row in df_clusters.iterrows():
                text = row["text"]
                if text in self.NODE_IDS:
                    node_id = self.NODE_IDS[text]
                    node_embeddings[node_id] = row["embd"]
        for node_id, content in self.ALL_NODES.items():
            if node_id not in node_embeddings:
                emb = self.embed_model.embed_documents([content["text"]])[0]
                node_embeddings[node_id] = emb
    
        return node_embeddings

