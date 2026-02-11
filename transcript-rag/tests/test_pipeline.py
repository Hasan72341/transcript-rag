"""Exercise the real graph and FAISS tree with deterministic model outputs."""

from types import SimpleNamespace

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM

from contracts import AnalysisResponse, HistoryMessage
from pipeline import RetrievalPipeline


class ScriptedModel(FakeListLLM):
    async def batch_acall(self, prompts):
        return ['Customers report cancellation fees.'] * len(prompts)

    async def select_turns(self, query, conversation):
        return '0, 1'


def test_graph_returns_grounded_transcript_turns_for_a_followup():
    tid = '2b292d7f-1c0b-49c9-b27b-b7653bf2496c'
    text = f'[Id]: {tid}\n---\nCustomer cancels because fees are high.'
    embeddings = DeterministicFakeEmbedding(size=8)
    # Replace only the expensive pretrained models; exercise the actual graph,
    # FAISS search, RAPTOR tree, leaf traversal, and evidence adapter.
    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.embeddings = embeddings
    pipeline.model = ScriptedModel(responses=[
        'Why did the hotel customer cancel?',
        'Plan: Find cancellation reasons. #E1 = Raptor[hotel cancellation fees]',
        'The customer canceled because the fees were high.',
    ])
    pipeline.retriever = FAISS.from_texts([text], embeddings).as_retriever()
    pipeline.reranker = SimpleNamespace(rerank=lambda query, documents, k: [{'content': document} for document in documents[:k]])
    pipeline.transcripts = {tid: {'conversation': [
        {'speaker': 'Customer', 'text': 'The fees are too high.\nPlease cancel.'},
        {'speaker': 'Agent', 'text': 'I have canceled your booking.'},
    ]}}
    pipeline.graph = pipeline._compile_graph()
    result = AnalysisResponse.model_validate(pipeline.answer('Why?', [HistoryMessage(role='user', text='Hotel cancellation')]))
    assert result.bot_answer.answer == 'The customer canceled because the fees were high.'
    assert result.bot_answer.node_ids
    assert len(result.evidence) == 1
    assert result.evidence[0].transcript_id == tid
    assert result.evidence[0].turn_numbers == [0, 1]
    assert result.evidence[0].full_turns[0].text == 'The fees are too high.\nPlease cancel.'
    assert result.bot_answer.leave_text[0].id == tid


def test_raptor_leaf_mapping_keeps_documents_in_overlapping_clusters():
    import numpy as np
    from raptor import Raptor
    from pipeline import leaf_ids

    tree = Raptor.__new__(Raptor)
    tree.NODE_IDS = {}
    tree.EDGES = []
    tree.embed_model = DeterministicFakeEmbedding(size=8)
    tree.summariser_model = ScriptedModel(responses=[])
    tree.perform_clustering = lambda embeddings, dim, threshold: [np.array([0, 1]), np.array([1])]
    tree.recursive_embed_cluster_summarize(['First transcript', 'Second transcript'], n_levels=1)
    adjacency, _ = tree.build_parent_child_adjacency()
    root = tree.NODE_IDS['Customers report cancellation fees.']
    assert set(leaf_ids([root], adjacency)) == {tree.NODE_IDS['First transcript'], tree.NODE_IDS['Second transcript']}
