import asyncio

from evidence import collect_evidence


class Selector:
    async def select_turns(self, query, conversation):
        return '2, 0, 2, 99, -1, invalid'


def test_evidence_uses_transcript_ids_and_preserves_multiline_turns():
    tid = '2b292d7f-1c0b-49c9-b27b-b7653bf2496c'
    records = {tid: {"conversation": [
        {"speaker": "Customer", "text": "Cancel\nmy booking"},
        {"speaker": "Agent", "text": "Which date?"},
        {"speaker": "Customer", "text": "Tomorrow"},
    ]}}
    leaves = [{"id": "tree-node", "text": f"[Id]: {tid}\n---\nSummary"}] * 2
    evidence, full_text = asyncio.run(collect_evidence('Why?', leaves, records, Selector()))
    assert len(evidence) == 1
    assert evidence[0]['turn_numbers'] == [2, 0]
    assert evidence[0]['relevant_turns'][1] == {"turn_index": 0, "speaker": "Customer", "text": "Cancel\nmy booking"}
    assert len(evidence[0]['full_turns']) == 3
    assert full_text[0]['id'] == tid
    assert 'Cancel\nmy booking' in full_text[0]['text']


def test_missing_transcript_is_not_silently_used_as_evidence():
    evidence, full_text = asyncio.run(collect_evidence('Why?', [{"id": "x", "text": "No transcript ID"}], {}, Selector()))
    assert evidence == []
    assert full_text == []
