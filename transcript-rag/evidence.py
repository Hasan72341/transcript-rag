"""Map retrieved tree leaves back to original transcript turns."""

import asyncio
import re

TRANSCRIPT_ID = re.compile(r"\[Id\]:\s*([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})", re.IGNORECASE)


async def collect_evidence(query, leaves, transcripts, model):
    transcript_ids = dict.fromkeys(
        match.group(1).lower()
        for leaf in leaves
        if (match := TRANSCRIPT_ID.search(leaf["text"]))
    )

    async def extract(transcript_id):
        conversation = transcripts[transcript_id]["conversation"]
        content = await model.select_turns(query, conversation)
        indices = list(dict.fromkeys(
            int(value.strip()) for value in content.split(",")
            if value.strip().isdigit() and int(value.strip()) < len(conversation)
        ))[:5]
        full_turns = [
            {"turn_index": index, "speaker": turn.get("speaker", ""), "text": turn["text"]}
            for index, turn in enumerate(conversation)
        ]
        return {
            "transcript_id": transcript_id,
            "turn_numbers": indices,
            "relevant_turns": [full_turns[index] for index in indices],
            "full_turns": full_turns,
        }

    evidence = await asyncio.gather(*(extract(tid) for tid in transcript_ids if tid in transcripts))
    conversations = [
        {"id": item["transcript_id"], "text": "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in item["full_turns"])}
        for item in evidence
    ]
    return evidence, conversations
