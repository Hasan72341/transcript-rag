"""Runtime paths and model settings; no model loading on import."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", str(Path(__file__).parent / "const"))))
    vllm_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"))
    model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "/model-gptoss"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cuda"))
    colbert_model: str = field(default_factory=lambda: os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0"))

    def validate_artifacts(self):
        required = ["faiss_index/index.faiss", "faiss_index/index.pkl", "summaries-20k.json"]
        missing = [name for name in required if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing RAG artifacts: " + ", ".join(missing))
