"""
Utility modules for query generation pipeline

This package contains:
- llm_client: Groq API wrapper
- prompts: Prompt templates for Self-Instruct and Auto Evol-Instruct
- failure_detector: Quality control and failure detection
"""

from .llm_client import GroqClient
from .failure_detector import EvolutionFailureDetector

__all__ = [
    'GroqClient',
    'EvolutionFailureDetector'
]
