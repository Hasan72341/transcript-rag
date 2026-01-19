"""
Configuration file for Query Generation Pipeline
Includes API settings, model configurations, and hyperparameters
"""

import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
# ============================================================================
# API CONFIGURATION
# ============================================================================
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in .env file")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model Selection based on Groq's available models
MODEL_CONFIG = {
    "self_instruct": {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9
    },
    "evol_transform": {
        "model": "llama-3.1-8b-instant",  # Fast and reliable model
        "temperature": 0.4,
        "max_tokens": 800,  # Shorter responses
        "top_p": 0.95
    },
    "evol_optimizer": {
        "model": "llama-3.1-8b-instant",  # Much cheaper for optimization
        "temperature": 0.6,
        "max_tokens": 1536,  # Reduced from 3072
        "top_p": 0.9
    }
}

# ============================================================================
# STAGE 1: SELF-INSTRUCT PARAMETERS
# ============================================================================

SELF_INSTRUCT_CONFIG = {
    "num_instructions_per_seed": 10,
    "num_prompt_instructions": 3,
    "request_batch_size": 5,
    "similarity_threshold": 0.7,
    "min_query_length": 10,
    "max_query_length": 300,
}

# ============================================================================
# STAGE 2: AUTO EVOL-INSTRUCT PARAMETERS
# ============================================================================

AUTO_EVOL_CONFIG = {
    "num_evolution_rounds": 2,        # Reduced from 3 to 2
    "num_optimizations": 1,           # Reduced from 3 to 2 candidate prompts
    "evolution_batch_size": 5,
    "failure_rate_threshold": 0.5,
    "max_evolution_attempts": 3,
    "max_optimization_steps": 3,
    "min_length_increase": 2,         # Need meaningful addition (not just 1 word)
    "max_length_total": 40,           # NEW: Maximum 40 words total (keep queries concise)
    "semantic_stagnation_threshold": 0.93,  # Stricter: need actual rewording (was 0.98)
    "max_failure_rate": 0.5,
    "enable_optimization": True,     # Enable optimization to save tokens
    "min_queries_for_optimization": 10,  # Minimum queries needed to run optimization
}
# ============================================================================
# DIFFICULTY DISTRIBUTION (for balanced dataset)
# ============================================================================

DIFFICULTY_CONFIG = {
    "easy_percentage": 0.30,      # 30% stay as-is (0 rounds)
    "medium_percentage": 0.50,    # 50% evolve 1-2 rounds
    "hard_percentage": 0.20,      # 20% evolve 3 rounds
    "medium_rounds": [1, 2],      # Random choice for medium
    "hard_rounds": 3              # Fixed for hard
}

# ============================================================================
# DOMAIN CONFIGURATION
# ============================================================================

DOMAINS = [
    "Hotel",
    "Flight", 
    "Retail",
    "Banking",
    "Telecom",
    "Insurance"

]

TARGET_QUERIES_STAGE1 = 20
TARGET_QUERIES_STAGE2 = 20

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    "stage1_output": "output/self_instruct_queries.json",
    "stage2_output": "output/evolved_queries.json",
    "logs_dir": "logs/",
    "save_intermediate": True,
}

# ============================================================================
# EVOLUTION COMPLEXITY DIMENSIONS
# ============================================================================

EVOLUTION_DIMENSIONS = [
    "Add constraints or requirements",
    "Deepen the inquiry", 
    "Increase reasoning complexity",
    "Complicate the input scenario",
    "Add emotional or contextual tone",
    "Introduce ambiguity or noise"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(stage: str) -> Dict[str, Any]:
    return MODEL_CONFIG.get(stage, MODEL_CONFIG["self_instruct"])

def get_output_path(stage: str) -> str:
    os.makedirs("output", exist_ok=True)
    return OUTPUT_CONFIG.get(f"stage{stage}_output", f"output/stage{stage}_output.json")

def get_logs_dir() -> str:
    os.makedirs(OUTPUT_CONFIG["logs_dir"], exist_ok=True)
    return OUTPUT_CONFIG["logs_dir"]
