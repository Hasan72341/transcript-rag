"""
Universal, Unbiased Failure Detector for Instruction Evolution
Domain-agnostic • Reasoning-centric • No word-count bias
OPTIMIZED: Lower F1 threshold, fixed hallucination patterns, added imperative verbs
"""

import re
from typing import Dict, List, Tuple
# Using standard Python stop words for better keyword extraction
from nltk.corpus import stopwords as nltk_stopwords
from collections import Counter
import math

# Download stopwords if not present (only needs to run once)
try:
    _ = nltk_stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

# Define stopwords set globally for efficiency
STOPWORDS = set(nltk_stopwords.words('english'))

class EvolutionFailureDetector:
    """Detects failed evolutions with *no domain bias* and *reasoning-based evaluation*."""

    def __init__(self):

        # 1. Answer-like patterns → if model *answers* instead of rewriting the instruction
        self.answer_like_patterns = [
            r"\bthe results\b",
            r"\bthe findings\b",
            r"\bwe found\b",
            r"\bthe data shows\b",
            r"\bthis indicates that\b",
            r"\bin conclusion\b",
            r"\btherefore\b",
            r"\bso this means\b",
            r"\bwe observed\b",
            r"\bthe outcome\b",
        ]

        # 2. Hallucination patterns (strictly domain-neutral)
        # FIXED: Removed standalone "where", "join" to avoid flagging common English words
        self.hallucination_patterns = [
            # dates with context (not just month names alone)
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",

            # fake IDs
            r"\b[A-Z]{2,}\d{4,}\b",
            r"\b(ref|case|id|order|acct|policy|accn|claim)[-_]?\d{3,}\b",

            # names (must include first AND last name to avoid false positives)
            r"\b(agent|mr|ms|mrs)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",

            # system / schema hallucinations (SQL-specific patterns only)
            r"\bselect\b.*\bfrom\b.*\bwhere\b",  # Full SQL SELECT statement
            r"\bjoin\b.*\bon\b",                  # JOIN only with ON clause
            r"\btbl_[a-z_]+\b",                   # Table names
            r"\b[A-Z]{3,}_[A-Z_]+\b",             # Constant names
        ]

        # 3. Over-formal patterns (not realistic for instructions)
        self.formal_patterns = [
            r"\bkindly\b",
            r"\bhereby\b",
            r"\bwith reference to\b",
            r"\bin accordance with\b",
            r"\bplease be advised\b",
        ]

        # 4. Reasoning markers (domain-neutral)
        self.reasoning_indicators = [
            "analyze", "explain", "reason", "deduce", "infer",
            "evaluate", "assess", "break down", "multi-step",
            "considering", "depending on", "case where",
            "hypothetical", "scenario", "causal", "relationship",
            "contrast", "compare", "sequence",
        ]

        # 5. Constraint markers
        self.constraint_indicators = [
            "under condition", "only if", "unless", "assuming",
            "when", "while also", "simultaneously", "excluding",
            "including", "across", "over time",
        ]

    # ============================================================
    #  INDIVIDUAL CHECKS
    # ============================================================

    def _get_keywords(self, text: str) -> List[str]:
        """Extracts non-stopword, non-number keywords."""
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in STOPWORDS and not w.isdigit() and len(w) > 2]
        return keywords

    def check_answer_like(self, text: str) -> Tuple[bool, str]:
        """Detect if the model produced an answer instead of an evolved instruction."""
        t = text.lower()

        for p in self.answer_like_patterns:
            if re.search(p, t):
                return True, f"Answer-like output detected: '{p}'"

        # OPTIMIZED: Added imperative verbs to recognize valid analytical instructions
        inquiry_markers = [
            "how", "why", "what", "should", "explain", "analyze",
            "assess", "evaluate", "examine", "investigate", "determine",
            "identify", "compare", "describe", "explore", "inspect",
            "find", "calculate", "measure", "estimate", "predict",
            "list", "outline", "summarize", "contrast", "test"
        ]
        
        if not any(m in t for m in inquiry_markers):
            if not text.strip().endswith("?"):
                return True, "Output is declarative, not an evolved instruction."

        return False, ""

    def check_hallucination(self, text: str) -> Tuple[bool, str]:
        """Reject fabricated details (dates, IDs, names, SQL, etc.)."""
        for p in self.hallucination_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return True, f"Hallucinated detail detected: '{m.group()}'"
        return False, ""

    def check_formality(self, text: str) -> Tuple[bool, str]:
        """Avoid unnatural bureaucratic tone."""
        t = text.lower()
        for p in self.formal_patterns:
            if re.search(p, t):
                return True, f"Over-formal phrasing: '{p}'"
        return False, ""

    def check_intent_preservation(self, original: str, evolved: str) -> Tuple[bool, str]:
        """Ensure evolved query maintains the core objective using F1-score on keywords."""

        orig_keywords = set(self._get_keywords(original))
        evol_keywords = set(self._get_keywords(evolved))

        if not orig_keywords:
            return False, "" # Cannot check intent if original is empty/stopwords

        # Precision: How many evolved keywords were in the original?
        precision = len(orig_keywords.intersection(evol_keywords)) / len(evol_keywords) if evol_keywords else 0
        
        # Recall: How many original keywords were preserved?
        recall = len(orig_keywords.intersection(evol_keywords)) / len(orig_keywords)

        # F1-score (harmonic mean)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # This allows analytical reframing while preserving core intent
        # Lower threshold needed because evolved queries SHOULD use different vocabulary
        INTENT_DRIFT_THRESHOLD = 0.25

        if f1_score < INTENT_DRIFT_THRESHOLD and len(evol_keywords) > 2:
            return True, f"Intent drift (F1={f1_score:.2f}): Evolved query lost core concepts."

        return False, ""

    def check_reasoning_growth(self, original: str, evolved: str) -> Tuple[bool, str]:
        """Ensure reasoning complexity is preserved and ideally increased."""
        o = original.lower()
        e = evolved.lower()

        # Score 1: Reasoning keywords
        orig_score_r = sum(1 for k in self.reasoning_indicators if k in o)
        evol_score_r = sum(1 for k in self.reasoning_indicators if k in e)

        # Score 2: Constraint markers (multi-hop complexity signals)
        orig_score_c = sum(1 for k in self.constraint_indicators if k in o)
        evol_score_c = sum(1 for k in self.constraint_indicators if k in e)

        orig_total = orig_score_r + orig_score_c
        evol_total = evol_score_r + evol_score_c

        # CRITICAL FIX: Fail only if the score *decreased*. 
        # Rely on auto_evol.py's ΔDepth metric to enforce *increase*.
        if evol_total < orig_total:
            return True, "Reasoning score decreased (lost complexity)."

        return False, ""
    
    # ============================================================
    # MASTER CHECK
    # ============================================================

    def detect_all_failures(self, evolved: str, original: str, domain: str) -> Dict[str, any]:
        """
        Unified failure decision. 
        FIXED: Accepts 'domain' argument to resolve TypeError in auto_evol.py
        """

        # Although 'domain' is not used in this domain-agnostic file, 
        # it is required here to match the signature in auto_evol.py.
        # Domain-specific checks could be added here later.

        checks = {
            "answer_like": self.check_answer_like(evolved),
            "hallucination": self.check_hallucination(evolved),
            "formality": self.check_formality(evolved),
            "intent_drift": self.check_intent_preservation(original, evolved),
            "reasoning_growth": self.check_reasoning_growth(original, evolved),
        }

        is_failed = any(f for f, _ in checks.values())

        return {
            "is_failed": is_failed,
            "failure_reasons": [r for f, r in checks.values() if f],
            "checks": {name: {"failed": f, "reason": r} for name, (f, r) in checks.items()},
        }

    # ============================================================
    # BATCH FILTER
    # ============================================================

    def filter_failed_queries(self, queries: List[Dict], verbose=False):
        passed, failed = [], []

        for q in queries:
            result = self.detect_all_failures(
                evolved=q["query"],
                original=q.get("original", ""),
                domain=q.get("domain", "") # Pass domain if available
            )

            if result["is_failed"]:
                q["failure_info"] = result
                failed.append(q)
                if verbose:
                    print("\nFAILED:", q["query"])
                    for r in result["failure_reasons"]:
                        print(" -", r)
            else:
                passed.append(q)

        return passed, failed
