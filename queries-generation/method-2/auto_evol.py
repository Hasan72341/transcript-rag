import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import random
import os
import time
import re

# --- New Imports for Non-LLM Metrics ---
import spacy
from bert_score import score as bert_score_func 
import textstat
# --- End New Imports ---

from utils.llm_client import GroqClient
from utils.prompts import (
    format_evol_prompt, 
    format_trajectory_analyzer_prompt, 
    format_method_optimizer_prompt,
    FAILURE_DETECTION_PROMPT,
    INITIAL_EVOLVING_METHOD  # ← ADDED: Import the paper's template
)
from utils.failure_detector import EvolutionFailureDetector
from config import AUTO_EVOL_CONFIG, MODEL_CONFIG, get_output_path, DIFFICULTY_CONFIG, SELF_INSTRUCT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Initialization for Metric Tools ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise
# --- End Global Initialization ---

class AutoEvolInstruct:
    """Auto Evol-Instruct with trajectory analysis and method optimization"""
    
    def __init__(self, evol_client: GroqClient, optimizer_client: GroqClient = None):
        self.evol_client = evol_client
        self.optimizer_client = optimizer_client or evol_client
        self.config = AUTO_EVOL_CONFIG
        self.evol_model = MODEL_CONFIG["evol_transform"]
        self.optimizer_model = MODEL_CONFIG["evol_optimizer"]
        self.failure_detector = EvolutionFailureDetector()
        self.evolved_queries = []
        self.evolution_trajectories = []
        self.optimization_history = []
        
        # Metrics tracking
        self.total_tokens_used = 0
        self.total_llm_calls = 0
        self.total_evolution_time = 0
        self.vote_agreement_count = 0
        self.vote_disagreement_count = 0
        
        #  FIXED: Use paper's Initial Evolving Method (e₀) from Figure 2
        # This will be optimized into e₁, e₂, ..., e* during run_optimization_cycle
        self.current_method = INITIAL_EVOLVING_METHOD
        
        logger.info(" Initialized with paper's Initial Evolving Method (e₀)")

    # ========================================================================
    # NON-LLM METRICS CALCULATION UTILITIES
    # ========================================================================
    
    def _calculate_non_llm_metrics(self, original_query: str, evolved_query: str) -> Dict[str, Any]:
        """Calculates all deterministic metrics and deltas (non-LLM)"""
        
        doc_orig = nlp(original_query)
        doc_evol = nlp(evolved_query)
        
        # --- 1. Syntactic Complexity (Tree Depth Delta) ---
        def get_max_depth(doc):
            """Calculates the max depth of the dependency tree (proxy for complexity)"""
            root_token = None
            for sent in doc.sents:
                root_token = sent.root
                break 
            
            if root_token is None: 
                return 0
                
            def recursive_depth(token):
                if not list(token.children):
                    return 1
                return 1 + max(recursive_depth(child) for child in token.children)
            
            return recursive_depth(root_token)

        depth_orig = get_max_depth(doc_orig)
        depth_evol = get_max_depth(doc_evol)
        
        # --- 2. Multi-Hop Causal Density (Proxy for Reasoning) ---
        causal_markers = [
            r'\b(because|since|therefore|thus|consequently|if.*then|causes|results\s+in|leads\s+to)\b',
        ]
        
        causal_count_evol = sum(len(re.findall(pattern, evolved_query.lower())) for pattern in causal_markers)
        
        # --- 3. Semantic Novelty (BERTScore Delta) ---
        bert_score_f1 = 0.0
        if evolved_query and original_query:
            try:
                P, R, F1 = bert_score_func([evolved_query], [original_query], lang="en", rescale_with_baseline=True, verbose=False)
                bert_score_f1 = F1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}. Returning 0.0.")

        # --- 4. Lexical Sophistication (Readability Delta) ---
        fk_grade_orig = textstat.flesch_kincaid_grade(original_query)
        fk_grade_evol = textstat.flesch_kincaid_grade(evolved_query)
        
        return {
            "depth_orig": depth_orig,
            "depth_evol": depth_evol,
            "depth_delta": depth_evol - depth_orig,
            "causal_count": causal_count_evol,
            "bert_score_f1": bert_score_f1,
            "fk_grade_delta": fk_grade_evol - fk_grade_orig,
        }

    # ========================================================================
    # OPTIMIZED: FAILURE CHECKING AND VOTING LOGIC
    # ========================================================================
    
    def check_failure_with_voting(self, original: str, evolved: str, domain: str) -> Tuple[bool, Dict]:
        """
        Voting system: Both function and LLM vote on failure.
        OPTIMIZED: Query marked as FAILED if any one agree (or logic)
        """
        vote_start_time = time.time()
        
        # --- STEP 1: CALCULATE NON-LLM METRICS ---
        metrics = self._calculate_non_llm_metrics(original, evolved)
        
        # --- STEP 2: FUNCTION VOTE (Deterministic Checks) ---
        function_failure_reasons = []
        is_failed_deterministic = False
        
        # 1. Length Stagnation Check
        length_increase = len(evolved.split()) - len(original.split())
        evolved_length = len(evolved.split())
        max_length = self.config.get("max_length_total", 50)
        
        if length_increase < self.config.get("min_length_increase", 3):
            function_failure_reasons.append("STAGNATION: Failed minimum length increase requirement.")
            is_failed_deterministic = True
        
        # 1b. TOO LONG Check
        if evolved_length > max_length:
            function_failure_reasons.append(f"TOO LONG: Query has {evolved_length} words, max is {max_length} (not friendly).")
            is_failed_deterministic = True

        # 2. OPTIMIZED: Complexity Check - Only fail if depth DECREASED (not stagnation)
        if metrics["depth_delta"] < 0:  # Negative = simpler structure
            function_failure_reasons.append("DEGENERATION: Syntactic depth decreased.")
            is_failed_deterministic = True
        
        # 3. OPTIMIZED: Semantic Stagnation - Lowered threshold from 0.95 to 0.93
        if metrics["bert_score_f1"] > self.config.get("semantic_stagnation_threshold", 0.93): 
             function_failure_reasons.append("STAGNATION: BERTScore too high, indicating mere paraphrasing (F1 > 0.93).")
             is_failed_deterministic = True

        # 4. Readability Degeneration
        if metrics["fk_grade_delta"] < 0:
            function_failure_reasons.append("DEGENERATION: Flesch-Kincaid grade level decreased (simplification).")
            is_failed_deterministic = True

        # 5. Failure Detector Checks (intent drift, hallucination, etc.)
        function_result = self.failure_detector.detect_all_failures(
            evolved=evolved,
            original=original,
            domain=domain
        )
        if function_result["is_failed"]:
            function_failure_reasons.extend(function_result.get("failure_reasons", []))
            is_failed_deterministic = True
            
        function_vote = "FAIL" if is_failed_deterministic else "PASS"
        
        # --- STEP 3: LLM VOTE (Nuanced Judgment) ---
        llm_prompt = FAILURE_DETECTION_PROMPT.format(
            original=original,
            evolved=evolved,
            domain=domain
        )
        
        llm_response = self.evol_client.chat_completion(
            messages=[{"role": "user", "content": llm_prompt}],
            model=self.evol_model["model"],
            temperature=0.3,
            max_tokens=10
        )
        
        self.total_llm_calls += 1
        estimated_tokens = 110
        self.total_tokens_used += estimated_tokens
        
        llm_vote = "FAIL" if (llm_response and "yes" in llm_response.lower()) else "PASS"
        
        # --- STEP 4: OPTIMIZED VOTING LOGIC - BOTH must agree for FAIL (AND logic) ---
        if function_vote == "FAIL" or llm_vote == "FAIL":
            final_decision = "FAIL"
        else:
            final_decision = "PASS"
        
        # Track agreement
        if function_vote == llm_vote:
            self.vote_agreement_count += 1
        else:
            self.vote_disagreement_count += 1
        
        vote_time = time.time() - vote_start_time
        
        voting_details = {
            "function_vote": function_vote,
            "llm_vote": llm_vote,
            "final_decision": final_decision,
            "function_reasons": function_failure_reasons,
            "votes_breakdown": {
                "function": function_vote,
                "llm": llm_vote,
                "agreement": function_vote == llm_vote
            },
            "vote_time_seconds": round(vote_time, 3),
            "tokens_used": estimated_tokens
        }
        
        is_successful = (final_decision == "PASS")
        
        if not voting_details["votes_breakdown"]["agreement"]:
            logger.debug(f"🗳️ Vote disagreement: Func={function_vote}, LLM={llm_vote} → Final={final_decision}")
        
        # Add metrics to output
        voting_details["non_llm_metrics"] = metrics
        
        return is_successful, voting_details

    # ========================================================================
    # REST OF THE CLASS (UNCHANGED)
    # ========================================================================
    
    def assign_difficulty_levels(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign difficulty levels and evolution rounds to queries"""
        total = len(queries)
        num_easy = int(total * DIFFICULTY_CONFIG["easy_percentage"])
        num_medium = int(total * DIFFICULTY_CONFIG["medium_percentage"])
        num_hard = total - num_easy - num_medium
        
        shuffled = queries.copy()
        random.shuffle(shuffled)
        
        for i, query in enumerate(shuffled):
            if i < num_easy:
                query["target_difficulty"] = "easy"
                query["evolution_rounds"] = 0
            elif i < num_easy + num_medium:
                query["target_difficulty"] = "medium"
                query["evolution_rounds"] = random.choice(DIFFICULTY_CONFIG["medium_rounds"])
            else:
                query["target_difficulty"] = "hard"
                query["evolution_rounds"] = DIFFICULTY_CONFIG["hard_rounds"]
        
        logger.info(f"\n📊 Difficulty Distribution Assigned:")
        logger.info(f"   Easy (0 rounds): {num_easy} queries ({DIFFICULTY_CONFIG['easy_percentage']*100:.0f}%)")
        logger.info(f"   Medium (1-2 rounds): {num_medium} queries ({DIFFICULTY_CONFIG['medium_percentage']*100:.0f}%)")
        logger.info(f"   Hard (3 rounds): {num_hard} queries ({DIFFICULTY_CONFIG['hard_percentage']*100:.0f}%)")
        
        return shuffled
    
    def evolve_single_query(self, query: str, domain: str, method: str = None, round_num: int = 0, trajectory: list = None) -> Tuple[str, bool, Dict]:
        """Evolve a single query with VOTING-BASED acceptance"""
        evolution_start_time = time.time()
        
        if method is None:
            method = self.current_method

        prompt = format_evol_prompt(query=query, domain=domain, method=method, trajectory=trajectory)

        evolved = self.evol_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.evol_model["model"],
            temperature=self.evol_model["temperature"],
            max_tokens=self.evol_model["max_tokens"]
        )
        
        self.total_llm_calls += 1
        evolution_tokens = 350
        self.total_tokens_used += evolution_tokens
        
        if not evolved:
            logger.error(f"❌ LLM returned empty response for query: '{query[:60]}...'")
            logger.error(f"   Model: {self.evol_model['model']}, Round: {round_num}")
            return query, False, {
                "voting_details": {
                    "votes_breakdown": {
                        "function": "FAIL",
                        "llm": "FAIL",
                        "agreement": True
                    }
                },
                "evolution_time_seconds": 0,
                "tokens_used": 0
            }
            
        evolved = self._clean_evolved_output(evolved)
        
        if evolved.strip().lower() == query.strip().lower():
            logger.warning(f"LLM returned unchanged query! Original: '{query[:50]}...'")
        
        # Voting system for failure detection
        is_successful, voting_details = self.check_failure_with_voting(
            original=query,
            evolved=evolved,
            domain=domain
        )
        
        evolution_time = time.time() - evolution_start_time
        
        metrics = {
            "voting_details": voting_details,
            "evolution_time_seconds": round(evolution_time, 3),
            "tokens_used": evolution_tokens + voting_details.get("tokens_used", 0)
        }
        
        return evolved, is_successful, metrics
    
    def _clean_evolved_output(self, text: str) -> str:
        """Clean LLM output to extract final evolved query"""
        text = text.strip()
        
        prefixes = [
            "evolved query:", 
            "evolved:", 
            "here is the evolved query:", 
            "output:", 
            "step 4 #finally rewritten instruction:",
            "finally rewritten instruction:"
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        if '\n' in text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if "step 4" in line.lower() and "finally" in line.lower():
                    if i + 1 < len(lines):
                        text = lines[i + 1].strip()
                        break
            else:
                non_empty = [l.strip() for l in lines if l.strip()]
                if non_empty:
                    text = non_empty[-1]
        
        return text.strip('"').strip("'").strip()
    
    def evolve_with_trajectory(self, query: str, domain: str, num_rounds: int = 3) -> Dict[str, Any]:
        """Evolve query through multiple rounds with voting and metrics"""
        trajectory_start_time = time.time()
        
        trajectory = [query]
        success_flags = []
        voting_history = []
        round_metrics = []
        current = query
        
        if num_rounds == 0:
            return {
                "original": query,
                "domain": domain,
                "trajectory": trajectory,
                "final": query,
                "success_flags": [],
                "failure_rate": 0.0,
                "voting_history": [],
                "total_time_seconds": 0,
                "total_tokens": 0
            }
        
        for round_num in range(num_rounds):
            evolved, is_successful, metrics = self.evolve_single_query(
                query=current, 
                domain=domain, 
                round_num=round_num,
                trajectory=trajectory.copy()
            )
            
            trajectory.append(evolved)
            success_flags.append(is_successful)
            voting_history.append(metrics["voting_details"])
            round_metrics.append(metrics)
            
            if not is_successful:
                failure_reasons = metrics.get("voting_details", {}).get("function_reasons", [])
                logger.debug(f"Round {round_num} FAILED: {', '.join(failure_reasons[:2])}")
            
            # Retry once if evolution failed
            if not is_successful:
                if round_num < num_rounds - 1:
                    logger.info(f"Round {round_num}: Evolution FAILED, retrying with adjusted prompt...")
                    
                    failure_reasons = metrics.get("voting_details", {}).get("function_reasons", [])
                    retry_hints = []
                    
                    if any("length" in r.lower() for r in failure_reasons):
                        retry_hints.append("- Add 5-10 more words to increase analytical depth")
                    if any("complexity" in r.lower() or "depth" in r.lower() for r in failure_reasons):
                        retry_hints.append("- Introduce a causal relationship or multi-step reasoning")
                    if any("paraphras" in r.lower() or "bert" in r.lower() for r in failure_reasons):
                        retry_hints.append("- Use different phrasing and introduce new analytical concepts")
                    if any("grade" in r.lower() or "simplif" in r.lower() for r in failure_reasons):
                        retry_hints.append("- Use more sophisticated analytical vocabulary")
                    
                    retry_method = self.current_method
                    if retry_hints:
                        retry_method += "\n\nIMPORTANT ADJUSTMENTS FOR THIS EVOLUTION:\n" + "\n".join(retry_hints)
                    
                    evolved_retry, is_successful_retry, metrics_retry = self.evolve_single_query(
                        query=current, 
                        domain=domain, 
                        method=retry_method,
                        round_num=round_num,
                        trajectory=trajectory.copy()
                    )
                    
                    if is_successful_retry:
                        logger.info(f" Round {round_num}: Retry succeeded!")
                        evolved = evolved_retry
                        success_flags[-1] = True
                        voting_history[-1] = metrics_retry["voting_details"]
                        round_metrics[-1] = metrics_retry
                    else:
                        logger.warning(f"❌ Round {round_num}: Retry also failed, keeping failed evolution")
            
            if is_successful:
                current = evolved
                logger.debug(f" Round {round_num}: Using evolved query for next round")
            else:
                logger.debug(f"Round {round_num}: Keeping current query, will retry evolution in next round")
                current = current
        
        final_query = trajectory[-1]
        failure_rate = 1 - (sum(success_flags) / len(success_flags)) if success_flags else 0.0
        trajectory_time = time.time() - trajectory_start_time
        total_tokens = sum(m.get("tokens_used", 0) for m in round_metrics)
        
        return {
            "original": query,
            "domain": domain,
            "trajectory": trajectory,
            "final": final_query,
            "success_flags": success_flags,
            "failure_rate": failure_rate,
            "voting_history": voting_history,
            "total_time_seconds": round(trajectory_time, 3),
            "total_tokens": total_tokens
        }
    
    def _calculate_agreement_rate(self, voting_history: List[Dict]) -> float:
        """Calculate how often function and LLM agreed"""
        if not voting_history:
            return 1.0
        
        agreements = sum(1 for v in voting_history if v.get("votes_breakdown", {}).get("agreement", False))
        return round(agreements / len(voting_history), 3)
    
    def analyze_trajectories(self, trajectories: List[Dict[str, Any]], sample_size: int = 10) -> str:
        """Analyze failed trajectories to identify patterns"""
        failed_trajs = [t for t in trajectories if t["failure_rate"] > 0.3]
        
        if not failed_trajs:
            failed_trajs = random.sample(trajectories, min(sample_size, len(trajectories)))
        else:
            failed_trajs = random.sample(failed_trajs, min(sample_size, len(failed_trajs)))
        
        analyses = []
        for traj in failed_trajs[:5]:
            prompt = format_trajectory_analyzer_prompt(
                trajectory=traj["trajectory"], 
                domain=traj["domain"]
            )
            
            analysis = self.optimizer_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.optimizer_model["model"],
                temperature=self.optimizer_model["temperature"],
                max_tokens=self.optimizer_model["max_tokens"]
            )
            
            self.total_llm_calls += 1
            self.total_tokens_used += 500
            
            if analysis:
                analyses.append(analysis)
        
        return "\n\n".join([f"Trajectory {i+1}:\n{a}" for i, a in enumerate(analyses)])
    
    def optimize_evolving_method(self, trajectory_analysis: str, current_problems: List[str]) -> str:
        """Generate improved evolving method"""
        prompt = format_method_optimizer_prompt(
            trajectory_analysis=trajectory_analysis,
            current_method=self.current_method,
            problems=current_problems
        )
        
        optimized = self.optimizer_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.optimizer_model["model"],
            temperature=self.optimizer_model["temperature"],
            max_tokens=self.optimizer_model["max_tokens"]
        )
        
        self.total_llm_calls += 1
        self.total_tokens_used += 800
        
        return optimized if optimized else self.current_method
    
    def run_optimization_cycle(self, sample_queries: List[Dict[str, Any]], num_optimizations: int = 3) -> str:
        """
        Run optimization cycle to find the best evolving prompt.
        Based on Auto Evol-Instruct paper Section 3.2-3.3.
        Optimizes e₀ → e₁ → e₂ → ... → e*
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔧 PROMPT OPTIMIZATION PHASE (Auto Evol-Instruct)")
        logger.info(f"{'='*70}")
        logger.info(f"Goal: Optimize Initial Evolving Method (e₀) → Optimal Method (e*)")
        logger.info(f"Queries available for testing: {len(sample_queries)}")
        
        test_sample_size = min(5, len(sample_queries))
        validation_sample_size = min(8, len(sample_queries))
        
        logger.info(f"Test sample size: {test_sample_size} queries")
        logger.info(f"Validation sample size: {validation_sample_size} queries")
        logger.info("-" * 70)
        
        logger.info("\n📊 STEP 1: Testing CURRENT baseline prompt (e₀)...")
        test_queries = random.sample(sample_queries, test_sample_size)
        
        baseline_trajectories = []
        for q in test_queries:
            traj = self.evolve_with_trajectory(q["query"], q["domain"], num_rounds=2)
            baseline_trajectories.append(traj)
        
        baseline_failure_rate = sum(t["failure_rate"] for t in baseline_trajectories) / len(baseline_trajectories)
        logger.info(f" Baseline (e₀) Failure Rate: {baseline_failure_rate:.1%}")
        
        logger.info(f"\n🔍 STEP 2: Analyzing trajectories to identify problems...")
        trajectory_analysis = self.analyze_trajectories(baseline_trajectories, sample_size=test_sample_size)
        
        actual_problems = set()
        for traj in baseline_trajectories:
            for vote in traj.get("voting_history", []):
                reasons = vote.get("function_reasons", [])
                for reason in reasons:
                    if "STAGNATION" in reason:
                        actual_problems.add("stagnation")
                    if "complexity" in reason.lower() or "depth" in reason.lower():
                        actual_problems.add("insufficient complexity increase")
                    if "paraphras" in reason.lower() or "bert" in reason.lower():
                        actual_problems.add("semantic similarity too high")
                    if "length" in reason.lower():
                        actual_problems.add("insufficient length increase")
                    if "grade" in reason.lower():
                        actual_problems.add("readability decreased")
        
        problems = list(actual_problems) if actual_problems else [
            "complexity unchanged", 
            "insufficient rewording", 
            "lacks analytical depth"
        ]
        
        logger.info(f"Problems identified: {', '.join(problems)}")
        
        logger.info(f"\n🧪 STEP 3: Generating {num_optimizations} candidate prompts (e₁ candidates)...")
        candidate_prompts = []
        
        for i in range(num_optimizations):
            logger.info(f"   Generating candidate {i+1}/{num_optimizations}...")
            optimized = self.optimize_evolving_method(trajectory_analysis, problems)
            if optimized and optimized != self.current_method:
                candidate_prompts.append(optimized)
        
        logger.info(f" Generated {len(candidate_prompts)} unique candidate prompts")
        
        if not candidate_prompts:
            logger.warning("No new candidate prompts generated, keeping baseline (e₀)")
            return self.current_method
        
        logger.info(f"\n🧪 STEP 4: Testing each candidate prompt...")
        logger.info("-" * 70)
        
        validation_queries = random.sample(sample_queries, validation_sample_size)
        
        all_methods = [
            {"name": "Baseline (e₀)", "method": self.current_method, "failure_rate": baseline_failure_rate}
        ]
        
        for i, candidate_method in enumerate(candidate_prompts):
            logger.info(f"\n📋 Testing Candidate {i+1}/{len(candidate_prompts)}...")
            
            original_method = self.current_method
            self.current_method = candidate_method
            
            test_trajectories = []
            for q in validation_queries:
                traj = self.evolve_with_trajectory(q["query"], q["domain"], num_rounds=2)
                test_trajectories.append(traj)
            
            failure_rate = sum(t["failure_rate"] for t in test_trajectories) / len(test_trajectories)
            
            all_methods.append({
                "name": f"Candidate {i+1} (e₁ option)",
                "method": candidate_method,
                "failure_rate": failure_rate
            })
            
            logger.info(f"   Failure Rate: {failure_rate:.1%}")
            
            self.current_method = original_method
        
        logger.info(f"\n🏆 STEP 5: Selecting best prompt (e*)...")
        logger.info("-" * 70)
        
        all_methods.sort(key=lambda x: x["failure_rate"])
        
        logger.info("\n📊 Results Summary:")
        for i, m in enumerate(all_methods):
            indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            logger.info(f"{indicator} {m['name']}: {m['failure_rate']:.1%} failure rate")
        
        best = all_methods[0]
        improvement = baseline_failure_rate - best["failure_rate"]
        
        logger.info(f"\n{'='*70}")
        if best["name"] != "Baseline (e₀)" and improvement > 0.01:
            self.current_method = best["method"]
            logger.info(f" ADOPTED: {best['name']} as e* (Improved by {improvement:.1%})")
            logger.info(f"   New failure rate: {best['failure_rate']:.1%} (was {baseline_failure_rate:.1%})")
        else:
            logger.info(f"⚪ KEEPING BASELINE (e₀): No significant improvement found")
            logger.info(f"   Baseline failure rate: {baseline_failure_rate:.1%}")
        
        logger.info(f"{'='*70}\n")
        
        return self.current_method
    
    def evolve_all_queries(self, queries: List[Dict[str, Any]], run_optimization: bool = True) -> List[Dict[str, Any]]:
        """Main evolution pipeline with metrics"""
        pipeline_start_time = time.time()
        
        logger.info(f"\n{'='*70}\nSTAGE 2: AUTO EVOL-INSTRUCT (OPTIMIZED VOTING SYSTEM)\n{'='*70}\nInput: {len(queries)} queries\n{'='*70}")
        
        queries = self.assign_difficulty_levels(queries)
        
        min_queries = self.config.get("min_queries_for_optimization", 10)
        enable_opt = self.config.get("enable_optimization", True)
        
        if run_optimization and enable_opt and len(queries) >= min_queries:
            queries_to_evolve = [q for q in queries if q["evolution_rounds"] > 0]
            if len(queries_to_evolve) >= min_queries:
                sample_size = min(30, len(queries_to_evolve))
                sample = random.sample(queries_to_evolve, sample_size)
                self.run_optimization_cycle(sample, self.config["num_optimizations"])
        elif run_optimization and len(queries) < min_queries:
            logger.info(f"\nSkipping optimization: Need at least {min_queries} queries, have {len(queries)}")
            logger.info("   Using Initial Evolving Method (e₀) for all queries\n")
        
        logger.info("\n🚀 EVOLUTION PHASE")
        logger.info("-" * 60)
        
        evolved_queries = []
        batch_size = self.config["evolution_batch_size"]
        total_batches = (len(queries) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(queries), batch_size):
            batch = queries[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logger.info(f"\nBatch {batch_num}/{total_batches} ({len(batch)} queries)")
            
            for i, q in enumerate(batch):
                num_rounds = q["evolution_rounds"]
                
                result = self.evolve_with_trajectory(
                    query=q["query"],
                    domain=q["domain"],
                    num_rounds=num_rounds
                )
                
                if num_rounds == 0:
                    evolution_successful = True
                else:
                    actually_evolved = result["final"] != q["query"]
                    low_failure_rate = result["failure_rate"] < 0.5
                    evolution_successful = actually_evolved and low_failure_rate
                
                evolved_queries.append({
                    "original_query": q["query"],
                    "evolved_query": result["final"],
                    "domain": q["domain"],
                    "source": q.get("source", "unknown"),
                    "target_difficulty": q["target_difficulty"],
                    "evolution_rounds": num_rounds,
                    "trajectory": result["trajectory"],
                    "failure_rate": result["failure_rate"],
                    "evolution_successful": evolution_successful,
                    "voting_history": result.get("voting_history", []),
                    "vote_agreement_rate": self._calculate_agreement_rate(result.get("voting_history", [])),
                    "total_time_seconds": result.get("total_time_seconds", 0),
                    "total_tokens_used": result.get("total_tokens", 0)
                })
                
                if (i + 1) % 5 == 0:
                    logger.info(f"  Processed {i+1}/{len(batch)} in batch")
            
            logger.info(f" Batch {batch_num} complete")
        
        self.evolved_queries = evolved_queries
        
        pipeline_time = time.time() - pipeline_start_time
        self.total_evolution_time = pipeline_time
        
        # Statistics
        successful = sum(1 for q in evolved_queries if q["evolution_successful"])
        easy_count = sum(1 for q in evolved_queries if q["target_difficulty"] == "easy")
        medium_count = sum(1 for q in evolved_queries if q["target_difficulty"] == "medium")
        hard_count = sum(1 for q in evolved_queries if q["target_difficulty"] == "hard")
        
        total_votes = self.vote_agreement_count + self.vote_disagreement_count
        agreement_percentage = (self.vote_agreement_count / total_votes * 100) if total_votes > 0 else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 2 COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total queries: {len(evolved_queries)}")
        logger.info(f"Successful evolutions: {successful} ({successful/len(evolved_queries)*100:.1f}%)")
        logger.info(f"\nDifficulty Breakdown:")
        logger.info(f"  Easy: {easy_count} ({easy_count/len(evolved_queries)*100:.1f}%)")
        logger.info(f"  Medium: {medium_count} ({medium_count/len(evolved_queries)*100:.1f}%)")
        logger.info(f"  Hard: {hard_count} ({hard_count/len(evolved_queries)*100:.1f}%)")
        logger.info(f"\n📊 Metrics:")
        logger.info(f"  Total LLM calls: {self.total_llm_calls}")
        logger.info(f"  Estimated tokens used: {self.total_tokens_used:,}")
        logger.info(f"  Total time: {pipeline_time:.1f} seconds ({pipeline_time/60:.1f} minutes)")
        logger.info(f"  Avg time per query: {pipeline_time/len(evolved_queries):.2f} seconds")
        logger.info(f"\n🗳️ Voting Statistics:")
        logger.info(f"  Agreements: {self.vote_agreement_count} ({agreement_percentage:.1f}%)")
        logger.info(f"  Disagreements: {self.vote_disagreement_count} ({100-agreement_percentage:.1f}%)")
        logger.info(f"{'='*70}\n")
        
        return evolved_queries
    
    def save_results(self, filepath: str = None):
        """Save results with metrics to JSON"""
        if not filepath:
            filepath = get_output_path("2")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        by_domain = defaultdict(int)
        by_difficulty = defaultdict(int)
        
        for q in self.evolved_queries:
            by_domain[q["domain"]] += 1
            by_difficulty[q["target_difficulty"]] += 1
        
        total_votes = self.vote_agreement_count + self.vote_disagreement_count
        
        output = {
            "stage": "auto_evol_instruct",
            "total_queries": len(self.evolved_queries),
            "queries_by_domain": dict(by_domain),
            "queries_by_difficulty": dict(by_difficulty),
            "difficulty_config": {
                "easy_percentage": DIFFICULTY_CONFIG["easy_percentage"],
                "medium_percentage": DIFFICULTY_CONFIG["medium_percentage"],
                "hard_percentage": DIFFICULTY_CONFIG["hard_percentage"]
            },
            "final_evolving_method": self.current_method,
            "metrics": {
                "total_llm_calls": self.total_llm_calls,
                "estimated_tokens_used": self.total_tokens_used,
                "total_evolution_time_seconds": round(self.total_evolution_time, 2),
                "average_time_per_query_seconds": round(self.total_evolution_time / len(self.evolved_queries), 2) if self.evolved_queries else 0,
                "vote_agreements": self.vote_agreement_count,
                "vote_disagreements": self.vote_disagreement_count,
                "vote_agreement_rate": round(self.vote_agreement_count / total_votes, 3) if total_votes > 0 else 0
            },
            "queries": self.evolved_queries
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filepath}")

def main():
    import sys
    stage1_output = get_output_path("1")
    
    if not os.path.exists(stage1_output):
        logger.error("Stage 1 output not found. Run stage1_self_instruct.py first")
        sys.exit(1)
    
    with open(stage1_output, 'r') as f:
        queries = json.load(f)["queries"]
    
    logger.info(f"Loaded {len(queries)} queries from Stage 1")
    
    auto_evol = AutoEvolInstruct(GroqClient(), GroqClient())
    evolved = auto_evol.evolve_all_queries(queries, run_optimization=True)

    auto_evol.save_results()
    
    logger.info(f"\n Stage 2 complete! {len(evolved)} queries with balanced difficulty")

if __name__ == "__main__":
    main()
