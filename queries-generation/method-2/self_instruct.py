"""
Stage 1: Self-Instruct Query Generation
USES SEEDS FROM seed_generator.py (NOT seed_data.py)
"""

import json
import random
import logging
from typing import List, Dict, Any
from collections import defaultdict, Counter
import re
import statistics
import os

from utils.llm_client import GroqClient
from utils.prompts import format_self_instruct_prompt
from config import SELF_INSTRUCT_CONFIG, MODEL_CONFIG, get_output_path, TARGET_QUERIES_STAGE1

# Embedding-based metrics
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_generated_seeds(filepath: str = "output/generated_seeds.json") -> Dict[str, List[Dict]]:
    """Load seeds from generated_seeds.json"""
    if not os.path.exists(filepath):
        logger.error(f" Seeds file not found: {filepath}")
        logger.error("Please run: python seed_generator.py first")
        raise FileNotFoundError(f"Seeds file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        seeds = json.load(f)
    
    logger.info(f" Loaded seeds from {filepath}")
    return seeds


def get_seed_queries(domain: str, seeds_data: Dict) -> List[str]:
    """Get seed queries for a domain from generated seeds"""
    if domain not in seeds_data:
        logger.warning(f" No seeds found for domain: {domain}")
        return []
    
    # Seeds are now plain strings, not dicts
    domain_seeds = seeds_data[domain]
    
    # Handle both formats for backward compatibility
    if isinstance(domain_seeds, list):
        if len(domain_seeds) > 0:
            # Check first item to determine format
            if isinstance(domain_seeds[0], dict):
                # Old format: [{"query": "...", "subcategory": "..."}]
                return [seed['query'] for seed in domain_seeds]
            elif isinstance(domain_seeds[0], str):
                # New format: ["query1", "query2", ...]
                return domain_seeds
    
    return []


def get_domains(seeds_data: Dict) -> List[str]:
    """Get list of available domains from generated seeds"""
    return list(seeds_data.keys())


class SelfInstructGenerator:
    """Self-Instruct query generation with subcategory tracking and metrics"""
    
    def __init__(self, client: GroqClient, seeds_filepath: str = "output/generated_seeds.json"):
        self.client = client
        self.config = SELF_INSTRUCT_CONFIG
        self.model_config = MODEL_CONFIG["self_instruct"]
        self.all_queries: List[Dict[str, Any]] = []
        self.queries_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.queries_by_subcategory: Dict[str, int] = defaultdict(int)
        
        #  Load generated seeds
        self.seeds_data = load_generated_seeds(seeds_filepath)
        
        # Valid subcategories
        self.valid_subcategories = {
            "Business Outcome Events",
            "Customer Frustration & Sentiment Drivers",
            "Agent Behavior & Response Quality",
            "Process & Workflow Breakdowns",
            "Information Gaps & Misalignment",
            "Policy / Compliance / Eligibility Friction",
            "Operational Delays & Inefficiencies",
            "Comparative Variation",
            "Anomalies & Outlier Patterns"
        }
        
        # Lazy-loaded embedding model
        self._embedding_model = None
    
    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------
    
    def normalize_seed_query(self, query_input) -> str:
        """
        Normalize seed query to plain string
        Seeds are ALWAYS plain strings (no subcategories)
        """
        if isinstance(query_input, dict):
            return query_input.get("query", "")
        return str(query_input)
    
    def select_prompt_examples(self, seed_queries: List, generated_queries: List[Dict], num_examples: int = 3) -> List:
        """Select examples for prompting"""
        pool: List[Any] = []
        
        # Add recent generated queries
        recent = generated_queries[-20:] if len(generated_queries) > 20 else generated_queries
        pool.extend(recent)
        
        # Add seed queries as plain strings
        normalized_seeds = [self.normalize_seed_query(sq) for sq in seed_queries]
        pool.extend(normalized_seeds)
        
        if len(pool) <= num_examples:
            return pool
        
        return random.sample(pool, num_examples)
    
    def parse_llm_output(self, output: str, domain: str) -> List[Dict[str, str]]:
        """Parse LLM output to extract queries with subcategories"""
        if not output:
            return []
        
        queries: List[Dict[str, str]] = []
        
        try:
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "query" in item:
                            query_text = item["query"].strip()
                            subcategory = item.get("subcategory", "Unknown")
                            
                            if subcategory not in self.valid_subcategories:
                                logger.debug(f"Invalid subcategory '{subcategory}', setting to 'Unknown'")
                                subcategory = "Unknown"
                            
                            queries.append({
                                "query": query_text,
                                "subcategory": subcategory,
                                "domain": domain
                            })
                        elif isinstance(item, str):
                            queries.append({
                                "query": item.strip(),
                                "subcategory": "Unknown",
                                "domain": domain
                            })
                    
                    return queries
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        # Fallback: extract plain text queries
        for line in output.split('\n'):
            line = line.strip()
            line = re.sub(r'^[\d\.\-\*]\s*', '', line)
            quoted = re.findall(r'"([^"]+)"', line)
            if quoted:
                for q in quoted:
                    queries.append({
                        "query": q.strip(),
                        "subcategory": "Unknown",
                        "domain": domain
                    })
            elif line and len(line) > 10:
                queries.append({
                    "query": line,
                    "subcategory": "Unknown",
                    "domain": domain
                })
        
        return queries
    
    def filter_query(self, query: str) -> bool:
        """Basic filtering for generated queries"""
        if len(query) < self.config["min_query_length"]:
            return False
        if len(query) > self.config["max_query_length"]:
            return False
        
        meta_phrases = ["here are", "example", "generate", "query 1", "query 2", "as follows"]
        query_lower = query.lower()
        if any(phrase in query_lower for phrase in meta_phrases):
            return False
        
        return True
    
    def generate_for_domain(self, domain: str, target_count: int = 60) -> List[Dict[str, Any]]:
        """Generate queries for a specific domain using Self-Instruct"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating queries for {domain}")
        logger.info(f"Target: {target_count} NEW queries (excluding seeds)")
        logger.info(f"{'='*60}")
        
        #  Get seed queries from loaded generated seeds
        seed_queries = get_seed_queries(domain, self.seeds_data)
        logger.info(f"Starting with {len(seed_queries)} seed queries (from seed_generator.py)")
        
        if not seed_queries:
            logger.error(f" No seeds available for {domain}. Cannot proceed.")
            return []
        
        generated_queries: List[Dict[str, Any]] = []
        all_query_texts = set()
        
        # Add seed query texts to dedupe set
        for sq in seed_queries:
            normalized_text = self.normalize_seed_query(sq)
            all_query_texts.add(normalized_text)
        
        iteration = 0
        max_iterations = 50
        
        while len(generated_queries) < target_count and iteration < max_iterations:
            iteration += 1
            
            # Select examples
            examples = self.select_prompt_examples(
                seed_queries,
                generated_queries,
                num_examples=self.config["num_prompt_instructions"]
            )
            
            # Format prompt
            prompt = format_self_instruct_prompt(
                domain=domain,
                examples=examples,
                num_to_generate=self.config["num_instructions_per_seed"]
            )
            
            logger.info(f"Iteration {iteration}: Generating batch...")
            
            # Call LLM
            output = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"]
            )
            
            if not output:
                logger.warning("Generation failed, skipping iteration")
                continue
            
            # Parse output
            new_queries = self.parse_llm_output(output, domain)
            logger.info(f"Parsed {len(new_queries)} queries from output")
            
            added = 0
            for query_obj in new_queries:
                if len(generated_queries) >= target_count:
                    logger.info(f" Target reached ({target_count}), stopping batch processing")
                    break
                
                query_text = query_obj["query"]
                subcategory = query_obj["subcategory"]
                
                if not self.filter_query(query_text):
                    continue
                
                if query_text in all_query_texts:
                    continue
                
                # Similarity check
                too_similar = False
                for existing in list(all_query_texts)[-50:]:
                    if self._simple_similarity(query_text, existing) > 0.8:
                        too_similar = True
                        break
                
                if too_similar:
                    continue
                
                # Add query
                generated_queries.append({
                    "query": query_text,
                    "domain": domain,
                    "subcategory": subcategory,
                    "source": "self_instruct",
                    "iteration": iteration
                })
                all_query_texts.add(query_text)
                self.queries_by_subcategory[subcategory] += 1
                added += 1
            
            logger.info(f"Added {added} new queries (total: {len(generated_queries)}/{target_count})")
            
            if len(generated_queries) >= target_count:
                logger.info(f"Target of {target_count} NEW queries reached!")
                break
        
        logger.info(f"\n Generated {len(generated_queries)} NEW queries for {domain}")
        
        # Log subcategory distribution
        domain_subcats = defaultdict(int)
        for q in generated_queries:
            domain_subcats[q["subcategory"]] += 1
        
        logger.info(f"Subcategory distribution for {domain}:")
        for subcat, count in sorted(domain_subcats.items(), key=lambda x: -x[1]):
            logger.info(f"  {subcat}: {count}")
        
        return generated_queries
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Simple word-overlap similarity"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def generate_all_domains(self) -> List[Dict[str, Any]]:
        """Generate queries for all domains"""
        #  Get domains from loaded seeds
        domains = get_domains(self.seeds_data)
        target_per_domain = TARGET_QUERIES_STAGE1 // len(domains)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 1: SELF-INSTRUCT GENERATION (USING GENERATED SEEDS)")
        logger.info(f"{'='*70}")
        logger.info(f"Domains: {len(domains)}")
        logger.info(f"Target per domain: {target_per_domain} NEW queries")
        logger.info(f"Total target: {TARGET_QUERIES_STAGE1} NEW queries")
        logger.info(f"{'='*70}\n")
        
        all_queries: List[Dict[str, Any]] = []
        
        for domain in domains:
            queries = self.generate_for_domain(domain, target_per_domain)
            all_queries.extend(queries)
            self.queries_by_domain[domain] = queries
        
        self.all_queries = all_queries
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STAGE 1 COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total NEW queries generated: {len(all_queries)}")
        logger.info(f"\nBreakdown by domain:")
        for domain, queries in self.queries_by_domain.items():
            logger.info(f"  {domain}: {len(queries)}")
        
        logger.info(f"{'='*70}\n")
        
        return all_queries
    
    # ---------------------------------------------------------------------
    # Metrics: non-LLM + embeddings (Stage 1 evaluation)
    # ---------------------------------------------------------------------
    
    def _get_embedding_model(self):
        """Lazy-load the sentence embedding model."""
        if self._embedding_model is None:
            logger.info("Loading sentence embedding model for metrics...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model
    
    def evaluate_metrics(self):
        """Evaluate Stage 1 dataset metrics"""
        if not self.all_queries:
            logger.warning("No queries available for metrics evaluation.")
            return
        
        logger.info(f"\n{'='*70}")
        logger.info("STAGE 1 DATASET METRICS")
        logger.info(f"{'='*70}")
        
        queries = [q["query"] for q in self.all_queries]
        total = len(queries)
        
        # ----------------- Length distribution -----------------
        lengths = [len(q.split()) for q in queries]
        mean_len = statistics.mean(lengths)
        std_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        min_len = min(lengths)
        max_len = max(lengths)
        min_target = self.config.get("min_query_length", 7)
        max_target = self.config.get("max_query_length", 35)
        within_band = sum(min_target <= l <= max_target for l in lengths) / total
        
        logger.info("\n[Length Distribution]")
        logger.info(f"  Total queries: {total}")
        logger.info(f"  Mean length: {mean_len:.2f} words (std={std_len:.2f})")
        logger.info(f"  Min / Max length: {min_len} / {max_len} words")
        logger.info(f"  Within [{min_target}, {max_target}] words: {within_band*100:.1f}%")
        
        # ----------------- Lexical self-similarity (Jaccard) -----------------
        logger.info("\n[Lexical Self-Similarity (Jaccard over words)]")
        sample_size = min(50, total)
        sample_indices = random.sample(range(total), sample_size)
        sim_values: List[float] = []
        
        for i_idx in range(sample_size):
            for j_idx in range(i_idx + 1, sample_size):
                q1 = queries[sample_indices[i_idx]]
                q2 = queries[sample_indices[j_idx]]
                sim = self._simple_similarity(q1, q2)
                sim_values.append(sim)
        
        avg_sim = statistics.mean(sim_values) if sim_values else 0.0
        logger.info(f"  Sample size: {sample_size}")
        logger.info(f"  Average pairwise similarity: {avg_sim:.3f}")
        logger.info(f"  Lexical diversity (1 - avg_sim): {1-avg_sim:.3f}")
        
        # ----------------- N-gram uniqueness (trigrams) -----------------
        logger.info("\n[N-gram Uniqueness (trigrams)]")
        all_trigrams = []
        for q in queries:
            tokens = q.lower().split()
            if len(tokens) < 3:
                continue
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            all_trigrams.extend(trigrams)
        
        if all_trigrams:
            trigram_counts = Counter(all_trigrams)
            unique_trigrams = sum(1 for _, c in trigram_counts.items() if c == 1)
            total_trigrams = len(all_trigrams)
            uniqueness_ratio = unique_trigrams / total_trigrams
            logger.info(f"  Total trigrams: {total_trigrams}")
            logger.info(f"  Unique trigrams: {unique_trigrams}")
            logger.info(f"  Uniqueness ratio: {uniqueness_ratio*100:.1f}%")
        else:
            logger.info("  Not enough tokens to compute trigrams.")
        
        # ----------------- Vocabulary richness (TTR) -----------------
        logger.info("\n[Vocabulary Richness (Type-Token Ratio)]")
        all_tokens = []
        for q in queries:
            all_tokens.extend(q.lower().split())
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Unique tokens: {unique_tokens}")
        logger.info(f"  TTR (unique / total): {ttr:.3f}")
        
        # ----------------- Question starter distribution -----------------
        logger.info("\n[Question Starter Distribution]")
        starter_counts: Counter = Counter()
        for q in queries:
            parts = q.strip().split()
            if not parts:
                continue
            starter = parts[0].strip().rstrip("?:,.").lower()
            starter_counts[starter] += 1
        
        if starter_counts:
            for starter, count in starter_counts.most_common(10):
                logger.info(f"  '{starter}': {count} ({count/total*100:.1f}%)")
        else:
            logger.info("  No valid starters found.")
        
        # ----------------- Embedding-based semantic metrics -----------------
        logger.info("\n[Semantic Similarity (Embeddings)]")
        model = self._get_embedding_model()
        embeddings = model.encode(queries, show_progress_bar=False)
        
        # To keep it cheap, sample pairs as before
        sem_sims: List[float] = []
        high_sim_threshold = 0.8
        high_sim_pairs = 0
        
        for i_idx in range(sample_size):
            for j_idx in range(i_idx + 1, sample_size):
                i = sample_indices[i_idx]
                j = sample_indices[j_idx]
                sim = float(cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0])
                sem_sims.append(sim)
                if sim >= high_sim_threshold:
                    high_sim_pairs += 1
        
        avg_sem_sim = statistics.mean(sem_sims) if sem_sims else 0.0
        total_pairs = len(sem_sims)
        high_sim_rate = high_sim_pairs / total_pairs if total_pairs > 0 else 0.0
        
        logger.info(f"  Average semantic similarity (sample): {avg_sem_sim:.3f}")
        logger.info(f"  Pairs with sim >= {high_sim_threshold}: {high_sim_pairs}/{total_pairs} ({high_sim_rate*100:.1f}%)")
        logger.info(f"  Semantic diversity (1 - avg_sem_sim): {1-avg_sem_sim:.3f}")

         # ----------------- Semantic outliers (embeddings) -----------------
        logger.info("\n[Semantic Outliers (Embeddings)]")
        # Compute centroid of all embeddings
        centroid = embeddings.mean(axis=0).reshape(1, -1)
        
        distances: List[tuple] = []
        for idx, emb in enumerate(embeddings):
            # Similarity to centroid
            sim_to_centroid = float(cosine_similarity(
                emb.reshape(1, -1),
                centroid
            )[0][0])
            dist = 1.0 - sim_to_centroid  # higher = farther from center
            distances.append((dist, idx))
        
        # Sort by distance descending (farthest first)
        distances.sort(reverse=True, key=lambda x: x[0])
        
        # Simple stats
        dist_values = [d for d, _ in distances]
        mean_dist = statistics.mean(dist_values)
        std_dist = statistics.pstdev(dist_values) if len(dist_values) > 1 else 0.0
        logger.info(f"  Mean distance to centroid: {mean_dist:.3f}")
        logger.info(f"  Std distance to centroid: {std_dist:.3f}")
        
        # Show top-K potential outliers
        top_k = min(5, len(distances))
        logger.info(f"  Top {top_k} farthest queries from centroid (potential outliers):")
        for rank in range(top_k):
            dist, idx = distances[rank]
            q = queries[idx]
            truncated = (q[:120] + "…") if len(q) > 120 else q
            logger.info(f"    #{rank+1} dist={dist:.3f}: {truncated}")

        
        # ----------------- Domain coverage -----------------
        logger.info("\n[Domain Coverage]")
        for domain, qs in self.queries_by_domain.items():
            logger.info(f"  {domain}: {len(qs)} ({len(qs)/total*100:.1f}%)")
        
        # ----------------- Subcategory balance -----------------
        logger.info("\n[Subcategory Balance]")
        subcat_counts = [self.queries_by_subcategory.get(k, 0) for k in sorted(self.valid_subcategories)]
        total_subcat = sum(subcat_counts)
        if total_subcat == 0:
            logger.info("  No subcategory data available.")
        else:
            mean_subcat = statistics.mean(subcat_counts)
            std_subcat = statistics.pstdev(subcat_counts) if len(subcat_counts) > 1 else 0.0
            cv = std_subcat / mean_subcat if mean_subcat > 0 else 0.0
            logger.info("  Counts per subcategory (sorted by name):")
            for name in sorted(self.valid_subcategories):
                count = self.queries_by_subcategory.get(name, 0)
                logger.info(f"    {name}: {count} ({count/total*100:.1f}%)")
            logger.info(f"  Mean per subcategory: {mean_subcat:.2f}")
            logger.info(f"  Std per subcategory: {std_subcat:.2f}")
            logger.info(f"  Coefficient of variation (CV): {cv:.3f}")
        
        logger.info(f"\n{'='*70}")
        logger.info("END OF STAGE 1 METRICS")
        logger.info(f"{'='*70}\n")
    
    def save_results(self, filepath: str = None):
        """Save generated queries to JSON file"""
        if not filepath:
            filepath = get_output_path("1")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        output = {
            "stage": "self_instruct",
            "total_queries": len(self.all_queries),
            "queries_by_domain": {
                domain: len(queries) for domain, queries in self.queries_by_domain.items()
            },
            "queries_by_subcategory": dict(self.queries_by_subcategory),
            "valid_subcategories": list(self.valid_subcategories),
            "queries": self.all_queries
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {filepath}")
    
    def save_seeds(self, seeds, filepath="output/generated_seeds.json"):
        """
        Save seeds to file
        Format: {"domain": ["query1", "query2", ...], ...}
        NO subcategories - those are assigned by self-instruct
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(seeds, f, indent=2)
        
        logger.info(f"Saved seeds to {filepath}")


def main():
    """Main execution function"""
    #  Check if seeds exist
    seeds_path = "output/generated_seeds.json"
    if not os.path.exists(seeds_path):
        logger.error(f"❌ Seeds file not found: {seeds_path}")
        logger.error("Please run: python seed_generator.py first")
        return
    
    client = GroqClient()
    generator = SelfInstructGenerator(client, seeds_filepath=seeds_path)
    queries = generator.generate_all_domains()
    
    # Run metrics
    generator.evaluate_metrics()
    
    generator.save_results()
    logger.info(f"\n Stage 1 complete! Generated {len(queries)} NEW queries")


if __name__ == "__main__":
    main()
