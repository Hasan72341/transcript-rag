"""
Seed Query Generator
Generates initial seed queries using LLM for each domain
"""

from utils.llm_client import GroqClient
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

SEED_PROMPT = """Generate 5 analytical queries for conversation intelligence in the {domain} domain.

Requirements:
- Must be answerable from call transcripts only
- Focus on behavioral patterns, causal analysis, or process insights
- 10-25 words each
- Avoid SQL, IDs, or external data references

IMPORTANT: Return ONLY a JSON array of query strings (no subcategories needed).

Format:
[
  "What patterns in agent responses correlate with customer satisfaction?",
  "How do wait times impact call escalation rates?",
  "Which greeting styles lead to faster issue resolution?",
  "What causes customers to request supervisors?",
  "How does sentiment shift during policy explanation calls?"
]

Respond with ONLY the JSON array of strings."""

class SeedGenerator:
    
    def __init__(self, client):
        self.client = client
        os.makedirs("output", exist_ok=True)
    
    def generate_seeds(self, domain, count=5):
        """Generate seeds for one domain - returns list of plain strings"""
        prompt = SEED_PROMPT.format(domain=domain)
        
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Lower temperature for more consistent JSON
            max_tokens=800
        )
        
        if not response:
            logger.error(f"Failed to generate seeds for {domain}")
            return []
        
        # Extract JSON array of strings
        seeds = self._extract_string_array(response)
        
        if not seeds:
            logger.error(f"JSON parse failed for {domain}. Raw response:")
            logger.error(response[:500])
            return []
        
        return seeds[:count]

    def _extract_string_array(self, response: str) -> list:
        """Extract array of strings from LLM response"""
        
        strategies = [
            # Direct parse
            lambda r: json.loads(r),
            # Array in code block
            lambda r: json.loads(re.search(r'```(?:json)?\s*(.*?)\s*```', r, re.DOTALL).group(1)),
            # Any array
            lambda r: json.loads(re.search(r'\[[\s\S]*?\]', r).group()),
        ]
        
        for strategy in strategies:
            try:
                result = strategy(response)
                # Validate it's a list of strings
                if isinstance(result, list) and all(isinstance(item, str) for item in result):
                    # Filter out empty strings
                    return [s.strip() for s in result if s.strip()]
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        return []
    
    def generate_all_domains(self, domains):
        """Generate seeds for all domains"""
        all_seeds = {}
        
        for domain in domains:
            logger.info(f"Generating seeds for {domain}...")
            seeds = self.generate_seeds(domain)
            
            if seeds:
                all_seeds[domain] = seeds
                logger.info(f"  Generated {len(seeds)} seeds")
            else:
                logger.warning(f"  No seeds generated for {domain}")
        
        return all_seeds
    
    def save_seeds(self, seeds, filepath="output/generated_seeds.json"):
        """Save seeds to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(seeds, f, indent=2)
        
        logger.info(f"Saved seeds to {filepath}")
    
    def load_seeds(self, filepath="output/generated_seeds.json"):
        """Load seeds from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Seeds not found: {filepath}")
        
        with open(filepath) as f:
            return json.load(f)

if __name__ == "__main__":
    import logging
    from config import DOMAINS
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize client
    client = GroqClient()
    generator = SeedGenerator(client)
    
    # Generate seeds
    print("Generating seed queries...")
    seeds = generator.generate_all_domains(DOMAINS)
    
    # Save seeds
    generator.save_seeds(seeds)
    
    # Print summary
    total = sum(len(queries) for queries in seeds.values())
    print(f"\n Generated {total} seeds across {len(seeds)} domains")
    print(f"Saved to: output/generated_seeds.json")
