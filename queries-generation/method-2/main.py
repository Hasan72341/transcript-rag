"""
Main orchestrator for Query Generation Pipeline
Runs Stage 1 (Self-Instruct) and Stage 2 (Auto Evol-Instruct)
"""

import os
import sys
import logging
import argparse
from datetime import datetime

from utils.llm_client import GroqClient
from self_instruct import SelfInstructGenerator
from auto_evol import AutoEvolInstruct
from config import get_output_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_stage1(client: GroqClient, output_path: str = None):
    """Run Stage 1: Self-Instruct Generation"""
    logger.info("\n" + "="*80)
    logger.info("STARTING STAGE 1: SELF-INSTRUCT GENERATION")
    logger.info("="*80 + "\n")
    
    generator = SelfInstructGenerator(client)
    queries = generator.generate_all_domains()
    generator.save_results(output_path)
    
    return queries


def run_stage2(
    evol_client: GroqClient,
    optimizer_client: GroqClient,
    input_path: str = None,
    output_path: str = None,
    run_optimization: bool = True
):
    """Run Stage 2: Auto Evol-Instruct Transformation"""
    logger.info("\n" + "="*80)
    logger.info("STARTING STAGE 2: AUTO EVOL-INSTRUCT TRANSFORMATION")
    logger.info("="*80 + "\n")
    
    if not input_path:
        input_path = get_output_path("1")
    
    if not os.path.exists(input_path):
        logger.error(f"Stage 1 output not found: {input_path}")
        logger.error("Please run Stage 1 first or provide correct input path")
        return None
    
    import json
    with open(input_path, 'r') as f:
        stage1_data = json.load(f)
    
    queries = stage1_data["queries"]
    logger.info(f"Loaded {len(queries)} queries from Stage 1\n")
    
    auto_evol = AutoEvolInstruct(evol_client, optimizer_client)
    evolved_queries = auto_evol.evolve_all_queries(queries, run_optimization)
    auto_evol.save_results(output_path)
    
    return evolved_queries


def run_full_pipeline(
    run_optimization: bool = True,
    stage1_only: bool = False,
    stage2_only: bool = False
):
    """Run complete pipeline or specific stages"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        logger.error("Please set it using: export GROQ_API_KEY=your_key_here")
        sys.exit(1)
    
    client = GroqClient(api_key)
    
    start_time = datetime.now()
    
    try:
        if not stage2_only:
            stage1_queries = run_stage1(client)
            logger.info(f" Stage 1 completed: {len(stage1_queries)} queries generated\n")
        
        if not stage1_only:
            evol_client = GroqClient(api_key)
            optimizer_client = GroqClient(api_key)
            
            stage2_queries = run_stage2(
                evol_client,
                optimizer_client,
                run_optimization=run_optimization
            )
            
            if stage2_queries:
                logger.info(f" Stage 2 completed: {len(stage2_queries)} queries evolved\n")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {duration}")
        logger.info(f"Output directory: ./output/")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Query Generation Pipeline: Self-Instruct + Auto Evol-Instruct"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Which stage to run: 1 (Self-Instruct), 2 (Auto Evol), or all (default: all)"
    )
    
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Skip optimization cycle in Stage 2 (faster but lower quality)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Groq API key (or set GROQ_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["GROQ_API_KEY"] = args.api_key
    
    stage1_only = args.stage == "1"
    stage2_only = args.stage == "2"
    run_optimization = not args.no_optimization
    
    run_full_pipeline(
        run_optimization=run_optimization,
        stage1_only=stage1_only,
        stage2_only=stage2_only
    )


if __name__ == "__main__":
    main()
