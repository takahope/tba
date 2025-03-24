"""
Main script for the Taiwan Human Biobank Literature Analysis Tool.
"""

import os
import logging
import sys
from typing import Dict, List, Any, Optional

from pubmed_api import PubMedAPI
from paper_analysis import PaperAnalysis
from paper_classification import PaperClassification
from statistic import Statistic
from utils import parse_arguments, ensure_dir, save_json, load_json
from config import DEFAULT_FILENAME_PREFIX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('taiwan_biobank_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    ensure_dir(args.output)
    
    # Initialize PubMed API
    pubmed_api = PubMedAPI()
    
    # Load or download papers
    if args.load:
        logger.info(f"Loading papers from {args.load}")
        papers = pubmed_api.load_papers_from_json(args.load)
        if not papers:
            logger.error(f"Failed to load papers from {args.load}")
            return
    else:
        logger.info(f"Searching for papers with query: {args.search}")
        pmids = pubmed_api.search_papers(args.search, max_results=args.limit)
        if not pmids:
            logger.error("No papers found")
            return
        
        logger.info(f"Downloading {len(pmids)} papers")
        papers = pubmed_api.fetch_papers_batch(pmids)
        
        # Save papers if requested
        if args.save:
            logger.info(f"Saving papers to {args.save}")
            pubmed_api.save_papers_to_json(papers, args.save)
    
    logger.info(f"Loaded {len(papers)} papers")
    
    # Analyze papers
    logger.info("Analyzing papers")
    paper_analysis = PaperAnalysis()
    analyzed_papers = paper_analysis.analyze_papers(papers)
    
    # Classify papers
    logger.info("Classifying papers")
    paper_classification = PaperClassification(n_clusters=args.clusters)
    classified_papers = paper_classification.classify_papers(analyzed_papers)
    
    # Run statistical analysis
    logger.info("Running statistical analysis")
    statistic = Statistic(output_dir=args.output)
    statistic.run_analysis(classified_papers)
    
    # Save final results
    final_output_file = os.path.join(args.output, f"{DEFAULT_FILENAME_PREFIX}_results.json")
    save_json(classified_papers, final_output_file)
    
    logger.info(f"Analysis complete. Results saved to {args.output}")
    logger.info(f"Excel file: {os.path.join(args.output, DEFAULT_FILENAME_PREFIX + '.xlsx')}")
    logger.info(f"HTML report: {os.path.join(args.output, DEFAULT_FILENAME_PREFIX + '.html')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)

