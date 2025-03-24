"""
Utility functions for the Taiwan Human Biobank Literature Analysis Tool.
"""

import os
import logging
import json
import argparse
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from wordcloud import WordCloud
from config import DEFAULT_SEARCH_TERM, DEFAULT_MAX_RESULTS, DEFAULT_OUTPUT_DIR, DEFAULT_NUM_CLUSTERS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Taiwan Human Biobank Literature Analysis Tool')
    
    parser.add_argument('--search', type=str, default=DEFAULT_SEARCH_TERM,
                        help=f'Search term (default: "{DEFAULT_SEARCH_TERM}")')
    
    parser.add_argument('--limit', type=int, default=DEFAULT_MAX_RESULTS,
                        help=f'Maximum number of papers to download (default: {DEFAULT_MAX_RESULTS})')
    
    parser.add_argument('--load', type=str, default=None,
                        help='Load papers from JSON file instead of downloading')
    
    parser.add_argument('--save', type=str, default=None,
                        help='Save downloaded papers to JSON file')
    
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: "{DEFAULT_OUTPUT_DIR}")')
    
    parser.add_argument('--clusters', type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f'Number of clusters (default: {DEFAULT_NUM_CLUSTERS})')
    
    return parser.parse_args()

def save_json(data: Any, filename: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filename: Filename to save to
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved data to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")

def load_json(filename: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filename: Filename to load from
        
    Returns:
        Loaded data
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}")
        return None

def create_wordcloud(text_data: str, width: int = 800, height: int = 400) -> WordCloud:
    """
    Create word cloud from text data.
    
    Args:
        text_data: Text data to create word cloud from
        width: Width of word cloud
        height: Height of word cloud
        
    Returns:
        WordCloud object
    """
    wordcloud = WordCloud(width=width, height=height, background_color='white',
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(text_data)
    return wordcloud

def plot_wordcloud(wordcloud: WordCloud, title: str = 'Word Cloud', figsize: tuple = (12, 8)) -> Figure:
    """
    Plot word cloud.
    
    Args:
        wordcloud: WordCloud object
        title: Title of plot
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return fig

def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory to ensure exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def format_author_list(authors: List[Dict]) -> str:
    """
    Format list of authors.
    
    Args:
        authors: List of author dictionaries
        
    Returns:
        Formatted author list
    """
    author_names = [author.get('full_name', '') for author in authors if author.get('full_name')]
    if len(author_names) <= 3:
        return ', '.join(author_names)
    else:
        return f"{author_names[0]} et al."

def extract_year(pub_date: Dict) -> Optional[int]:
    """
    Extract year from publication date.
    
    Args:
        pub_date: Publication date dictionary
        
    Returns:
        Year as integer or None
    """
    year_str = pub_date.get('year', '')
    if year_str:
        try:
            return int(year_str)
        except ValueError:
            return None
    return None

