"""
Configuration module for the Taiwan Human Biobank Literature Analysis Tool.
"""

# PubMed API configuration
PUBMED_API_KEY = ""  # Optional: Your PubMed API key for higher rate limits
PUBMED_EMAIL = "your.email@example.com"  # Required for PubMed API
PUBMED_TOOL = "TaiwanBiobankAnalyzer"

# Search configuration
DEFAULT_SEARCH_TERM = "Taiwan Human Biobank OR Taiwan Biobank OR Taiwanese Biobank"
DEFAULT_MAX_RESULTS = 100

# Analysis configuration
DEFAULT_NUM_CLUSTERS = 5
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_TOP_KEYWORDS = 20
DEFAULT_TOP_JOURNALS = 10
DEFAULT_TOP_AUTHORS = 15

# Output configuration
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_FILENAME_PREFIX = "taiwan_biobank_analysis"

# NLP configuration
STOPWORDS = [
    "the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", 
    "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", 
    "at", "as", "your", "all", "have", "new", "more", "an", "was", "we", 
    "will", "can", "us", "about", "if", "my", "has", "but", "our", "one", 
    "other", "do", "no", "they", "he", "she", "study", "research", "analysis",
    "result", "results", "method", "methods", "conclusion", "conclusions",
    "background", "objective", "objectives", "aim", "aims", "purpose",
    "data", "patient", "patients", "subject", "subjects", "participant",
    "participants", "taiwan", "taiwanese", "biobank", "human"
]

