"""
Module for analyzing papers.
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter, defaultdict
from config import STOPWORDS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

class PaperAnalysis:
    """Class for analyzing papers."""
    
    def __init__(self, custom_stopwords: List[str] = None):
        """
        Initialize the PaperAnalysis class.
        
        Args:
            custom_stopwords: Additional stopwords to use
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Combine NLTK stopwords with custom stopwords
        self.stopwords = set(stopwords.words('english'))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        self.stopwords.update(STOPWORDS)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        tokens = self.preprocess_text(text)
        keyword_freq = Counter(tokens)
        return keyword_freq.most_common(top_n)
    
    def analyze_paper(self, paper: Dict) -> Dict:
        """
        Analyze a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Dictionary with analysis results
        """
        # Combine title and abstract for text analysis
        full_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        
        # Extract keywords
        keywords = self.extract_keywords(full_text)
        
        # Count sentences in abstract
        abstract = paper.get('abstract', '')
        sentences = sent_tokenize(abstract) if abstract else []
        
        # Analyze author information
        authors = paper.get('authors', [])
        num_authors = len(authors)
        
        # Extract publication year
        pub_date = paper.get('publication_date', {})
        pub_year = pub_date.get('year', '')
        
        # Create analysis results
        analysis = {
            "pmid": paper.get('pmid', ''),
            "extracted_keywords": keywords,
            "num_keywords": len(keywords),
            "abstract_length": len(abstract),
            "abstract_sentences": len(sentences),
            "num_authors": num_authors,
            "publication_year": pub_year,
            "has_mesh_terms": len(paper.get('mesh_terms', [])) > 0,
            "preprocessed_text": " ".join(self.preprocess_text(full_text))
        }
        
        return analysis
    
    def analyze_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Analyze multiple papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of dictionaries with analysis results
        """
        logger.info(f"Analyzing {len(papers)} papers")
        analyzed_papers = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Analyzing paper {i+1}/{len(papers)}: {paper.get('pmid', '')}")
            analysis = self.analyze_paper(paper)
            
            # Combine original paper data with analysis
            analyzed_paper = {**paper, "analysis": analysis}
            analyzed_papers.append(analyzed_paper)
        
        logger.info(f"Completed analysis of {len(analyzed_papers)} papers")
        return analyzed_papers
    
    def extract_common_keywords(self, papers: List[Dict], top_n: int = 30) -> List[Tuple[str, int]]:
        """
        Extract common keywords across all papers.
        
        Args:
            papers: List of paper dictionaries
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        all_keywords = []
        
        for paper in papers:
            analysis = paper.get('analysis', {})
            keywords = [kw for kw, _ in analysis.get('extracted_keywords', [])]
            all_keywords.extend(keywords)
        
        keyword_freq = Counter(all_keywords)
        return keyword_freq.most_common(top_n)
    
    def extract_research_topics(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Extract research topics from papers based on MeSH terms.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary mapping topics to frequency
        """
        topics = []
        
        for paper in papers:
            mesh_terms = paper.get('mesh_terms', [])
            topics.extend(mesh_terms)
        
        topic_freq = Counter(topics)
        return dict(topic_freq)
    
    def extract_publication_trends(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Extract publication trends by year.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary mapping years to publication count
        """
        years = []
        
        for paper in papers:
            pub_date = paper.get('publication_date', {})
            year = pub_date.get('year', '')
            if year:
                years.append(year)
        
        year_freq = Counter(years)
        return dict(sorted(year_freq.items()))
    
    def extract_journal_distribution(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Extract journal distribution.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary mapping journals to publication count
        """
        journals = []
        
        for paper in papers:
            journal = paper.get('journal', {})
            journal_name = journal.get('name', '')
            if journal_name:
                journals.append(journal_name)
        
        journal_freq = Counter(journals)
        return dict(journal_freq)
    
    def extract_author_network(self, papers: List[Dict]) -> Dict[str, Dict]:
        """
        Extract author collaboration network.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary with author network information
        """
        # Count papers per author
        author_papers = defaultdict(int)
        # Track collaborations between authors
        collaborations = defaultdict(set)
        
        for paper in papers:
            authors = paper.get('authors', [])
            author_names = [author.get('full_name', '') for author in authors if author.get('full_name')]
            
            # Update paper count for each author
            for author in author_names:
                author_papers[author] += 1
            
            # Update collaborations
            for i, author1 in enumerate(author_names):
                for author2 in author_names[i+1:]:
                    collaborations[author1].add(author2)
                    collaborations[author2].add(author1)
        
        # Convert sets to lists for JSON serialization
        collaborations_list = {author: list(collabs) for author, collabs in collaborations.items()}
        
        # Calculate network metrics
        network = {
            "author_papers": author_papers,
            "collaborations": collaborations_list,
            "top_authors": dict(Counter(author_papers).most_common(20)),
            "collaboration_counts": {author: len(collabs) for author, collabs in collaborations.items()}
        }
        
        return network
    
    def extract_affiliation_distribution(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Extract affiliation distribution.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary mapping affiliations to frequency
        """
        all_affiliations = []
        
        for paper in papers:
            affiliations = paper.get('affiliations', [])
            all_affiliations.extend(affiliations)
        
        # Clean and normalize affiliations
        cleaned_affiliations = []
        for affiliation in all_affiliations:
            # Extract institution name (usually before the first comma)
            parts = affiliation.split(',')
            if parts:
                institution = parts[0].strip()
                cleaned_affiliations.append(institution)
        
        affiliation_freq = Counter(cleaned_affiliations)
        return dict(affiliation_freq)

