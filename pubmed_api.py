"""
Module for downloading papers from PubMed API.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Union
import requests
from Bio import Entrez
from config import PUBMED_EMAIL, PUBMED_TOOL, PUBMED_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PubMedAPI:
    """Class to interact with PubMed API using Biopython's Entrez module."""
    
    def __init__(self, email: str = PUBMED_EMAIL, tool: str = PUBMED_TOOL, api_key: str = PUBMED_API_KEY):
        """
        Initialize the PubMedAPI class.
        
        Args:
            email: Email to use for PubMed API
            tool: Tool name to use for PubMed API
            api_key: API key for PubMed API (optional)
        """
        Entrez.email = email
        Entrez.tool = tool
        if api_key:
            Entrez.api_key = api_key
        
        self.rate_limit_delay = 0.34  # Default delay between requests (3 requests per second)
        if api_key:
            self.rate_limit_delay = 0.1  # With API key, 10 requests per second
    
    def search_papers(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search for papers in PubMed.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PubMed IDs
        """
        logger.info(f"Searching PubMed for: {query}")
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            pmids = record["IdList"]
            logger.info(f"Found {len(pmids)} papers")
            return pmids
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def fetch_paper_details(self, pmid: str) -> Optional[Dict]:
        """
        Fetch details for a single paper from PubMed.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Dictionary with paper details or None if error
        """
        try:
            time.sleep(self.rate_limit_delay)  # Respect rate limits
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            if not records["PubmedArticle"]:
                logger.warning(f"No data found for PMID {pmid}")
                return None
                
            article = records["PubmedArticle"][0]
            
            # Extract basic information
            paper_data = {
                "pmid": pmid,
                "title": self._extract_title(article),
                "abstract": self._extract_abstract(article),
                "authors": self._extract_authors(article),
                "journal": self._extract_journal(article),
                "publication_date": self._extract_publication_date(article),
                "keywords": self._extract_keywords(article),
                "doi": self._extract_doi(article),
                "mesh_terms": self._extract_mesh_terms(article),
                "publication_types": self._extract_publication_types(article),
                "affiliations": self._extract_affiliations(article),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
            return paper_data
        except Exception as e:
            logger.error(f"Error fetching details for PMID {pmid}: {e}")
            return None
    
    def fetch_papers_batch(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch details for multiple papers from PubMed.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of dictionaries with paper details
        """
        logger.info(f"Fetching details for {len(pmids)} papers")
        papers = []
        
        for i, pmid in enumerate(pmids):
            logger.info(f"Fetching paper {i+1}/{len(pmids)}: PMID {pmid}")
            paper_data = self.fetch_paper_details(pmid)
            if paper_data:
                papers.append(paper_data)
        
        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers
    
    def save_papers_to_json(self, papers: List[Dict], filename: str) -> None:
        """
        Save papers to a JSON file.
        
        Args:
            papers: List of paper dictionaries
            filename: Filename to save to
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(papers, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(papers)} papers to {filename}")
        except Exception as e:
            logger.error(f"Error saving papers to {filename}: {e}")
    
    def load_papers_from_json(self, filename: str) -> List[Dict]:
        """
        Load papers from a JSON file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            List of paper dictionaries
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            logger.info(f"Loaded {len(papers)} papers from {filename}")
            return papers
        except Exception as e:
            logger.error(f"Error loading papers from {filename}: {e}")
            return []
    
    # Helper methods to extract specific information from PubMed records
    
    def _extract_title(self, article: Dict) -> str:
        """Extract title from PubMed article."""
        try:
            return article["MedlineCitation"]["Article"]["ArticleTitle"]
        except KeyError:
            return ""
    
    def _extract_abstract(self, article: Dict) -> str:
        """Extract abstract from PubMed article."""
        try:
            abstract_parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            if isinstance(abstract_parts, list):
                # Some abstracts are divided into sections
                abstract = " ".join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)
            return abstract
        except KeyError:
            return ""
    
    def _extract_authors(self, article: Dict) -> List[Dict]:
        """Extract authors from PubMed article."""
        authors = []
        try:
            author_list = article["MedlineCitation"]["Article"]["AuthorList"]
            for author in author_list:
                if "LastName" in author and "ForeName" in author:
                    authors.append({
                        "last_name": author["LastName"],
                        "fore_name": author["ForeName"],
                        "full_name": f"{author['ForeName']} {author['LastName']}"
                    })
            return authors
        except KeyError:
            return []
    
    def _extract_journal(self, article: Dict) -> Dict:
        """Extract journal information from PubMed article."""
        journal_info = {}
        try:
            journal = article["MedlineCitation"]["Article"]["Journal"]
            journal_info["name"] = journal.get("Title", "")
            journal_info["iso_abbreviation"] = journal.get("ISOAbbreviation", "")
            
            # Extract journal metadata if available
            if "ISSN" in journal:
                journal_info["issn"] = journal["ISSN"]
            
            return journal_info
        except KeyError:
            return {"name": "", "iso_abbreviation": ""}
    
    def _extract_publication_date(self, article: Dict) -> Dict:
        """Extract publication date from PubMed article."""
        pub_date = {}
        try:
            date_info = article["MedlineCitation"]["Article"]["Journal"]["JournalIssue"]["PubDate"]
            
            # Extract year, month, day if available
            if "Year" in date_info:
                pub_date["year"] = date_info["Year"]
            if "Month" in date_info:
                pub_date["month"] = date_info["Month"]
            if "Day" in date_info:
                pub_date["day"] = date_info["Day"]
                
            # Create formatted date string
            date_str = ""
            if "year" in pub_date:
                date_str = pub_date["year"]
                if "month" in pub_date:
                    date_str += f"-{pub_date['month']}"
                    if "day" in pub_date:
                        date_str += f"-{pub_date['day']}"
            
            pub_date["date_string"] = date_str
            
            return pub_date
        except KeyError:
            return {"year": "", "date_string": ""}
    
    def _extract_keywords(self, article: Dict) -> List[str]:
        """Extract keywords from PubMed article."""
        keywords = []
        try:
            keyword_list = article["MedlineCitation"]["KeywordList"][0]
            keywords = [keyword for keyword in keyword_list]
            return keywords
        except (KeyError, IndexError):
            return []
    
    def _extract_doi(self, article: Dict) -> str:
        """Extract DOI from PubMed article."""
        try:
            article_id_list = article["PubmedData"]["ArticleIdList"]
            for article_id in article_id_list:
                if article_id.attributes.get("IdType") == "doi":
                    return str(article_id)
            return ""
        except KeyError:
            return ""
    
    def _extract_mesh_terms(self, article: Dict) -> List[str]:
        """Extract MeSH terms from PubMed article."""
        mesh_terms = []
        try:
            mesh_heading_list = article["MedlineCitation"]["MeshHeadingList"]
            for mesh_heading in mesh_heading_list:
                descriptor = mesh_heading["DescriptorName"]
                mesh_terms.append(str(descriptor))
            return mesh_terms
        except KeyError:
            return []
    
    def _extract_publication_types(self, article: Dict) -> List[str]:
        """Extract publication types from PubMed article."""
        pub_types = []
        try:
            pub_type_list = article["MedlineCitation"]["Article"]["PublicationTypeList"]
            pub_types = [str(pub_type) for pub_type in pub_type_list]
            return pub_types
        except KeyError:
            return []
    
    def _extract_affiliations(self, article: Dict) -> List[str]:
        """Extract author affiliations from PubMed article."""
        affiliations = []
        try:
            author_list = article["MedlineCitation"]["Article"]["AuthorList"]
            for author in author_list:
                if "AffiliationInfo" in author:
                    for affiliation in author["AffiliationInfo"]:
                        if "Affiliation" in affiliation and affiliation["Affiliation"] not in affiliations:
                            affiliations.append(affiliation["Affiliation"])
            return affiliations
        except KeyError:
            return []

