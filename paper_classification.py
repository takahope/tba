"""
Module for classifying papers.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from config import DEFAULT_NUM_CLUSTERS, DEFAULT_MIN_CLUSTER_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperClassification:
    """Class for classifying papers."""
    
    def __init__(self, n_clusters: int = DEFAULT_NUM_CLUSTERS, min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE):
        """
        Initialize the PaperClassification class.
        
        Args:
            n_clusters: Number of clusters for K-means
            min_cluster_size: Minimum cluster size for DBSCAN
        """
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.kmeans_model = None
        self.dbscan_model = None
        self.feature_names = None
        self.tfidf_matrix = None
        self.pca_model = None
        self.svd_model = None
    
    def prepare_data(self, papers: List[Dict]) -> np.ndarray:
        """
        Prepare data for classification.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            TF-IDF matrix
        """
        # Extract preprocessed text from papers
        texts = []
        for paper in papers:
            analysis = paper.get('analysis', {})
            preprocessed_text = analysis.get('preprocessed_text', '')
            if not preprocessed_text:
                # Fallback to title and abstract if preprocessed text is not available
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                preprocessed_text = f"{title} {abstract}"
            texts.append(preprocessed_text)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
        return self.tfidf_matrix
    
    def cluster_kmeans(self, tfidf_matrix: np.ndarray) -> List[int]:
        """
        Cluster papers using K-means.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            List of cluster labels
        """
        logger.info(f"Clustering papers using K-means with {self.n_clusters} clusters")
        
        # Normalize the TF-IDF matrix
        normalized_matrix = normalize(tfidf_matrix)
        
        # Apply K-means clustering
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(normalized_matrix)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(normalized_matrix, cluster_labels)
            logger.info(f"Silhouette score: {silhouette_avg:.4f}")
        
        return cluster_labels.tolist()
    
    def cluster_dbscan(self, tfidf_matrix: np.ndarray) -> List[int]:
        """
        Cluster papers using DBSCAN.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            List of cluster labels
        """
        logger.info("Clustering papers using DBSCAN")
        
        # Normalize the TF-IDF matrix
        normalized_matrix = normalize(tfidf_matrix)
        
        # Apply DBSCAN clustering
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
        cluster_labels = self.dbscan_model.fit_predict(normalized_matrix)
        
        # Count number of clusters (excluding noise points with label -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        logger.info(f"DBSCAN found {n_clusters} clusters")
        
        return cluster_labels.tolist()
    
    def reduce_dimensions(self, tfidf_matrix: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions of TF-IDF matrix.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            method: Dimensionality reduction method ('pca' or 'svd')
            n_components: Number of components
            
        Returns:
            Reduced matrix
        """
        logger.info(f"Reducing dimensions using {method.upper()} to {n_components} components")
        
        if method == 'pca':
            # Convert sparse matrix to dense for PCA
            dense_matrix = tfidf_matrix.toarray()
            self.pca_model = PCA(n_components=n_components, random_state=42)
            reduced_matrix = self.pca_model.fit_transform(dense_matrix)
        elif method == 'svd':
            # Use TruncatedSVD for sparse matrices
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = self.svd_model.fit_transform(tfidf_matrix)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        logger.info(f"Reduced matrix shape: {reduced_matrix.shape}")
        return reduced_matrix
    
    def get_cluster_keywords(self, cluster_labels: List[int], top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top keywords for each cluster.
        
        Args:
            cluster_labels: List of cluster labels
            top_n: Number of top keywords to return per cluster
            
        Returns:
            Dictionary mapping cluster IDs to lists of (keyword, importance) tuples
        """
        if self.kmeans_model is None or self.feature_names is None:
            logger.error("K-means model or feature names not available")
            return {}
        
        cluster_keywords = {}
        
        # Get cluster centers from K-means
        cluster_centers = self.kmeans_model.cluster_centers_
        
        # For each cluster, get the top keywords
        for cluster_id in range(self.n_clusters):
            # Get the indices of the top features for this cluster
            ordered_indices = np.argsort(cluster_centers[cluster_id])[::-1]
            top_indices = ordered_indices[:top_n]
            
            # Get the feature names and their importance scores
            keywords = [(self.feature_names[i], cluster_centers[cluster_id][i]) for i in top_indices]
            
            # Store the keywords for this cluster
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords
    
    def classify_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Classify papers using clustering.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of paper dictionaries with classification results
        """
        logger.info(f"Classifying {len(papers)} papers")
        
        # Prepare data
        tfidf_matrix = self.prepare_data(papers)
        
        # Cluster papers using K-means
        kmeans_labels = self.cluster_kmeans(tfidf_matrix)
        
        # Cluster papers using DBSCAN
        dbscan_labels = self.cluster_dbscan(tfidf_matrix)
        
        # Reduce dimensions for visualization
        reduced_data = self.reduce_dimensions(tfidf_matrix, method='svd')
        
        # Get cluster keywords
        cluster_keywords = self.get_cluster_keywords(kmeans_labels)
        
        # Add classification results to papers
        classified_papers = []
        for i, paper in enumerate(papers):
            classification = {
                "kmeans_cluster": kmeans_labels[i],
                "dbscan_cluster": dbscan_labels[i],
                "coordinates": reduced_data[i].tolist()
            }
            
            # Add cluster keywords if available
            kmeans_cluster = kmeans_labels[i]
            if kmeans_cluster in cluster_keywords:
                classification["cluster_keywords"] = cluster_keywords[kmeans_cluster]
            
            # Combine original paper data with classification
            classified_paper = {**paper, "classification": classification}
            classified_papers.append(classified_paper)
        
        logger.info(f"Completed classification of {len(classified_papers)} papers")
        return classified_papers

