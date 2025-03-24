"""
Module for statistical analysis of papers.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import networkx as nx
from wordcloud import WordCloud
from config import DEFAULT_OUTPUT_DIR, DEFAULT_FILENAME_PREFIX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Statistic:
    """Class for statistical analysis of papers."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Initialize the Statistic class.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.2)
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def create_dataframe(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            DataFrame with paper data
        """
        logger.info(f"Creating DataFrame from {len(papers)} papers")
        
        # Extract basic paper information
        data = []
        for paper in papers:
            # Extract basic information
            pmid = paper.get('pmid', '')
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            journal_info = paper.get('journal', {})
            journal_name = journal_info.get('name', '')
            pub_date = paper.get('publication_date', {})
            pub_year = pub_date.get('year', '')
            
            # Extract author information
            authors = paper.get('authors', [])
            author_names = [author.get('full_name', '') for author in authors if author.get('full_name')]
            author_count = len(author_names)
            first_author = author_names[0] if author_names else ''
            
            # Extract analysis information
            analysis = paper.get('analysis', {})
            abstract_length = analysis.get('abstract_length', 0)
            abstract_sentences = analysis.get('abstract_sentences', 0)
            
            # Extract classification information
            classification = paper.get('classification', {})
            kmeans_cluster = classification.get('kmeans_cluster', -1)
            dbscan_cluster = classification.get('dbscan_cluster', -1)
            
            # Create row
            row = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal_name,
                'publication_year': pub_year,
                'author_count': author_count,
                'first_author': first_author,
                'abstract_length': abstract_length,
                'abstract_sentences': abstract_sentences,
                'kmeans_cluster': kmeans_cluster,
                'dbscan_cluster': dbscan_cluster,
                'url': paper.get('url', '')
            }
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert publication year to numeric
        df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from DataFrame.
        
        Args:
            df: DataFrame with paper data
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary statistics")
        
        # Basic statistics
        total_papers = len(df)
        year_range = (df['publication_year'].min(), df['publication_year'].max())
        journals_count = df['journal'].nunique()
        authors_count = df['first_author'].nunique()
        
        # Publication trends
        papers_by_year = df['publication_year'].value_counts().sort_index().to_dict()
        
        # Journal distribution
        top_journals = df['journal'].value_counts().head(10).to_dict()
        
        # Cluster distribution
        kmeans_clusters = df['kmeans_cluster'].value_counts().to_dict()
        dbscan_clusters = df['dbscan_cluster'].value_counts().to_dict()
        
        # Create summary statistics
        summary = {
            'total_papers': total_papers,
            'year_range': year_range,
            'journals_count': journals_count,
            'authors_count': authors_count,
            'papers_by_year': papers_by_year,
            'top_journals': top_journals,
            'kmeans_clusters': kmeans_clusters,
            'dbscan_clusters': dbscan_clusters
        }
        
        return summary
    
    def plot_publication_trend(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot publication trend by year.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the plot
        """
        logger.info("Plotting publication trend")
        
        # Count papers by year
        papers_by_year = df['publication_year'].value_counts().sort_index()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = papers_by_year.plot(kind='bar', color='skyblue')
        
        # Add trend line
        years = papers_by_year.index.astype(float)
        counts = papers_by_year.values
        z = np.polyfit(years, counts, 1)
        p = np.poly1d(z)
        plt.plot(range(len(years)), p(years), "r--", linewidth=2)
        
        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.title('Publication Trend of Taiwan Human Biobank Research')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + 0.5, str(int(v)), ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, 'publication_trend.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved publication trend plot to {self.output_dir}/publication_trend.png")
        
        plt.close()
    
    def plot_journal_distribution(self, df: pd.DataFrame, top_n: int = 10, save: bool = True) -> None:
        """
        Plot journal distribution.
        
        Args:
            df: DataFrame with paper data
            top_n: Number of top journals to show
            save: Whether to save the plot
        """
        logger.info(f"Plotting journal distribution (top {top_n})")
        
        # Count papers by journal
        journal_counts = df['journal'].value_counts().head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = journal_counts.plot(kind='barh', color='lightgreen')
        
        # Add labels and title
        plt.xlabel('Number of Publications')
        plt.ylabel('Journal')
        plt.title(f'Top {top_n} Journals Publishing Taiwan Human Biobank Research')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(journal_counts.values):
            ax.text(v + 0.5, i, str(int(v)), va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, 'journal_distribution.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved journal distribution plot to {self.output_dir}/journal_distribution.png")
        
        plt.close()
    
    def plot_cluster_distribution(self, df: pd.DataFrame, cluster_type: str = 'kmeans', save: bool = True) -> None:
        """
        Plot cluster distribution.
        
        Args:
            df: DataFrame with paper data
            cluster_type: Type of clustering ('kmeans' or 'dbscan')
            save: Whether to save the plot
        """
        logger.info(f"Plotting {cluster_type} cluster distribution")
        
        # Get cluster column
        cluster_col = f'{cluster_type}_cluster'
        if cluster_col not in df.columns:
            logger.error(f"Cluster column '{cluster_col}' not found in DataFrame")
            return
        
        # Count papers by cluster
        cluster_counts = df[cluster_col].value_counts().sort_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = cluster_counts.plot(kind='bar', color='salmon')
        
        # Add labels and title
        plt.xlabel('Cluster')
        plt.ylabel('Number of Papers')
        plt.title(f'Distribution of Papers by {cluster_type.upper()} Cluster')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(cluster_counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, f'{cluster_type}_cluster_distribution.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved {cluster_type} cluster distribution plot to {self.output_dir}/{cluster_type}_cluster_distribution.png")
        
        plt.close()
    
    def plot_cluster_scatter(self, papers: List[Dict], save: bool = True) -> None:
        """
        Plot scatter plot of papers by cluster.
        
        Args:
            papers: List of paper dictionaries
            save: Whether to save the plot
        """
        logger.info("Plotting cluster scatter plot")
        
        # Extract coordinates and cluster labels
        coordinates = []
        kmeans_clusters = []
        titles = []
        
        for paper in papers:
            classification = paper.get('classification', {})
            coords = classification.get('coordinates', [0, 0])
            if len(coords) >= 2:
                coordinates.append(coords[:2])  # Use first two dimensions
                kmeans_clusters.append(classification.get('kmeans_cluster', -1))
                titles.append(paper.get('title', ''))
        
        if not coordinates:
            logger.error("No coordinates found for scatter plot")
            return
        
        # Convert to numpy arrays
        coordinates = np.array(coordinates)
        kmeans_clusters = np.array(kmeans_clusters)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans_clusters, 
                   cmap='viridis', alpha=0.7, s=100, edgecolors='w')
        
        # Add labels and title
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Clustering of Taiwan Human Biobank Papers')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Add annotations for a few points
        for i in range(min(10, len(coordinates))):
            plt.annotate(titles[i][:30] + '...', 
                        xy=(coordinates[i, 0], coordinates[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, 'cluster_scatter.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster scatter plot to {self.output_dir}/cluster_scatter.png")
        
        plt.close()
    
    def plot_keyword_wordcloud(self, papers: List[Dict], save: bool = True) -> None:
        """
        Plot word cloud of keywords.
        
        Args:
            papers: List of paper dictionaries
            save: Whether to save the plot
        """
        logger.info("Plotting keyword word cloud")
        
        # Extract keywords
        keywords = []
        for paper in papers:
            analysis = paper.get('analysis', {})
            extracted_keywords = analysis.get('extracted_keywords', [])
            keywords.extend([kw for kw, _ in extracted_keywords])
        
        if not keywords:
            logger.error("No keywords found for word cloud")
            return
        
        # Count keywords
        keyword_counts = Counter(keywords)
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate_from_frequencies(keyword_counts)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Keywords in Taiwan Human Biobank Research', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, 'keyword_wordcloud.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved keyword word cloud to {self.output_dir}/keyword_wordcloud.png")
        
        plt.close()
    
    def plot_author_network(self, papers: List[Dict], top_n: int = 30, save: bool = True) -> None:
        """
        Plot author collaboration network.
        
        Args:
            papers: List of paper dictionaries
            top_n: Number of top authors to include
            save: Whether to save the plot
        """
        logger.info(f"Plotting author network (top {top_n} authors)")
        
        # Extract author collaborations
        author_papers = defaultdict(int)
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
        
        # Get top authors by paper count
        top_authors = [author for author, _ in Counter(author_papers).most_common(top_n)]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (authors)
        for author in top_authors:
            G.add_node(author, papers=author_papers[author])
        
        # Add edges (collaborations)
        for author1 in top_authors:
            for author2 in collaborations[author1]:
                if author2 in top_authors:
                    G.add_edge(author1, author2)
        
        # Create plot
        plt.figure(figsize=(14, 12))
        
        # Set node size based on number of papers
        node_size = [author_papers[author] * 100 for author in G.nodes()]
        
        # Set node color based on degree (number of collaborators)
        node_color = [len(collaborations[author]) for author in G.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, 
                              alpha=0.8, cmap='viridis')
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.0)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        # Add title and colorbar
        plt.title('Author Collaboration Network in Taiwan Human Biobank Research', fontsize=16)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(node_color), max(node_color)))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Number of Collaborators')
        
        # Remove axis
        plt.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save:
            plt.savefig(os.path.join(self.output_dir, 'author_network.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Saved author network plot to {self.output_dir}/author_network.png")
        
        plt.close()
    
    def export_to_excel(self, df: pd.DataFrame, filename: str = None) -> None:
        """
        Export DataFrame to Excel.
        
        Args:
            df: DataFrame to export
            filename: Filename to save to
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{DEFAULT_FILENAME_PREFIX}.xlsx")
        
        logger.info(f"Exporting data to Excel: {filename}")
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Papers', index=False)
            
            # Write summary statistics
            summary = self.generate_summary_statistics(df)
            
            # Create summary DataFrame
            summary_data = {
                'Metric': [
                    'Total Papers',
                    'Year Range',
                    'Number of Journals',
                    'Number of Authors'
                ],
                'Value': [
                    summary['total_papers'],
                    f"{summary['year_range'][0]} - {summary['year_range'][1]}",
                    summary['journals_count'],
                    summary['authors_count']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create papers by year DataFrame
            years_df = pd.DataFrame({
                'Year': list(summary['papers_by_year'].keys()),
                'Papers': list(summary['papers_by_year'].values())
            })
            years_df.to_excel(writer, sheet_name='Papers by Year', index=False)
            
            # Create journals DataFrame
            journals_df = pd.DataFrame({
                'Journal': list(summary['top_journals'].keys()),
                'Papers': list(summary['top_journals'].values())
            })
            journals_df.to_excel(writer, sheet_name='Top Journals', index=False)
            
            # Create clusters DataFrame
            clusters_df = pd.DataFrame({
                'Cluster': list(summary['kmeans_clusters'].keys()),
                'Papers': list(summary['kmeans_clusters'].values())
            })
            clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
        
        logger.info(f"Exported data to Excel: {filename}")
    
    def generate_html_report(self, df: pd.DataFrame, papers: List[Dict], filename: str = None) -> None:
        """
        Generate HTML report.
        
        Args:
            df: DataFrame with paper data
            papers: List of paper dictionaries
            filename: Filename to save to
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f"{DEFAULT_FILENAME_PREFIX}.html")
        
        logger.info(f"Generating HTML report: {filename}")
        
        # Generate summary statistics
        summary = self.generate_summary_statistics(df)
        
        # Extract cluster keywords
        cluster_keywords = {}
        for paper in papers:
            classification = paper.get('classification', {})
            cluster = classification.get('kmeans_cluster', -1)
            if 'cluster_keywords' in classification and cluster not in cluster_keywords:
                cluster_keywords[cluster] = classification['cluster_keywords']
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Taiwan Human Biobank Literature Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .chart {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .cluster {{
                    margin-bottom: 30px;
                }}
                .keyword {{
                    display: inline-block;
                    background-color: #e9ecef;
                    padding: 5px 10px;
                    margin: 5px;
                    border-radius: 15px;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Taiwan Human Biobank Literature Analysis</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total Papers: <strong>{summary['total_papers']}</strong></p>
                    <p>Year Range: <strong>{summary['year_range'][0]} - {summary['year_range'][1]}</strong></p>
                    <p>Number of Journals: <strong>{summary['journals_count']}</strong></p>
                    <p>Number of Authors: <strong>{summary['authors_count']}</strong></p>
                </div>
                
                <h2>Publication Trends</h2>
                <div class="chart">
                    <img src="publication_trend.png" alt="Publication Trend">
                </div>
                
                <h2>Journal Distribution</h2>
                <div class="chart">
                    <img src="journal_distribution.png" alt="Journal Distribution">
                </div>
                
                <h2>Keyword Analysis</h2>
                <div class="chart">
                    <img src="keyword_wordcloud.png" alt="Keyword Word Cloud">
                </div>
                
                <h2>Cluster Analysis</h2>
                <div class="chart">
                    <img src="cluster_scatter.png" alt="Cluster Scatter Plot">
                </div>
                <div class="chart">
                    <img src="kmeans_cluster_distribution.png" alt="Cluster Distribution">
                </div>
                
                <h3>Cluster Keywords</h3>
        """
        
        # Add cluster keywords
        for cluster, keywords in cluster_keywords.items():
            html_content += f"""
                <div class="cluster">
                    <h4>Cluster {cluster}</h4>
                    <div>
            """
            
            for keyword, score in keywords:
                html_content += f"""
                        <span class="keyword">{keyword} ({score:.4f})</span>
                """
            
            html_content += """
                    </div>
                </div>
            """
        
        # Add top papers table
        html_content += """
                <h2>Top Papers</h2>
                <table>
                    <tr>
                        <th>Title</th>
                        <th>Journal</th>
                        <th>Year</th>
                        <th>Cluster</th>
                    </tr>
        """
        
        # Add top 20 papers
        for _, row in df.head(20).iterrows():
            html_content += f"""
                    <tr>
                        <td><a href="{row['url']}" target="_blank">{row['title']}</a></td>
                        <td>{row['journal']}</td>
                        <td>{row['publication_year']}</td>
                        <td>{row['kmeans_cluster']}</td>
                    </tr>
            """
        
        # Close table and add footer
        html_content += """
                </table>
                
                <div class="footer">
                    <p>Generated by Taiwan Human Biobank Literature Analysis Tool</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {filename}")
    
    def run_analysis(self, papers: List[Dict]) -> None:
        """
        Run full statistical analysis.
        
        Args:
            papers: List of paper dictionaries
        """
        logger.info("Running full statistical analysis")
        
        # Create DataFrame
        df = self.create_dataframe(papers)
        
        # Generate plots
        self.plot_publication_trend(df)
        self.plot_journal_distribution(df)
        self.plot_cluster_distribution(df, cluster_type='kmeans')
        self.plot_cluster_scatter(papers)
        self.plot_keyword_wordcloud(papers)
        self.plot_author_network(papers)
        
        # Export to Excel
        self.export_to_excel(df)
        
        # Generate HTML report
        self.generate_html_report(df, papers)
        
        logger.info("Completed statistical analysis")

