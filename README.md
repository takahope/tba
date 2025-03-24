# Literature Analysis Tool

A comprehensive tool for downloading, analyzing, and visualizing scientific literature related to the Taiwan Human Biobank.

## Features

- Automated literature retrieval from PubMed
- Text analysis and keyword extraction
- Paper classification using machine learning
- Statistical analysis and visualization
- Interactive HTML report generation
- Excel export for further analysis

## Installation

1. Clone this repository:
   \`\`\`
   git clone https://github.com/yourusername/taiwan-biobank-analyzer.git
   cd taiwan-biobank-analyzer
   \`\`\`

2. Install the required dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Download NLTK resources (first time only):
   \`\`\`python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   \`\`\`

## Usage

### Basic Usage

Run the tool with default settings:

\`\`\`
python main.py
\`\`\`

This will:
1. Search PubMed for papers related to the Taiwan Human Biobank
2. Download and analyze up to 100 papers
3. Classify the papers into 5 clusters
4. Generate visualizations and reports in the `output` directory

### Command Line Arguments

The tool supports several command line arguments:

- `--search`: Search query (default: "Taiwan Human Biobank OR Taiwan Biobank OR Taiwanese Biobank")
- `--limit`: Maximum number of papers to download (default: 100)
- `--load`: Load papers from a JSON file instead of downloading
- `--save`: Save downloaded papers to a JSON file
- `--output`: Output directory (default: "output")
- `--clusters`: Number of clusters for classification (default: 5)

### Examples

Search for a specific topic within the Taiwan Biobank literature:

\`\`\`
python main.py --search "Taiwan Biobank AND (genetics OR genomics)" --limit 50
\`\`\`

Load previously downloaded papers and re-analyze them:

\`\`\`
python main.py --load papers.json --clusters 8
\`\`\`

Download papers and save them for later use:

\`\`\`
python main.py --save papers.json --output results
\`\`\`

## Output Files

The tool generates the following output files:

1. **Excel file**: Contains detailed information about all papers and analysis results
2. **HTML report**: A user-friendly summary of the analysis with visualizations
3. **Visualizations**: Multiple plots showing publication trends, journal distribution, clusters, etc.
4. **JSON file**: Raw data for further analysis

## Modules

- `config.py`: Configuration settings
- `pubmed_api.py`: PubMed API integration for paper retrieval
- `paper_analysis.py`: Text analysis and keyword extraction
- `paper_classification.py`: Machine learning-based paper classification
- `statistic.py`: Statistical analysis and visualization
- `utils.py`: Utility functions
- `main.py`: Main script

## Advanced Usage

### Custom Analysis

You can modify the `PaperAnalysis` class to extract different information from papers:

\`\`\`python
# Example: Add a method to analyze citation patterns
def analyze_citation_patterns(self, papers):
    # Your custom analysis code here
    pass
\`\`\`

### Different Classification Methods

Modify the `PaperClassification` class to use different algorithms:

\`\`\`python
# Example: Use hierarchical clustering instead of K-means
from sklearn.cluster import AgglomerativeClustering

def cluster_hierarchical(self, tfidf_matrix):
    model = AgglomerativeClustering(n_clusters=self.n_clusters)
    return model.fit_predict(tfidf_matrix.toarray()).tolist()
\`\`\`

### Additional Visualizations

Add custom visualization functions to the `Statistic` class:

\`\`\`python
# Example: Add a method to visualize citation networks
def plot_citation_network(self, papers, save=True):
    # Your custom visualization code here
    pass
\`\`\`

## Project Structure

\`\`\`
taiwan-biobank-analyzer/
├── config.py                # Configuration settings
├── main.py                  # Main script
├── paper_analysis.py        # Text analysis module
├── paper_classification.py  # Classification module
├── pubmed_api.py            # PubMed API module
├── requirements.txt         # Dependencies
├── statistic.py             # Statistical analysis module
├── utils.py                 # Utility functions
└── output/                  # Generated output (created at runtime)
    ├── taiwan_biobank_analysis.xlsx     # Excel report
    ├── taiwan_biobank_analysis.html     # HTML report
    ├── publication_trend.png            # Trend visualization
    ├── journal_distribution.png         # Journal visualization
    ├── keyword_wordcloud.png            # Keyword visualization
    ├── cluster_scatter.png              # Cluster visualization
    ├── kmeans_cluster_distribution.png  # Cluster distribution
    └── author_network.png               # Author network visualization
\`\`\`

## Detailed Module Descriptions

### config.py

Contains configuration parameters for the entire application:
- PubMed API settings
- Search parameters
- Analysis settings
- Output settings
- NLP configuration (stopwords, etc.)

### pubmed_api.py

Handles all interactions with the PubMed API:
- Searching for papers
- Downloading paper details
- Extracting metadata (authors, journals, dates, etc.)
- Saving and loading papers from JSON files

### paper_analysis.py

Performs text analysis on the downloaded papers:
- Text preprocessing (tokenization, stopword removal, etc.)
- Keyword extraction
- Topic identification
- Publication trend analysis
- Author network analysis

### paper_classification.py

Classifies papers using machine learning techniques:
- TF-IDF vectorization
- K-means clustering
- DBSCAN clustering
- Dimensionality reduction (PCA, SVD)
- Cluster analysis and keyword extraction

### statistic.py

Generates statistics and visualizations:
- Summary statistics
- Publication trend plots
- Journal distribution plots
- Cluster visualizations
- Keyword word clouds
- Author network graphs
- Excel and HTML report generation

### utils.py

Contains utility functions used across the application:
- Command line argument parsing
- File I/O operations
- Directory management
- Text formatting
- Visualization helpers

### main.py

The main script that orchestrates the entire process:
- Parses command line arguments
- Coordinates the workflow
- Handles error logging
- Manages the overall execution

## Troubleshooting

### Common Issues

1. **PubMed API Rate Limits**

   **Problem**: Receiving errors when downloading many papers.
   
   **Solution**: Add your email and API key in `config.py` to increase rate limits, or use the `--limit` option to reduce the number of papers.

2. **NLTK Resource Errors**

   **Problem**: Errors about missing NLTK resources.
   
   **Solution**: Run the NLTK download commands mentioned in the Installation section.

3. **Memory Issues with Large Datasets**

   **Problem**: Out of memory errors when processing many papers.
   
   **Solution**: Use the `--limit` option to reduce the number of papers, or process papers in batches using the `--load` and `--save` options.

4. **Visualization Errors**

   **Problem**: Matplotlib errors when generating plots.
   
   **Solution**: Ensure you have a working display or use a non-interactive backend by adding `matplotlib.use('Agg')` at the top of your script.

### Debugging

For detailed debugging information, check the log file:

\`\`\`
tail -f taiwan_biobank_analysis.log
\`\`\`

## Contributing

Contributions to improve the tool are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Write unit tests for new features
- Update documentation to reflect changes

## Future Development

Planned features for future releases:

1. **Full-text Analysis**: Integrate with full-text APIs to analyze complete papers, not just abstracts.

2. **Citation Network Analysis**: Analyze citation patterns between papers to identify influential research.

3. **Machine Learning Models**: Implement more advanced classification and topic modeling techniques.

4. **Interactive Dashboard**: Create a web-based interactive dashboard for exploring results.

5. **Multi-language Support**: Add support for papers in Chinese and other languages.

6. **Automated Updates**: Schedule regular updates to keep the analysis current with new publications.

7. **API Integration**: Provide an API for programmatic access to the analysis results.

## FAQ

### General Questions

**Q: Do I need a PubMed API key?**

A: No, but having one will allow you to make more requests per second. You can get a free API key from the NCBI.

**Q: How many papers can I analyze at once?**

A: The default limit is 100 papers, but you can increase this with the `--limit` option. Be aware that processing many papers may require significant memory and time.

### Technical Questions

**Q: Can I use this tool for other biobanks or research topics?**

A: Yes, you can modify the search query using the `--search` parameter to analyze literature on any topic available in PubMed.

**Q: How accurate is the clustering?**

A: The clustering accuracy depends on the similarity of the papers and the number of clusters specified. You may need to experiment with different cluster numbers to find the optimal grouping for your dataset.

**Q: Can I customize the stopwords list?**

A: Yes, you can modify the `STOPWORDS` list in `config.py` to add or remove stopwords specific to your analysis needs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NCBI for providing the PubMed API
- The Taiwan Human Biobank for their valuable contribution to biomedical research
- The scientific community for their publications on Taiwan Human Biobank research
- Open-source libraries: NLTK, scikit-learn, matplotlib, pandas, and others

