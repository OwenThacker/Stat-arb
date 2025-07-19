import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from statsmodels.tsa.stattools import coint
from scipy import stats

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data.data_fetch import DataFetch

class Pairs:
    """
    A class to perform comprehensive pair trading analysis using machine learning techniques.
    
    This class implements a complete pipeline for identifying profitable trading pairs through:
    1. Data fetching and preprocessing
    2. Principal Component Analysis (PCA) for dimensionality reduction
    3. Clustering analysis using DBSCAN or OPTICS algorithms
    4. Cointegration testing for statistical validation
    5. Visualization of results using t-SNE
    
    The analysis helps identify pairs of stocks that move together historically and may
    offer arbitrage opportunities when they diverge from their normal relationship.
    
    Attributes:
        file_name (str): Name of the CSV file containing stock price data
        pricing (pd.DataFrame): Raw stock pricing data
        returns (pd.DataFrame): Stock returns data
        tickers_list (list): List of stock ticker symbols
        clustered_series (pd.Series): Series mapping tickers to cluster labels
        X_df (pd.DataFrame): PCA-transformed and standardized data
        X_pairs (pd.DataFrame): PCA data subset for identified pairs
        X_tsne (np.ndarray): t-SNE transformed coordinates for visualization
        pairs (list): List of tuples containing cointegrated stock pairs
        cluster_dict (dict): Dictionary containing clustering results and statistics
        
    """

    def __init__(self, file_name: str):
        """
        Initialize the Pairs class with the specified data file.
        
        Args:
            file_name (str): Path to the CSV file containing stock price data.
                           The file should have dates as index and stock symbols as columns.
        
        """
        self.file_name = file_name
        self.pricing = None
        self.returns = None
        self.tickers_list = None
        self.clustered_series = None
        self.X_df = None
        self.X_pairs = None
        self.X_tsne = None
        self.pairs = None
        self.cluster_dict = None

    def get_data(self):
        """
        Fetch and process stock price data from the specified CSV file.
        
        Uses the DataFetch class to load pricing data, calculate returns, and extract
        the list of ticker symbols. This method populates the pricing, returns, and
        tickers_list attributes.

        """
        data_fetch = DataFetch(self.file_name)
        self.pricing = data_fetch.get_pricing()
        self.returns = data_fetch.get_returns()
        self.tickers_list = data_fetch.get_tickers_list()

    def perform_pca(self, n_components=12):
        """
        Perform Principal Component Analysis on log returns to reduce dimensionality.
        
        Converts pricing data to log returns, applies PCA to extract the most significant
        components, and standardizes the resulting features. This reduces the dimensionality
        while preserving the most important variance in the data.
        
        Args:
            n_components (int, optional): Number of principal components to extract.
                                        Defaults to 12. Should be less than the number
                                        of stocks in the dataset.
        
        Returns:
            None: Updates the X_df attribute with standardized PCA components
        
        """
        log_returns = np.log(self.pricing) - np.log(self.pricing.shift(1))
        log_returns = log_returns.fillna(0)
        pca = PCA(n_components=n_components)
        pca.fit(log_returns)
        X = pca.components_.T
        X_scaled = preprocessing.StandardScaler().fit_transform(X)
        self.X_df = pd.DataFrame(X_scaled, index=self.returns.columns)

    def cluster_data(self):
        """
        Apply DBSCAN clustering algorithm to group similar stocks.
        
        Uses DBSCAN (Density-Based Spatial Clustering) to identify clusters of stocks
        that behave similarly based on their PCA-transformed features. Stocks in the
        same cluster are more likely to be cointegrated and suitable for pair trading.
        
        The method filters out noise points (labeled as -1) and only keeps stocks
        that belong to valid clusters.
        
        Algorithm parameters:
            - eps=1.9: Maximum distance between points in the same neighborhood
            - min_samples=3: Minimum number of points required to form a dense region
        
        Returns:
            None: Updates the clustered_series attribute with cluster assignments
        
        Note:
            The current implementation uses DBSCAN, but OPTICS is available as an
            alternative by uncommenting the relevant lines.
  
        """
        # clf = OPTICS(min_samples=3, xi=0.05, min_cluster_size=0.1)
        clf = DBSCAN(eps=1.9, min_samples=3)
        clf.fit(self.X_df)
        labels = clf.labels_
        self.clustered_series = pd.Series(labels, index=self.returns.columns)
        self.clustered_series = self.clustered_series[self.clustered_series != -1]

    def find_cointegrated_pairs(self, data, significance=0.05):
        """
        Identify cointegrated pairs within a given dataset using statistical tests.
        
        Performs the Augmented Dickey-Fuller test for cointegration on all possible
        pairs within the provided dataset. Cointegrated pairs have a long-term
        equilibrium relationship and are suitable for pairs trading strategies.
        
        Args:
            data (pd.DataFrame): Stock price data with tickers as columns and dates as index
            significance (float, optional): Significance level for the cointegration test.
                                          Defaults to 0.05 (5% significance level).
        
        Returns:
            tuple: A 3-tuple containing:
                - score_matrix (np.ndarray): Matrix of cointegration test statistics
                - pvalue_matrix (np.ndarray): Matrix of p-values for each pair test
                - pairs (list): List of tuples containing significantly cointegrated pairs
        
        Note:
            Only pairs with p-values below the significance threshold are included
            in the results. Lower p-values indicate stronger evidence of cointegration.
    
        """
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.columns
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]].dropna()
                S2 = data[keys[j]].dropna()
                if len(S1) == len(S2):
                    score, pvalue, _ = coint(S1, S2)
                    score_matrix[i, j] = score
                    pvalue_matrix[i, j] = pvalue
                    if pvalue < significance:
                        pairs.append((keys[i], keys[j]))

        return score_matrix, pvalue_matrix, pairs

    def evaluate_clusters(self):
        """
        Evaluate each cluster and identify cointegrated pairs within clusters.
        
        For each cluster with more than one stock, this method applies cointegration
        testing to find statistically significant pairs. This approach is more
        efficient than testing all possible pairs across the entire dataset.
        
        The method creates a comprehensive dictionary of results for each cluster
        and compiles a master list of all identified pairs across clusters.
        
        Returns:
            None: Updates cluster_dict and pairs attributes

        """
        self.cluster_dict = {}
        counts = self.clustered_series.value_counts()
        ticker_count_reduced = counts[counts > 1]

        for cluster_label in ticker_count_reduced.index:
            tickers = self.clustered_series[self.clustered_series == cluster_label].index
            score_matrix, pvalue_matrix, pairs = self.find_cointegrated_pairs(self.pricing[tickers])
            self.cluster_dict[cluster_label] = {
                'score_matrix': score_matrix,
                'pvalue_matrix': pvalue_matrix,
                'pairs': pairs
            }

        self.pairs = [pair for cluster in self.cluster_dict.values() for pair in cluster['pairs']]

    def visualize_pairs(self, filename='tsne_visualization.png'):
        """
        Create a t-SNE visualization of identified pairs and save as an image.
        
        Uses t-Distributed Stochastic Neighbor Embedding (t-SNE) to create a 2D
        visualization of the high-dimensional stock relationships. Connected lines
        represent cointegrated pairs, and colors indicate cluster membership.
        
        Args:
            filename (str, optional): Name of the output image file. 
                                    Defaults to 'tsne_visualization.png'.
                                    File will be saved in 'plots/Pair_Plot/' directory.
        
        Returns:
            None: Saves the visualization plot to disk
        
        Features of the visualization:
            - Scatter points represent individual stocks
            - Lines connect cointegrated pairs
            - Colors indicate cluster membership
            - Stock symbols are labeled on each point
        
        t-SNE Parameters:
            - learning_rate=50: Controls the learning rate for optimization
            - perplexity=3: Balances local vs global structure (low for small datasets)
            - random_state=1337: Ensures reproducible results
    
        """
        stocks = np.unique(self.pairs)
        self.X_pairs = self.X_df.loc[stocks]
        self.X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(self.X_pairs)

        plt.figure(1, facecolor='white')
        plt.clf()
        plt.axis('off')
        for pair in self.pairs:
            ticker1 = pair[0]
            loc1 = self.X_pairs.index.get_loc(pair[0])
            x1, y1 = self.X_tsne[loc1, :]

            ticker2 = pair[1]
            loc2 = self.X_pairs.index.get_loc(pair[1])
            x2, y2 = self.X_tsne[loc2, :]

            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')

        scatter = plt.scatter(self.X_tsne[:, 0], self.X_tsne[:, 1], s=220, alpha=0.9, c=self.clustered_series.loc[stocks].values, cmap=cm.Paired)

        for i, ticker in enumerate(self.X_pairs.index):
            plt.text(self.X_tsne[i, 0], self.X_tsne[i, 1], ticker, fontsize=5)

        plt.title('T-SNE Visualization of Validated Pairs')
        plt.savefig(f'plots\\Pair_Plot\\{filename}', bbox_inches='tight')
        plt.close()

    def Run_pairs(self):
        """
        Execute the complete pairs trading analysis pipeline.
        
        This is the main method that orchestrates the entire analysis workflow:
        1. Data fetching and preprocessing
        2. PCA dimensionality reduction
        3. Clustering analysis
        4. Cointegration testing
        5. Results visualization
        
        Returns:
            None: All results are stored in class attributes
    
        """
        self.get_data()
        self.perform_pca()
        self.cluster_data()
        self.evaluate_clusters()
        self.visualize_pairs()