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
    A class to perform pair trading analysis, including data fetching, PCA, clustering, 
    cointegration testing, and visualization.

    Can make a choice of either using DBSCAN or OPTICS for clustering.
    """

    def __init__(self, file_name: str):
        """
        Initialize the Pairs class with the given file name.
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
        Fetch and process the data from the CSV file.
        """
        data_fetch = DataFetch(self.file_name)
        self.pricing = data_fetch.get_pricing()
        self.returns = data_fetch.get_returns()
        self.tickers_list = data_fetch.get_tickers_list()

    def perform_pca(self, n_components=12):
        """
        Perform PCA on the returns data and standardize the components.
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
        Applying the OPTICS clustering algorithm to the PCA-transformed data.
        """
        # clf = OPTICS(min_samples=3, xi=0.05, min_cluster_size=0.1)
        clf = DBSCAN(eps=1.9, min_samples=3)
        clf.fit(self.X_df)
        labels = clf.labels_
        self.clustered_series = pd.Series(labels, index=self.returns.columns)
        self.clustered_series = self.clustered_series[self.clustered_series != -1]

    def find_cointegrated_pairs(self, data, significance=0.05):
        """
        Finding cointegrated pairs within the data.
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
        Evaluate the clusters and finding cointegrated pairs within each cluster.
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
        Visualize the pairs using t-SNE and save the plot as an image.
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
        Run the entire pair trading analysis pipeline.
        """
        self.get_data()
        self.perform_pca()
        self.cluster_data()
        self.evaluate_clusters()
        self.visualize_pairs()
