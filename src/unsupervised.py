"""
Unsupervised learning: PCA and Clustering for Steam Games.
Implements dimensionality reduction and market segmentation analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import SteamDataPreprocessor
from src.utils import save_model, save_metrics


class SteamUnsupervisedAnalysis:
    """
    Unsupervised learning pipeline for Steam games analysis.
    Includes PCA and K-Means clustering.
    """

    def __init__(self):
        """
        Initialize unsupervised analysis.
        """
        self.pca_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.feature_names = None

    def prepare_data(self, df, feature_cols=None):
        """
        Prepare data for unsupervised learning.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed dataset
        feature_cols : list, optional
            Feature columns to use

        Returns:
        --------
        np.ndarray
            Scaled feature matrix
        """
        print("\n" + "="*60)
        print("PREPARING DATA FOR UNSUPERVISED LEARNING")
        print("="*60 + "\n")

        if feature_cols is None:
            # Use all numeric columns except target variables
            exclude_cols = ['owners', 'owners_mid', 'success', 'name',
                           'appid', 'release_date', 'developer', 'publisher',
                           'genres', 'categories', 'platforms', 'price_category']
            feature_cols = [col for col in df.columns
                          if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_cols
        print(f"Number of features: {len(feature_cols)}")
        print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")

        # Extract features
        X = df[feature_cols].copy()

        # Handle NaN
        X = X.fillna(0)

        # Scale
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"\n✓ Data scaled. Shape: {self.X_scaled.shape}")

        return self.X_scaled

    def perform_pca(self, n_components=None, explained_variance_threshold=0.95):
        """
        Perform PCA analysis.

        Parameters:
        -----------
        n_components : int, optional
            Number of components (if None, will choose based on variance threshold)
        explained_variance_threshold : float
            Minimum cumulative explained variance

        Returns:
        --------
        np.ndarray
            Transformed data
        """
        print("\n" + "="*60)
        print("PCA ANALYSIS")
        print("="*60 + "\n")

        # First, fit PCA with all components to analyze variance
        pca_full = PCA()
        pca_full.fit(self.X_scaled)

        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

        if n_components is None:
            # Find number of components needed for threshold
            n_components = np.argmax(cumsum_variance >= explained_variance_threshold) + 1

        print(f"Number of components: {n_components}")
        print(f"Explained variance threshold: {explained_variance_threshold*100:.1f}%")

        # Fit PCA with chosen components
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(self.X_scaled)

        # Report variance
        explained_var = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        print(f"\nExplained variance by component:")
        for i, (var, cumvar) in enumerate(zip(explained_var, cumsum_var)):
            print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cumvar*100:.2f}%)")

        print(f"\n✓ PCA complete. Reduced from {self.X_scaled.shape[1]} to {n_components} dimensions")

        return X_pca

    def plot_pca_variance(self, save_dir=None):
        """
        Plot explained variance by PCA components.

        Parameters:
        -----------
        save_dir : str, optional
            Directory to save plot
        """
        if self.pca_model is None:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")

        explained_var = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Individual variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(alpha=0.3)

        # Cumulative variance
        ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o', linestyle='--', color='darkorange')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'pca_variance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA variance plot saved to {save_path}")

        plt.show()

    def plot_pca_components_interpretation(self, top_features=10, save_dir=None):
        """
        Plot top features contributing to each principal component.

        Parameters:
        -----------
        top_features : int
            Number of top features to display
        save_dir : str, optional
            Directory to save plot
        """
        if self.pca_model is None:
            raise ValueError("PCA not performed yet. Call perform_pca() first.")

        components = self.pca_model.components_
        n_components = min(3, len(components))

        fig, axes = plt.subplots(1, n_components, figsize=(18, 6))
        if n_components == 1:
            axes = [axes]

        for idx in range(n_components):
            # Get component loadings
            loadings = components[idx]
            indices = np.argsort(np.abs(loadings))[::-1][:top_features]

            axes[idx].barh(range(top_features), loadings[indices])
            axes[idx].set_yticks(range(top_features))
            axes[idx].set_yticklabels([self.feature_names[i] for i in indices])
            axes[idx].set_xlabel('Loading')
            axes[idx].set_title(f'PC{idx+1} Top Features')
            axes[idx].invert_yaxis()
            axes[idx].axvline(x=0, color='k', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'pca_component_interpretation.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA components interpretation saved to {save_path}")

        plt.show()

    def plot_pca_2d(self, X_pca, df=None, color_by='success', save_dir=None):
        """
        Plot 2D PCA visualization.

        Parameters:
        -----------
        X_pca : np.ndarray
            PCA-transformed data
        df : pd.DataFrame, optional
            Original dataframe for coloring
        color_by : str
            Column name to color points by
        save_dir : str, optional
            Directory to save plot
        """
        plt.figure(figsize=(12, 8))

        if df is not None and color_by in df.columns:
            colors = df[color_by].values
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis',
                                 alpha=0.6, edgecolors='k', s=30)
            plt.colorbar(scatter, label=color_by)
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolors='k', s=30)

        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]*100:.2f}%)')
        plt.title('PCA - First Two Principal Components')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'pca_2d.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA 2D plot saved to {save_path}")

        plt.show()

    def plot_pca_3d(self, X_pca, df=None, color_by='success', save_dir=None):
        """
        Plot 3D PCA visualization.

        Parameters:
        -----------
        X_pca : np.ndarray
            PCA-transformed data
        df : pd.DataFrame, optional
            Original dataframe for coloring
        color_by : str
            Column name to color points by
        save_dir : str, optional
            Directory to save plot
        """
        if X_pca.shape[1] < 3:
            print("Warning: Less than 3 components available for 3D plot")
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if df is not None and color_by in df.columns:
            colors = df[color_by].values
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                               c=colors, cmap='viridis', alpha=0.6, edgecolors='k', s=30)
            plt.colorbar(scatter, label=color_by, ax=ax, shrink=0.5)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                      alpha=0.6, edgecolors='k', s=30)

        ax.set_xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]*100:.2f}%)')
        ax.set_ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]*100:.2f}%)')
        ax.set_zlabel(f'PC3 ({self.pca_model.explained_variance_ratio_[2]*100:.2f}%)')
        ax.set_title('PCA - First Three Principal Components')

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'pca_3d.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA 3D plot saved to {save_path}")

        plt.show()

    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette score.

        Parameters:
        -----------
        X : np.ndarray
            Data to cluster
        max_clusters : int
            Maximum number of clusters to test

        Returns:
        --------
        dict
            Dictionary with inertia and silhouette scores
        """
        print("\n" + "="*60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60 + "\n")

        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)

        for k in K_range:
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
            print(f"  Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil_score:.4f}")

        return {
            'k_values': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }

    def plot_cluster_optimization(self, optimization_results, save_dir=None):
        """
        Plot elbow curve and silhouette scores.

        Parameters:
        -----------
        optimization_results : dict
            Results from find_optimal_clusters
        save_dir : str, optional
            Directory to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        k_values = optimization_results['k_values']
        inertias = optimization_results['inertias']
        silhouette_scores = optimization_results['silhouette_scores']

        # Elbow plot
        ax1.plot(k_values, inertias, marker='o', linestyle='--', color='steelblue')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
        ax1.set_title('Elbow Method')
        ax1.grid(alpha=0.3)

        # Silhouette plot
        ax2.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='darkorange')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs Number of Clusters')
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'cluster_optimization.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster optimization plot saved to {save_path}")

        plt.show()

    def perform_clustering(self, X, n_clusters=4):
        """
        Perform K-Means clustering.

        Parameters:
        -----------
        X : np.ndarray
            Data to cluster
        n_clusters : int
            Number of clusters

        Returns:
        --------
        np.ndarray
            Cluster labels
        """
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING")
        print("="*60 + "\n")

        print(f"Number of clusters: {n_clusters}")

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans_model.fit_predict(X)

        # Calculate silhouette score
        sil_score = silhouette_score(X, labels)
        print(f"\nSilhouette Score: {sil_score:.4f}")

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} games ({count/len(labels)*100:.2f}%)")

        print(f"\n✓ Clustering complete")

        return labels

    def analyze_clusters(self, df, labels, feature_cols=None):
        """
        Analyze cluster characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe
        labels : np.ndarray
            Cluster labels
        feature_cols : list, optional
            Features to analyze

        Returns:
        --------
        pd.DataFrame
            Cluster statistics
        """
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60 + "\n")

        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        if feature_cols is None:
            feature_cols = ['price', 'positive_ratio', 'total_ratings', 'owners_mid']

        # Calculate mean statistics per cluster
        cluster_stats = df_analysis.groupby('cluster')[feature_cols].mean()

        print("Cluster Statistics (Mean Values):")
        print(cluster_stats)

        return cluster_stats

    def plot_clusters_2d(self, X_pca, labels, save_dir=None):
        """
        Plot clusters in 2D PCA space.

        Parameters:
        -----------
        X_pca : np.ndarray
            PCA-transformed data
        labels : np.ndarray
            Cluster labels
        save_dir : str, optional
            Directory to save plot
        """
        plt.figure(figsize=(12, 8))

        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10',
                             alpha=0.6, edgecolors='k', s=30)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1]*100:.2f}%)')
        plt.title('K-Means Clusters in PCA Space')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'clusters_2d.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster 2D plot saved to {save_path}")

        plt.show()

    def save_models(self, save_dir='models/unsupervised'):
        """
        Save PCA and K-Means models.

        Parameters:
        -----------
        save_dir : str
            Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.pca_model:
            pca_path = os.path.join(save_dir, f'pca_model_{timestamp}.joblib')
            joblib.dump(self.pca_model, pca_path)
            print(f"✓ Saved PCA model to {pca_path}")

        if self.kmeans_model:
            kmeans_path = os.path.join(save_dir, f'kmeans_model_{timestamp}.joblib')
            joblib.dump(self.kmeans_model, kmeans_path)
            print(f"✓ Saved K-Means model to {kmeans_path}")


def main():
    """
    Main function to run unsupervised learning pipeline.
    """
    print("\n" + "="*60)
    print("STEAM GAMES UNSUPERVISED LEARNING PIPELINE")
    print("="*60 + "\n")

    # Paths
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'steam_games_processed.csv')

    # Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"ERROR: Processed data not found at {PROCESSED_DATA_PATH}")
        print("Please run preprocessing.py first")
        return

    # Load processed data
    print(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"✓ Data loaded. Shape: {df.shape}\n")

    # Initialize analysis
    analysis = SteamUnsupervisedAnalysis()

    # Prepare data
    X_scaled = analysis.prepare_data(df)

    # ===== PCA ANALYSIS =====
    X_pca = analysis.perform_pca(n_components=None, explained_variance_threshold=0.95)

    results_dir = os.path.join('results', 'figures', 'unsupervised')

    print("\n" + "="*60)
    print("GENERATING PCA VISUALIZATIONS")
    print("="*60 + "\n")

    analysis.plot_pca_variance(save_dir=results_dir)
    analysis.plot_pca_components_interpretation(top_features=10, save_dir=results_dir)
    analysis.plot_pca_2d(X_pca, df=df, color_by='success', save_dir=results_dir)
    analysis.plot_pca_3d(X_pca, df=df, color_by='success', save_dir=results_dir)

    # ===== CLUSTERING ANALYSIS =====
    # Find optimal number of clusters
    optimization_results = analysis.find_optimal_clusters(X_pca, max_clusters=10)
    analysis.plot_cluster_optimization(optimization_results, save_dir=results_dir)

    # Perform clustering with chosen k
    n_clusters = 4  # Adjust based on optimization results
    labels = analysis.perform_clustering(X_pca, n_clusters=n_clusters)

    # Analyze clusters
    cluster_stats = analysis.analyze_clusters(df, labels)

    # Visualize clusters
    print("\n" + "="*60)
    print("GENERATING CLUSTERING VISUALIZATIONS")
    print("="*60 + "\n")

    analysis.plot_clusters_2d(X_pca, labels, save_dir=results_dir)

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save models
    analysis.save_models()

    # Save cluster statistics
    cluster_stats_path = os.path.join('results', 'metrics', 'cluster_statistics.csv')
    os.makedirs(os.path.dirname(cluster_stats_path), exist_ok=True)
    cluster_stats.to_csv(cluster_stats_path)
    print(f"✓ Cluster statistics saved to {cluster_stats_path}")

    print("\n" + "="*60)
    print("UNSUPERVISED LEARNING PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
