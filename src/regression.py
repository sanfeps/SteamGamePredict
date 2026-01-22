"""
Regression models for Steam Games owners prediction.
Implements multiple regressors: Linear Regression, Random Forest,
Gradient Boosting, and SVR.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import SteamDataPreprocessor
from src.utils import (
    evaluate_regression,
    plot_regression_predictions,
    plot_feature_importance,
    save_model,
    save_metrics
)


class SteamRegressor:
    """
    Steam games owners prediction pipeline using regression.
    """

    def __init__(self):
        """
        Initialize regressor with multiple models.
        """
        self.models = {}
        self.results = {}
        self.trained = False

    def initialize_models(self):
        """
        Initialize all regression models.
        """
        print("\n" + "="*60)
        print("INITIALIZING REGRESSION MODELS")
        print("="*60 + "\n")

        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVR': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
        }

        print("Initialized models:")
        for name in self.models.keys():
            print(f"  ✓ {name}")

    def train_models(self, X_train, y_train):
        """
        Train all regression models.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training targets
        """
        print("\n" + "="*60)
        print("TRAINING REGRESSION MODELS")
        print("="*60 + "\n")

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            print(f"✓ {name} trained successfully\n")

        self.trained = True

    def evaluate_models(self, X_test, y_test, feature_names=None):
        """
        Evaluate all trained models.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets
        feature_names : list, optional
            Names of features for importance plots

        Returns:
        --------
        pd.DataFrame
            Comparison of all models
        """
        if not self.trained:
            raise ValueError("Models not trained yet. Call train_models() first.")

        print("\n" + "="*60)
        print("EVALUATING REGRESSION MODELS")
        print("="*60 + "\n")

        all_metrics = []

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"{'='*60}")

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = evaluate_regression(y_test, y_pred, model_name=name)

            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics
            }

            all_metrics.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('r2', ascending=False)

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60 + "\n")
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_all_predictions(self, y_test, save_dir=None):
        """
        Plot predictions vs actual for all models.

        Parameters:
        -----------
        y_test : pd.Series
            Test targets
        save_dir : str, optional
            Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 4:
                break

            y_pred = result['predictions']

            axes[idx].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=30)
            axes[idx].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()],
                          'r--', lw=2)

            # Calculate R²
            r2 = r2_score(y_test, y_pred)
            axes[idx].set_xlabel('Actual Owners')
            axes[idx].set_ylabel('Predicted Owners')
            axes[idx].set_title(f'{name} (R² = {r2:.4f})')
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPrediction plots saved to {save_path}")

        plt.show()

    def plot_residuals(self, y_test, save_dir=None):
        """
        Plot residuals for all models.

        Parameters:
        -----------
        y_test : pd.Series
            Test targets
        save_dir : str, optional
            Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 4:
                break

            y_pred = result['predictions']
            residuals = y_test - y_pred

            axes[idx].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=30)
            axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[idx].set_xlabel('Predicted Owners')
            axes[idx].set_ylabel('Residuals')
            axes[idx].set_title(f'{name} - Residual Plot')
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_residuals.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plots saved to {save_path}")

        plt.show()

    def plot_metrics_comparison(self, comparison_df, save_dir=None):
        """
        Plot bar chart comparing metrics across models.

        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison metrics
        save_dir : str, optional
            Directory to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['mae', 'rmse', 'r2']
        titles = ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'R² Score']
        colors = ['#ff6b6b', '#feca57', '#48dbfb']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            if metric in comparison_df.columns:
                ascending = False if metric == 'r2' else True
                data = comparison_df.sort_values(metric, ascending=ascending)

                axes[idx].barh(data['model'], data[metric], color=color)
                axes[idx].set_xlabel(title)
                axes[idx].set_title(title)
                axes[idx].invert_yaxis()

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'metrics_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")

        plt.show()

    def plot_feature_importance_comparison(self, feature_names, save_dir=None, top_n=15):
        """
        Plot feature importance for tree-based models.

        Parameters:
        -----------
        feature_names : list
            Names of features
        save_dir : str, optional
            Directory to save plots
        top_n : int
            Number of top features to display
        """
        tree_models = ['Random Forest', 'Gradient Boosting']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, model_name in enumerate(tree_models):
            if model_name in self.results:
                model = self.results[model_name]['model']
                importances = model.feature_importances_

                # Get top features
                indices = np.argsort(importances)[::-1][:top_n]

                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name}')
                axes[idx].invert_yaxis()

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'feature_importance_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance comparison saved to {save_path}")

        plt.show()

    def save_all_models(self, save_dir='models/regression'):
        """
        Save all trained models.

        Parameters:
        -----------
        save_dir : str
            Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for name, result in self.results.items():
            model = result['model']
            filename = f"{name.replace(' ', '_').lower()}_{timestamp}.joblib"
            filepath = os.path.join(save_dir, filename)
            joblib.dump(model, filepath)
            print(f"✓ Saved {name} to {filepath}")


def main():
    """
    Main function to run regression pipeline.
    """
    print("\n" + "="*60)
    print("STEAM GAMES REGRESSION PIPELINE")
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

    # Initialize preprocessor for train-test split
    preprocessor = SteamDataPreprocessor()

    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(
        df,
        target_col='owners_mid',
        test_size=0.2,
        random_state=42,
        scale=True
    )

    # Initialize regressor
    regressor = SteamRegressor()
    regressor.initialize_models()

    # Train models
    regressor.train_models(X_train, y_train)

    # Evaluate models
    comparison_df = regressor.evaluate_models(X_test, y_test, feature_names=preprocessor.feature_names)

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")

    results_dir = os.path.join('results', 'figures', 'regression')

    regressor.plot_all_predictions(y_test, save_dir=results_dir)
    regressor.plot_residuals(y_test, save_dir=results_dir)
    regressor.plot_metrics_comparison(comparison_df, save_dir=results_dir)
    regressor.plot_feature_importance_comparison(
        preprocessor.feature_names,
        save_dir=results_dir,
        top_n=15
    )

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save models
    regressor.save_all_models()

    # Save metrics
    metrics_path = os.path.join('results', 'metrics', 'regression_metrics.csv')
    save_metrics(comparison_df.to_dict('records'), metrics_path)

    print("\n" + "="*60)
    print("REGRESSION PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
