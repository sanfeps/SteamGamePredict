"""
Classification models for Steam Games success prediction.
Implements multiple classifiers: Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, and SVM.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import SteamDataPreprocessor
from src.utils import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    save_model,
    save_metrics
)


class SteamClassifier:
    """
    Steam games success classification pipeline.
    """

    def __init__(self):
        """
        Initialize classifier with multiple models.
        """
        self.models = {}
        self.results = {}
        self.trained = False

    def initialize_models(self):
        """
        Initialize all classification models.
        """
        print("\n" + "="*60)
        print("INITIALIZING CLASSIFICATION MODELS")
        print("="*60 + "\n")

        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }

        print("Initialized models:")
        for name in self.models.keys():
            print(f"  ✓ {name}")

    def train_models(self, X_train, y_train):
        """
        Train all classification models.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        """
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODELS")
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
            Test labels
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
        print("EVALUATING CLASSIFICATION MODELS")
        print("="*60 + "\n")

        all_metrics = []

        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"{'='*60}")

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Evaluate
            metrics = evaluate_classification(y_test, y_pred, y_proba, model_name=name)

            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics
            }

            all_metrics.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False) if 'roc_auc' in comparison_df.columns else comparison_df

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60 + "\n")
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_all_confusion_matrices(self, y_test, save_dir=None):
        """
        Plot confusion matrices for all models.

        Parameters:
        -----------
        y_test : pd.Series
            Test labels
        save_dir : str, optional
            Directory to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 6:
                break

            y_pred = result['predictions']
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Not Success', 'Success'],
                       yticklabels=['Not Success', 'Success'])
            axes[idx].set_title(f'{name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        # Hide extra subplot
        if len(self.results) < 6:
            axes[5].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrices saved to {save_path}")

        plt.show()

    def plot_all_roc_curves(self, y_test, save_dir=None):
        """
        Plot ROC curves for all models on same plot.

        Parameters:
        -----------
        y_test : pd.Series
            Test labels
        save_dir : str, optional
            Directory to save plot
        """
        plt.figure(figsize=(10, 8))

        for name, result in self.results.items():
            if result['probabilities'] is not None:
                y_proba = result['probabilities']
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Classification Models')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_roc_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")

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
        tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting']

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

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

    def save_all_models(self, save_dir='models/classification'):
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
    Main function to run classification pipeline.
    """
    print("\n" + "="*60)
    print("STEAM GAMES CLASSIFICATION PIPELINE")
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
        target_col='success',
        test_size=0.2,
        random_state=42,
        scale=True
    )

    # Initialize classifier
    classifier = SteamClassifier()
    classifier.initialize_models()

    # Train models
    classifier.train_models(X_train, y_train)

    # Evaluate models
    comparison_df = classifier.evaluate_models(X_test, y_test, feature_names=preprocessor.feature_names)

    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")

    results_dir = os.path.join('results', 'figures', 'classification')

    classifier.plot_all_confusion_matrices(y_test, save_dir=results_dir)
    classifier.plot_all_roc_curves(y_test, save_dir=results_dir)
    classifier.plot_feature_importance_comparison(
        preprocessor.feature_names,
        save_dir=results_dir,
        top_n=15
    )

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save models
    classifier.save_all_models()

    # Save metrics
    metrics_path = os.path.join('results', 'metrics', 'classification_metrics.csv')
    save_metrics(comparison_df.to_dict('records'), metrics_path)

    print("\n" + "="*60)
    print("CLASSIFICATION PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
