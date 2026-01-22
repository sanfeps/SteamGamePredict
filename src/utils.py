"""
Utility functions for the Steam Games ML project.
Contains helper functions for data loading, visualization, and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import os


def load_data(file_path):
    """
    Load dataset from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def parse_owners_range(owners_str):
    """
    Convert owners range string to numeric value (mean of range).
    Example: "20,000 - 50,000" -> 35000

    Parameters:
    -----------
    owners_str : str
        String representing range of owners

    Returns:
    --------
    float
        Mean value of the range
    """
    if pd.isna(owners_str):
        return np.nan

    try:
        # Remove commas and split by '-'
        owners_str = str(owners_str).replace(',', '')
        if '-' in owners_str:
            low, high = owners_str.split('-')
            return (float(low.strip()) + float(high.strip())) / 2
        else:
            return float(owners_str.strip())
    except:
        return np.nan


def create_success_label(owners_mid, threshold=100000):
    """
    Create binary success label based on owners threshold.

    Parameters:
    -----------
    owners_mid : float or pd.Series
        Number of owners
    threshold : int
        Threshold for success definition (default: 100,000)

    Returns:
    --------
    int or pd.Series
        Binary label (1 = success, 0 = not success)
    """
    return (owners_mid >= threshold).astype(int)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Success', 'Success'],
                yticklabels=['Not Success', 'Success'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, title="ROC Curve", save_path=None):
    """
    Plot ROC curve.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities for positive class
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()


def evaluate_classification(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive evaluation of classification model.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for ROC-AUC
    model_name : str
        Name of the model for reporting

    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Classification Evaluation: {model_name}")
    print(f"{'='*60}\n")

    # Classification report
    print(classification_report(y_true, y_pred, target_names=['Not Success', 'Success']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}\n")

    metrics = {
        'model': model_name,
        'accuracy': (y_pred == y_true).mean()
    }

    # ROC-AUC if probabilities provided
    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        metrics['roc_auc'] = roc_auc
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")

    return metrics


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Comprehensive evaluation of regression model.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for reporting

    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Regression Evaluation: {model_name}")
    print(f"{'='*60}\n")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE):  {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}\n")

    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def plot_regression_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Plot predicted vs actual values for regression.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")

    plt.show()


def plot_feature_importance(feature_names, importances, title="Feature Importance",
                           top_n=20, save_path=None):
    """
    Plot feature importances.

    Parameters:
    -----------
    feature_names : array-like
        Names of features
    importances : array-like
        Importance values
    title : str
        Title for the plot
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def save_model(model, file_path):
    """
    Save trained model to disk.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    file_path : str
        Path to save the model
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    """
    Load trained model from disk.

    Parameters:
    -----------
    file_path : str
        Path to the saved model

    Returns:
    --------
    sklearn model
        Loaded model
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model not found at {file_path}")

    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


def save_metrics(metrics_dict, file_path):
    """
    Save metrics dictionary to CSV.

    Parameters:
    -----------
    metrics_dict : dict or list of dict
        Metrics to save
    file_path : str
        Path to save the metrics
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if isinstance(metrics_dict, dict):
        metrics_dict = [metrics_dict]

    df = pd.DataFrame(metrics_dict)
    df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")


def display_dataset_info(df):
    """
    Display comprehensive information about the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60 + "\n")

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Column Information:")
    print("-" * 60)
    print(df.info())

    print("\n" + "-" * 60)
    print("Missing Values:")
    print("-" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

    print("\n" + "-" * 60)
    print("Numerical Columns Statistics:")
    print("-" * 60)
    print(df.describe())

    print("\n" + "="*60 + "\n")
