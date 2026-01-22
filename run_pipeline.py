"""
Master script to run the complete Steam Games ML pipeline.
Executes preprocessing, classification, regression, and unsupervised learning.
"""

import os
import sys
import argparse
from datetime import datetime


def check_data_exists():
    """Check if raw data file exists."""
    data_path = os.path.join('data', 'raw', 'steam_games.csv')
    if not os.path.exists(data_path):
        print("\n" + "="*70)
        print("ERROR: Raw data not found!")
        print("="*70)
        print(f"\nPlease download the Steam games dataset and place it at:")
        print(f"  {data_path}")
        print("\nRecommended datasets:")
        print("  1. Kaggle: Steam Store Games Dataset")
        print("     https://www.kaggle.com/datasets/nikdavis/steam-store-games")
        print("\n  2. Kaggle: Steam Games Dataset")
        print("     https://www.kaggle.com/datasets/fronkongames/steam-games-dataset")
        print("\nAfter downloading, rename the file to 'steam_games.csv'")
        print("and place it in the data/raw/ directory.")
        print("="*70 + "\n")
        return False
    return True


def run_preprocessing():
    """Run data preprocessing."""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70 + "\n")

    from src.preprocessing import main as preprocess_main
    preprocess_main()


def run_classification():
    """Run classification models."""
    print("\n" + "="*70)
    print("STEP 2: CLASSIFICATION MODELS")
    print("="*70 + "\n")

    from src.classification import main as classification_main
    classification_main()


def run_regression():
    """Run regression models."""
    print("\n" + "="*70)
    print("STEP 3: REGRESSION MODELS")
    print("="*70 + "\n")

    from src.regression import main as regression_main
    regression_main()


def run_unsupervised():
    """Run unsupervised learning (PCA and Clustering)."""
    print("\n" + "="*70)
    print("STEP 4: UNSUPERVISED LEARNING (PCA + CLUSTERING)")
    print("="*70 + "\n")

    from src.unsupervised import main as unsupervised_main
    unsupervised_main()


def main():
    """
    Main pipeline execution.
    """
    parser = argparse.ArgumentParser(description='Steam Games ML Pipeline')
    parser.add_argument('--steps', nargs='+',
                       choices=['preprocess', 'classification', 'regression', 'unsupervised', 'all'],
                       default=['all'],
                       help='Steps to run (default: all)')
    parser.add_argument('--skip-preprocess', action='store_true',
                       help='Skip preprocessing if data is already processed')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("STEAM GAMES MACHINE LEARNING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Check if data exists
    if not check_data_exists():
        sys.exit(1)

    steps = args.steps
    if 'all' in steps:
        steps = ['preprocess', 'classification', 'regression', 'unsupervised']

    # Skip preprocessing if requested and data already processed
    if args.skip_preprocess and 'preprocess' in steps:
        processed_data = os.path.join('data', 'processed', 'steam_games_processed.csv')
        if os.path.exists(processed_data):
            print(f"\nSkipping preprocessing (data already exists at {processed_data})\n")
            steps.remove('preprocess')

    # Execute steps
    try:
        if 'preprocess' in steps:
            run_preprocessing()

        if 'classification' in steps:
            run_classification()

        if 'regression' in steps:
            run_regression()

        if 'unsupervised' in steps:
            run_unsupervised()

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nResults saved in:")
        print("  - models/ : Trained models")
        print("  - results/figures/ : Visualizations")
        print("  - results/metrics/ : Performance metrics")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"\n{str(e)}\n")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
