"""
Data preprocessing module for Steam Games ML project.
Handles data cleaning, feature engineering, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import parse_owners_range, create_success_label, display_dataset_info


class SteamDataPreprocessor:
    """
    Preprocessor for Steam games dataset.
    Handles cleaning, feature engineering, and encoding.
    """

    def __init__(self, success_threshold=100000):
        """
        Initialize preprocessor.

        Parameters:
        -----------
        success_threshold : int
            Threshold for defining success (default: 100,000 owners)
        """
        self.success_threshold = success_threshold
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def load_data(self, file_path):
        """
        Load raw data from CSV.

        Parameters:
        -----------
        file_path : str
            Path to raw data file

        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        print(f"\nLoading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded. Shape: {df.shape}")
        return df

    def clean_data(self, df):
        """
        Clean the dataset: handle missing values, duplicates, etc.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset

        Returns:
        --------
        pd.DataFrame
            Cleaned dataset
        """
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)

        df_clean = df.copy()
        initial_rows = len(df_clean)

        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        print(f"\nDuplicates removed: {initial_rows - len(df_clean)}")

        # Parse owners to numeric
        if 'owners' in df_clean.columns:
            print("\nParsing owners range to numeric...")
            df_clean['owners_mid'] = df_clean['owners'].apply(parse_owners_range)

        # Handle missing values in key columns
        print("\nHandling missing values...")

        # Numeric columns: fill with median
        numeric_cols = ['price', 'positive_ratings', 'negative_ratings', 'owners_mid']
        for col in numeric_cols:
            if col in df_clean.columns:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"  {col}: Filled {missing_count} missing values with median ({median_val:.2f})")

        # Categorical columns: fill with 'Unknown'
        categorical_cols = ['genres', 'categories', 'platforms', 'developer', 'publisher']
        for col in categorical_cols:
            if col in df_clean.columns:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    df_clean[col].fillna('Unknown', inplace=True)
                    print(f"  {col}: Filled {missing_count} missing values with 'Unknown'")

        # Remove rows with missing owners_mid (our target variable)
        if 'owners_mid' in df_clean.columns:
            rows_with_missing_target = df_clean['owners_mid'].isnull().sum()
            df_clean = df_clean.dropna(subset=['owners_mid'])
            print(f"\nRows removed due to missing target (owners_mid): {rows_with_missing_target}")

        print(f"\nFinal dataset shape after cleaning: {df_clean.shape}")

        return df_clean

    def create_features(self, df):
        """
        Create engineered features.

        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned dataset

        Returns:
        --------
        pd.DataFrame
            Dataset with engineered features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        df_feat = df.copy()

        # 1. Positive ratio
        if 'positive_ratings' in df_feat.columns and 'negative_ratings' in df_feat.columns:
            total_ratings = df_feat['positive_ratings'] + df_feat['negative_ratings']
            df_feat['positive_ratio'] = np.where(
                total_ratings > 0,
                df_feat['positive_ratings'] / total_ratings,
                0.5  # Neutral if no ratings
            )
            print("\n✓ Created 'positive_ratio' feature")

        # 2. Total ratings
        if 'positive_ratings' in df_feat.columns and 'negative_ratings' in df_feat.columns:
            df_feat['total_ratings'] = df_feat['positive_ratings'] + df_feat['negative_ratings']
            print("✓ Created 'total_ratings' feature")

        # 3. Price category
        if 'price' in df_feat.columns:
            df_feat['is_free'] = (df_feat['price'] == 0).astype(int)
            df_feat['price_category'] = pd.cut(
                df_feat['price'],
                bins=[-0.01, 0, 10, 30, 60, np.inf],
                labels=['Free', 'Budget', 'Standard', 'Premium', 'AAA']
            )
            print("✓ Created 'is_free' and 'price_category' features")

        # 4. Number of genres (indicator of game complexity)
        if 'genres' in df_feat.columns:
            df_feat['genre_count'] = df_feat['genres'].str.split(',').apply(
                lambda x: len(x) if isinstance(x, list) else 1
            )
            print("✓ Created 'genre_count' feature")

        # 5. Number of platforms
        if 'platforms' in df_feat.columns:
            df_feat['platform_count'] = df_feat['platforms'].str.split(',').apply(
                lambda x: len(x) if isinstance(x, list) else 1
            )
            print("✓ Created 'platform_count' feature")

        # 6. Release date features
        if 'release_date' in df_feat.columns:
            df_feat['release_date'] = pd.to_datetime(df_feat['release_date'], errors='coerce')
            df_feat['release_year'] = df_feat['release_date'].dt.year
            df_feat['release_month'] = df_feat['release_date'].dt.month
            df_feat['release_quarter'] = df_feat['release_date'].dt.quarter

            # Fill missing years with median
            median_year = df_feat['release_year'].median()
            df_feat['release_year'].fillna(median_year, inplace=True)
            df_feat['release_month'].fillna(6, inplace=True)  # Mid-year if missing
            df_feat['release_quarter'].fillna(2, inplace=True)

            print("✓ Created 'release_year', 'release_month', 'release_quarter' features")

        # 7. Success label (for classification)
        if 'owners_mid' in df_feat.columns:
            df_feat['success'] = create_success_label(
                df_feat['owners_mid'],
                threshold=self.success_threshold
            )
            success_rate = df_feat['success'].mean() * 100
            print(f"✓ Created 'success' label (threshold: {self.success_threshold:,} owners)")
            print(f"  Success rate: {success_rate:.2f}%")

        return df_feat

    def encode_categorical_features(self, df, categorical_cols, method='onehot', max_categories=50):
        """
        Encode categorical variables.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with features
        categorical_cols : list
            List of categorical column names
        method : str
            Encoding method: 'onehot' or 'label'
        max_categories : int
            Maximum number of categories to keep for one-hot encoding

        Returns:
        --------
        pd.DataFrame
            Dataset with encoded features
        """
        print("\n" + "="*60)
        print("CATEGORICAL ENCODING")
        print("="*60)

        df_encoded = df.copy()

        for col in categorical_cols:
            if col not in df_encoded.columns:
                continue

            if method == 'onehot':
                # Check if column contains multiple values separated by ';'
                sample_value = df_encoded[col].dropna().iloc[0] if len(df_encoded[col].dropna()) > 0 else ''
                is_multi_value = isinstance(sample_value, str) and ';' in sample_value

                if is_multi_value:
                    # Handle multi-value columns (genres, categories, platforms, tags)
                    print(f"\nProcessing multi-value column '{col}'...")

                    # Collect all unique values across all rows
                    all_values = []
                    for value in df_encoded[col].dropna():
                        if isinstance(value, str):
                            values = [v.strip() for v in value.split(';')]
                            all_values.extend(values)

                    # Get top values by frequency
                    from collections import Counter
                    value_counts = Counter(all_values)
                    top_values = [item[0] for item in value_counts.most_common(max_categories)]

                    # Create binary columns for each top value
                    for value in top_values:
                        col_name = f"{col}_{value}".replace(' ', '_').replace('-', '_').replace('&', 'and')
                        # Check if the value appears in each row
                        df_encoded[col_name] = df_encoded[col].apply(
                            lambda x: 1 if isinstance(x, str) and value in x.split(';') else 0
                        )

                    print(f"✓ One-hot encoded '{col}': {len(top_values)} unique values from multi-value field")
                    print(f"  Top values: {top_values[:10]}...")

                else:
                    # Handle single-value columns (developer, publisher)
                    # Get top categories
                    top_categories = df_encoded[col].value_counts().head(max_categories).index.tolist()

                    # Create binary columns for top categories
                    for category in top_categories:
                        col_name = f"{col}_{category}".replace(' ', '_').replace('-', '_').replace('&', 'and')
                        df_encoded[col_name] = (df_encoded[col] == category).astype(int)

                    print(f"\n✓ One-hot encoded '{col}': {len(top_categories)} categories")

            elif method == 'label':
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"\n✓ Label encoded '{col}': {len(le.classes_)} unique values")

        return df_encoded

    def prepare_data_for_modeling(self, df, target_col, feature_cols=None,
                                  test_size=0.2, random_state=42, scale=True):
        """
        Prepare data for modeling: split into train/test and scale.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed dataset
        target_col : str
            Name of target column
        feature_cols : list, optional
            List of feature columns (if None, use all except target)
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
        scale : bool
            Whether to scale features

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("TRAIN-TEST SPLIT")
        print("="*60)

        # Select features
        if feature_cols is None:
            # Use all numeric columns except target and identifiers
            exclude_cols = [target_col, 'owners', 'owners_mid', 'success', 'name',
                           'appid', 'release_date', 'developer', 'publisher',
                           'genres', 'categories', 'platforms', 'price_category']
            feature_cols = [col for col in df.columns
                          if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

        self.feature_names = feature_cols
        print(f"\nNumber of features: {len(feature_cols)}")
        print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")

        # Extract X and y
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle any remaining NaN values
        X = X.fillna(0)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if target_col == 'success' else None
        )

        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        if target_col == 'success':
            print(f"\nClass distribution in train:")
            print(f"  Not Success (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
            print(f"  Success (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")

        # Scale features
        if scale:
            print("\nScaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
            print("✓ Features scaled using StandardScaler")

        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, raw_data_path, save_processed=True):
        """
        Complete preprocessing pipeline.

        Parameters:
        -----------
        raw_data_path : str
            Path to raw data CSV
        save_processed : bool
            Whether to save processed data

        Returns:
        --------
        pd.DataFrame
            Fully preprocessed dataset
        """
        # Load
        df = self.load_data(raw_data_path)

        # Display initial info
        display_dataset_info(df)

        # Clean
        df_clean = self.clean_data(df)

        # Feature engineering
        df_feat = self.create_features(df_clean)

        # Encode categorical features
        # Include genres, categories, platforms, and top developers/publishers
        categorical_cols = ['genres', 'categories', 'platforms', 'developer', 'publisher']
        df_processed = self.encode_categorical_features(
            df_feat,
            categorical_cols=categorical_cols,
            method='onehot',
            max_categories=20  # Top 20 for each category
        )

        # Save processed data
        if save_processed:
            processed_path = raw_data_path.replace('raw', 'processed').replace('.csv', '_processed.csv')
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            df_processed.to_csv(processed_path, index=False)
            print(f"\n{'='*60}")
            print(f"Processed data saved to: {processed_path}")
            print(f"{'='*60}\n")

        return df_processed


def main():
    """
    Main function to run preprocessing.
    """
    # File paths
    RAW_DATA_PATH = os.path.join('data', 'raw', 'steam_games.csv')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'steam_games_processed.csv')

    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data not found at {RAW_DATA_PATH}")
        print("Please download the Steam games dataset and place it in data/raw/")
        print("Example datasets:")
        print("  - https://www.kaggle.com/datasets/nikdavis/steam-store-games")
        print("  - https://www.kaggle.com/datasets/fronkongames/steam-games-dataset")
        return

    # Initialize preprocessor
    preprocessor = SteamDataPreprocessor(success_threshold=100000)

    # Run preprocessing pipeline
    df_processed = preprocessor.preprocess_pipeline(RAW_DATA_PATH, save_processed=True)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
