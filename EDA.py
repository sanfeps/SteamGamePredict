"""
Calculate descriptive statistics for Steam games dataset.
Run this from the MMLL directory: python calculate_dataset_stats.py
"""

import pandas as pd
import numpy as np
import os
import sys

def parse_owners_range(owners_str):
    """Convert owner range string to midpoint value."""
    if pd.isna(owners_str):
        return np.nan
    try:
        parts = str(owners_str).split('-')
        if len(parts) == 2:
            lower = int(parts[0])
            upper = int(parts[1])
            return (lower + upper) / 2
        return np.nan
    except:
        return np.nan

def main():
    # Check if file exists
    data_path = os.path.join('data', 'raw', 'steam_games.csv')
    
    if not os.path.exists(data_path):
        print(f"ERROR: File not found: {data_path}")
        print("Please run this script from the MMLL directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Load the raw data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print("="*70)
    print("STEAM GAMES DATASET - DESCRIPTIVE STATISTICS")
    print("="*70)
    
    # Parse owners to get midpoint
    df['owners_mid'] = df['owners'].apply(parse_owners_range)
    
    # Variables to analyze
    variables = {
        'price': 'Price ($)',
        'positive_ratings': 'Positive Ratings',
        'negative_ratings': 'Negative Ratings',
        'average_playtime': 'Average Playtime (min)',
        'achievements': 'Achievements',
        'owners_mid': 'Owners (mid)'
    }
    
    print("\n" + "="*70)
    print("STATISTICS TABLE")
    print("="*70)
    print(f"\n{'Variable':<30} {'Mean':>15} {'Median':>15} {'Std Dev':>15}")
    print("-"*75)
    
    for col, display_name in variables.items():
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            # Format based on magnitude
            if mean_val >= 1000:
                print(f"{display_name:<30} {mean_val:>15,.0f} {median_val:>15,.0f} {std_val:>15,.0f}")
            elif mean_val >= 10:
                print(f"{display_name:<30} {mean_val:>15,.2f} {median_val:>15,.2f} {std_val:>15,.2f}")
            else:
                print(f"{display_name:<30} {mean_val:>15.2f} {median_val:>15.2f} {std_val:>15.2f}")
        else:
            print(f"{display_name:<30} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
    
    print("="*70)
    
    # Additional useful info
    print("\n" + "="*70)
    print("ADDITIONAL DATASET INFO")
    print("="*70)
    print(f"\nTotal games in dataset: {len(df):,}")
    print(f"Games with missing price: {df['price'].isnull().sum():,}")
    print(f"Games with missing ratings: {df['positive_ratings'].isnull().sum():,}")
    print(f"Games with missing owners: {df['owners_mid'].isnull().sum():,}")
    
    # Price distribution
    print(f"\n--- Price Distribution ---")
    print(f"Free games (price = 0): {(df['price'] == 0).sum():,} ({(df['price'] == 0).mean()*100:.2f}%)")
    print(f"Paid games: {(df['price'] > 0).sum():,} ({(df['price'] > 0).mean()*100:.2f}%)")
    print(f"Max price: ${df['price'].max():.2f}")
    
    # Owners distribution
    print(f"\n--- Owners Distribution ---")
    if 'owners_mid' in df.columns:
        df_with_owners = df[df['owners_mid'].notna()]
        print(f"Games with < 10k owners: {(df_with_owners['owners_mid'] < 10000).sum():,}")
        print(f"Games with 10k-100k owners: {((df_with_owners['owners_mid'] >= 10000) & (df_with_owners['owners_mid'] < 100000)).sum():,}")
        print(f"Games with 100k-1M owners: {((df_with_owners['owners_mid'] >= 100000) & (df_with_owners['owners_mid'] < 1000000)).sum():,}")
        print(f"Games with 1M+ owners: {(df_with_owners['owners_mid'] >= 1000000).sum():,}")
    
    print("\n" + "="*70)
    print("LATEX TABLE FORMAT")
    print("="*70)
    print("\n% Copy this into your LaTeX document (replace Table 6):\n")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("\\textbf{Variable} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Std Dev} \\\\")
    print("\\midrule")
    
    for col, display_name in variables.items():
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            # Format for LaTeX
            if mean_val >= 1000:
                print(f"{display_name} & {mean_val:,.0f} & {median_val:,.0f} & {std_val:,.0f} \\\\")
            elif mean_val >= 10:
                print(f"{display_name} & {mean_val:,.2f} & {median_val:,.2f} & {std_val:,.2f} \\\\")
            else:
                print(f"{display_name} & {mean_val:.2f} & {median_val:.2f} & {std_val:.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Statistical Summary of Key Numerical Variables}")
    print("\\label{tab:stats_summary}")
    print("\\end{table}")
    
    print("\n" + "="*70)
    print("Script completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()