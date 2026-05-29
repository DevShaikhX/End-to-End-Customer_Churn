"""teleco_data_validation.py

Comprehensive data validation profiling script for the Kaggle Telco Customer Churn dataset.

Features:
- Loads CSV and prints strict structural shape (records x features)
- Identifies and fixes the "TotalCharges Missing Value Bug" (spaces -> NaN)
- Splits into train/test (stratified) and imputes TotalCharges using training median
- Classifies columns into: Numerical Continuous, Numerical Discrete,
  Nominal Categorical, Ordinal Categorical
- Computes Churn class imbalance (counts and percentages)
- Outputs a structured operational report to the console

Usage:
    python -m validation.teleco_data_validation --path data/WA_Fn-UseC_-Telco-Customer-Churn.csv

"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame without automatically converting whitespace-only cells to NaN.

    We keep default pandas behavior but ensure whitespace-only strings are preserved so
    we can explicitly detect and convert them for `TotalCharges`.
    """
    df = pd.read_csv(path, low_memory=False)
    return df


def report_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Return (n_records, n_features) for a strict structural shape report."""
    return df.shape


def fix_total_charges(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, int, float]:
    """Handle the TotalCharges whitespace bug and impute missing using training-set median.

    Steps:
    - Detect rows where TotalCharges is whitespace (e.g., '   ').
    - Convert those to np.nan and count them.
    - Cast column to float64 via `pd.to_numeric(..., errors='coerce')`.
    - Perform a stratified train/test split on `Churn` to compute training median.
    - Impute missing TotalCharges in both sets using training median (prevents leakage).

    Returns: (df_after_imputation, missing_count, training_median)
    """
    if 'TotalCharges' not in df.columns:
        raise KeyError('Column "TotalCharges" not found in DataFrame')

    # Detect whitespace-only entries (including empty strings)
    tc_series = df['TotalCharges'].astype(str)
    whitespace_mask = tc_series.str.strip() == ''
    missing_count = int(whitespace_mask.sum())

    # Convert whitespace-only -> NaN
    df = df.copy()
    df.loc[whitespace_mask, 'TotalCharges'] = np.nan

    # Cast to numeric float64 (coerce any residual non-numeric to NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype('float64')

    # Ensure target exists for stratification
    if 'Churn' not in df.columns:
        # If no churn column, compute median on whole df
        training_median = float(df['TotalCharges'].median(skipna=True))
        df['TotalCharges'] = df['TotalCharges'].fillna(training_median)
        return df, missing_count, training_median

    # Perform stratified split using indices to retain original DataFrame
    stratify_col = df['Churn']
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=stratify_col, random_state=random_state
    )

    training_median = float(df.loc[train_idx, 'TotalCharges'].median(skipna=True))

    # Impute using the training median for both train and test partitions
    df.loc[train_idx, 'TotalCharges'] = df.loc[train_idx, 'TotalCharges'].fillna(training_median)
    df.loc[test_idx, 'TotalCharges'] = df.loc[test_idx, 'TotalCharges'].fillna(training_median)

    return df, missing_count, training_median


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify columns into four sub-profiles.

    Heuristics used:
    - Numerical columns: pandas numeric dtypes. Float dtypes -> Continuous; Integer dtypes -> Discrete.
    - Object/category columns: default to Nominal Categorical unless recognized as Ordinal.
      Known ordinal columns (from Telco dataset) are detected explicitly (e.g., `Contract`).
    """
    profiles = {
        'numerical_continuous': [],
        'numerical_discrete': [],
        'nominal_categorical': [],
        'ordinal_categorical': [],
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col].dtype):
            profiles['numerical_continuous'].append(col)
        else:
            profiles['numerical_discrete'].append(col)

    # Known ordinals for Telco dataset
    KNOWN_ORDINALS = {
        'Contract': ['Month-to-month', 'One year', 'Two year']
    }

    # Object / category columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        # Skip numeric-like that are object due to bad typing
        if col in numeric_cols:
            continue

        vals = list(pd.Series(df[col].dropna().unique()))
        # Recognize known ordinals
        if col in KNOWN_ORDINALS:
            profiles['ordinal_categorical'].append(col)
            continue

        # Heuristic: if values include 'Month-to-month' or ordering tokens, treat as ordinal
        lowered = [str(v).lower() for v in vals]
        if any('month' in s for s in lowered) and any('year' in s for s in lowered):
            profiles['ordinal_categorical'].append(col)
            continue

        # Default to nominal categorical
        profiles['nominal_categorical'].append(col)

    return profiles


def compute_target_distribution(df: pd.DataFrame, target: str = 'Churn') -> Dict[str, Dict[str, float]]:
    """Compute class counts and percentages for the binary target.

    Returns a dict: {'counts': {'Yes': n, 'No': m}, 'percent': {'Yes': p, 'No': q}}
    """
    result = {'counts': {}, 'percent': {}}
    if target not in df.columns:
        return result

    counts = df[target].value_counts(dropna=False)
    total = len(df)
    for cls in counts.index:
        label = str(cls)
        cnt = int(counts.loc[cls])
        pct = float(cnt) / total * 100.0
        result['counts'][label] = cnt
        result['percent'][label] = pct

    return result


def generate_report(df: pd.DataFrame, path: str) -> None:
    """Run full validation profile and print a structured operational report to console."""
    n_rows, n_cols = report_shape(df)

    print('=' * 80)
    print('Telco Customer Churn — Data Validation Profile')
    print('=' * 80)
    print(f'Source file: {path}')
    print(f'Strict shape: {n_rows} records x {n_cols} features')
    print()

    # TotalCharges fix (detect, cast, impute)
    df_fixed, missing_count, training_median = fix_total_charges(df)
    print('TotalCharges:')
    print(f' - Detected whitespace-only missing rows: {missing_count}')
    print(f' - Cast to dtype: {df_fixed["TotalCharges"].dtype}')
    print(f' - Imputed missing values using training median: {training_median:.4f}')
    print()

    # Column classification
    profiles = classify_columns(df_fixed)
    print('Column Sub-profiles:')
    for k, cols in profiles.items():
        print(f' - {k} ({len(cols)}): {cols}')
    print()

    # Target distribution
    target_stats = compute_target_distribution(df_fixed, target='Churn')
    if target_stats:
        yes_count = target_stats['counts'].get('Yes', 0)
        no_count = target_stats['counts'].get('No', 0)
        yes_pct = target_stats['percent'].get('Yes', 0.0)
        no_pct = target_stats['percent'].get('No', 0.0)
        print('Target `Churn` distribution:')
        print(f' - Yes (Churn): {yes_count} ({yes_pct:.2f}%)')
        print(f' - No  (No Churn): {no_count} ({no_pct:.2f}%)')
        if no_count > 0:
            ratio = yes_count / no_count
            print(f' - Ratio (Yes / No): {ratio:.4f}')
    else:
        print('Target `Churn` not found in dataset; distribution not computed.')

    print('\nValidation complete.')
    print('=' * 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Telco Churn data validation profile')
    p.add_argument('--path', type=str, default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(args.path)
    generate_report(df, args.path)


if __name__ == '__main__':
    main()
