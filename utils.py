"""utils.py

Core data transformation and preprocessing utilities for the Telco Customer Churn ML pipeline.

Ensures strict separation of concerns:
- Data loading and cleaning (target mapping, whitespace anomaly resolution)
- Feature preprocessing with zero data leakage (fit on train, apply to test)
- Categorical encoding (OneHotEncoder for multi-class, binary encoding for binary features)
- Numerical scaling (StandardScaler fit exclusively on training set)

Usage:
    from utils import load_and_clean_base_data, preprocess_features
    
    df = load_and_clean_base_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test, feature_names = preprocess_features(X_train, X_test)
"""

from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_and_clean_base_data(filepath: str) -> pd.DataFrame:
    """Load the Telco CSV and perform baseline cleaning.
    
    Steps:
    1. Read CSV with low_memory=False to avoid dtype inference warnings.
    2. Map target variable 'Churn' (Yes → 1, No → 0) to binary integer vector.
    3. Resolve TotalCharges whitespace bug: detect empty/space-only strings → np.nan → convert to float64.
    4. Return cleaned DataFrame ready for feature preprocessing.
    
    Args:
        filepath: Absolute or relative path to the Telco CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataset with binary target and resolved TotalCharges.
        
    Raises:
        FileNotFoundError: If filepath does not exist.
        KeyError: If required columns ('Churn', 'TotalCharges') are missing.
    """
    # Load dataset
    df = pd.read_csv(filepath, low_memory=False)
    
    # Validate required columns
    required_cols = {'Churn', 'TotalCharges'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Map Churn target: Yes → 1, No → 0
    churn_map = {'Yes': 1, 'No': 0}
    df['Churn'] = df['Churn'].map(churn_map)
    
    # Fix TotalCharges whitespace bug
    # Detect rows where TotalCharges is whitespace-only or empty
    tc_series = df['TotalCharges'].astype(str)
    whitespace_mask = tc_series.str.strip() == ''
    df.loc[whitespace_mask, 'TotalCharges'] = np.nan
    
    # Convert to numeric, coerce invalid values to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').astype('float64')
    
    return df


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target_col: str = 'Churn',
    numerical_features: List[str] = None,
    binary_categorical_features: List[str] = None,
    multi_class_categorical_features: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Preprocess train and test feature matrices with zero data leakage.
    
    Pipeline:
    1. Separate numerical, binary categorical, and multi-class categorical features.
    2. Fit OneHotEncoder (drop='first') on multi-class categoricals using ONLY training data.
    3. Encode binary categoricals using LabelEncoder (fit on train, transform both).
    4. Fit StandardScaler EXCLUSIVELY on training numerical features.
    5. Transform both train and test using the pre-fit scaler.
    6. Concatenate all preprocessed feature arrays.
    7. Return (X_train_final, X_test_final, feature_names) with zero leakage guarantee.
    
    Args:
        X_train: Training feature DataFrame (must include target_col if present).
        X_test: Test feature DataFrame.
        target_col: Name of target column to exclude from features (default: 'Churn').
        numerical_features: List of numerical column names. If None, auto-detect numeric dtypes.
        binary_categorical_features: List of binary categorical column names. If None, auto-detect.
        multi_class_categorical_features: List of multi-class categorical column names. If None, auto-detect.
        
    Returns:
        Tuple of:
        - X_train_preprocessed (np.ndarray): Transformed training features.
        - X_test_preprocessed (np.ndarray): Transformed test features.
        - feature_names (List[str]): Final feature column names after all transformations.
        
    Raises:
        ValueError: If train and test have incompatible column sets.
    """
    # Validate inputs
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train and test must have same number of features. "
            f"Got {X_train.shape[1]} (train) vs {X_test.shape[1]} (test)."
        )
    
    # Remove target column if present
    X_train = X_train.drop(columns=[target_col], errors='ignore').copy()
    X_test = X_test.drop(columns=[target_col], errors='ignore').copy()
    
    # Auto-detect feature types if not explicitly provided
    if numerical_features is None:
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if binary_categorical_features is None or multi_class_categorical_features is None:
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if binary_categorical_features is None and multi_class_categorical_features is None:
            # Auto-split: if unique values <= 2, treat as binary; else as multi-class
            binary_categorical_features = []
            multi_class_categorical_features = []
            for col in categorical_features:
                n_unique = X_train[col].nunique()
                if n_unique <= 2:
                    binary_categorical_features.append(col)
                else:
                    multi_class_categorical_features.append(col)
    
    # ==================== NUMERICAL FEATURES ====================
    # Fit StandardScaler ONLY on training data (prevents leakage)
    # First impute missing values using training set median
    imputer = SimpleImputer(strategy='median')
    X_train_numerical_imputed = imputer.fit_transform(X_train[numerical_features])
    X_test_numerical_imputed = imputer.transform(X_test[numerical_features])
    
    scaler = StandardScaler()
    X_train_numerical = scaler.fit_transform(X_train_numerical_imputed)
    X_test_numerical = scaler.transform(X_test_numerical_imputed)
    
    # ==================== BINARY CATEGORICAL FEATURES ====================
    # For binary features, use LabelEncoder or simple binary mapping
    X_train_binary = []
    X_test_binary = []
    binary_feature_names = []
    
    for col in binary_categorical_features:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        
        X_train_binary.append(le.transform(X_train[col].astype(str)).reshape(-1, 1))
        X_test_binary.append(le.transform(X_test[col].astype(str)).reshape(-1, 1))
        
        # Create feature name (e.g., "feature_0" or use the encoded class names)
        binary_feature_names.append(f"{col}_encoded")
    
    X_train_binary_arr = np.hstack(X_train_binary) if X_train_binary else np.empty((X_train.shape[0], 0))
    X_test_binary_arr = np.hstack(X_test_binary) if X_test_binary else np.empty((X_test.shape[0], 0))
    
    # ==================== MULTI-CLASS CATEGORICAL FEATURES ====================
    # Fit OneHotEncoder ONLY on training data (prevents leakage)
    ohe_feature_names = []
    if multi_class_categorical_features:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        ohe.fit(X_train[multi_class_categorical_features])
        
        X_train_ohe = ohe.transform(X_train[multi_class_categorical_features])
        X_test_ohe = ohe.transform(X_test[multi_class_categorical_features])
        
        # Extract feature names from OneHotEncoder
        ohe_feature_names = ohe.get_feature_names_out(multi_class_categorical_features).tolist()
    else:
        X_train_ohe = np.empty((X_train.shape[0], 0))
        X_test_ohe = np.empty((X_test.shape[0], 0))
    
    # ==================== CONCATENATE ALL ====================
    X_train_final = np.hstack([
        X_train_numerical,
        X_train_binary_arr,
        X_train_ohe
    ])
    
    X_test_final = np.hstack([
        X_test_numerical,
        X_test_binary_arr,
        X_test_ohe
    ])
    
    # Build final feature names list
    feature_names = (
        numerical_features +
        binary_feature_names +
        ohe_feature_names
    )
    
    return X_train_final, X_test_final, feature_names


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test with optional stratification on target.
    
    Args:
        df: Complete DataFrame with target column.
        target_col: Name of target column.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify split on target (default: True).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    return X_train, X_test, y_train, y_test


def create_feature_summary(feature_names: List[str]) -> Dict[str, int]:
    """Create a summary of encoded feature counts by type.
    
    Args:
        feature_names: List of feature names after preprocessing.
        
    Returns:
        Dictionary with feature type counts.
    """
    summary = {
        'total_features': len(feature_names),
        'numerical_features': sum(1 for f in feature_names if not any(c in f for c in ['_x', '_encoded'])),
        'encoded_features': sum(1 for f in feature_names if '_encoded' in f or '_x' in f),
    }
    return summary


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split with explicit minority class balancing.
    
    **Critical:** Uses `stratify=y` parameter to guarantee that both training and test sets
    maintain the same target class distribution as the original dataset. This is essential
    for imbalanced datasets (e.g., Churn: 26.5% Yes, 73.5% No).
    
    Allocation: 80% training, 20% testing (fixed via test_size=0.2).
    Random state fixed at 42 for reproducibility across multiple runs.
    
    Args:
        X: Feature DataFrame.
        y: Target Series (with class imbalance).
        test_size: Fraction for test set (default: 0.2 → 80/20 split).
        random_state: Random seed for reproducibility (default: 42).
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with stratified class distribution.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    
    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # ← CRITICAL: Maintains class ratios in train/test
    )
    
    return X_train, X_test, y_train, y_test


def apply_smote_balancing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE (Synthetic Minority Over-sampling Technique) to training data ONLY.
    
    **CRITICAL DATA LEAKAGE PREVENTION:**
    
    SMOTE is applied EXCLUSIVELY to the training subset. Applying SMOTE to test/validation
    data causes catastrophic validation inflation because:
    
    1. SMOTE synthesizes new minority class samples by interpolating between existing
       minority samples in the feature space.
    2. If applied to test data, it artificially increases the number of minority samples
       beyond reality, inflating recall and f1-score metrics.
    3. The synthetic samples in test data do not represent real customer behavior,
       leading to overly optimistic performance estimates.
    4. Cross-validation and production deployment will show significantly lower
       performance than reported during development.
    
    **Correct Pipeline:**
    ✓ Stratified split (train 80% / test 20%)
    ✓ SMOTE applied to X_train, y_train only
    ✓ Models trained on balanced training set
    ✓ Evaluation on unbalanced, untouched test set (real-world scenario)
    
    Args:
        X_train: Training feature matrix (preprocessed numpy array).
        y_train: Training target vector (binary: 0/1).
        random_state: Random seed for reproducibility (default: 42).
        
    Returns:
        Tuple of (X_train_balanced, y_train_balanced) with synthetic minority samples.
        
    Raises:
        ImportError: If imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn not found. Install with: pip install imbalanced-learn"
        )
    
    # Instantiate SMOTE with fixed random state for reproducibility
    smote = SMOTE(random_state=random_state)
    
    # Apply SMOTE ONLY to training data
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced


def report_class_distribution(
    y: pd.Series,
    dataset_name: str = "Dataset"
) -> Dict[str, Any]:
    """Generate a detailed class distribution report.
    
    Args:
        y: Target Series.
        dataset_name: Label for reporting (e.g., 'Training', 'Test').
        
    Returns:
        Dictionary with counts and percentages.
    """
    counts = y.value_counts().sort_index()
    total = len(y)
    percentages = (counts / total * 100).round(2)
    
    report = {
        'dataset': dataset_name,
        'total_samples': total,
        'class_counts': counts.to_dict(),
        'class_percentages': percentages.to_dict(),
        'imbalance_ratio': counts.max() / counts.min() if len(counts) > 1 else 1.0
    }
    
    return report


if __name__ == '__main__':
    # ============================================================================
    # TELCO ML PIPELINE — COMPLETE DATA PREPARATION & BALANCING WORKFLOW
    # ============================================================================
    print("\n" + "=" * 80)
    print("TELCO ML PIPELINE — Data Utilities & SMOTE Balancing Demo")
    print("=" * 80)
    
    DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # ==================== STEP 1: Load & Clean ====================
    print("\n[STEP 1] Loading & Cleaning Base Data")
    print("-" * 80)
    df = load_and_clean_base_data(DATA_PATH)
    print(f"✓ Loaded dataset: {df.shape[0]} records × {df.shape[1]} features")
    
    orig_dist = report_class_distribution(df['Churn'], "Original Dataset")
    print(f"\nOriginal target distribution (Churn):")
    for cls, count in orig_dist['class_counts'].items():
        pct = orig_dist['class_percentages'][cls]
        print(f"  Class {cls}: {count:,} ({pct}%)")
    print(f"Imbalance ratio: {orig_dist['imbalance_ratio']:.3f}:1")
    
    # ==================== STEP 2: Extract Features & Target ====================
    print("\n[STEP 2] Extracting Features & Target")
    print("-" * 80)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # ==================== STEP 3: Stratified Train/Test Split ====================
    print("\n[STEP 3] Stratified Train/Test Split (80/20)")
    print("-" * 80)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train set: {X_train.shape[0]} records ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✓ Test set:  {X_test.shape[0]} records ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Verify stratification
    train_dist = report_class_distribution(y_train, "Training Set (Before SMOTE)")
    test_dist = report_class_distribution(y_test, "Test Set")
    
    print(f"\nTraining set class distribution (BEFORE SMOTE):")
    for cls, count in train_dist['class_counts'].items():
        pct = train_dist['class_percentages'][cls]
        print(f"  Class {cls}: {count:,} ({pct}%)")
    print(f"Imbalance ratio (train): {train_dist['imbalance_ratio']:.3f}:1")
    
    print(f"\nTest set class distribution (UNTOUCHED):")
    for cls, count in test_dist['class_counts'].items():
        pct = test_dist['class_percentages'][cls]
        print(f"  Class {cls}: {count:,} ({pct}%)")
    print(f"Imbalance ratio (test): {test_dist['imbalance_ratio']:.3f}:1 ← Reflects real-world scenario")
    
    # ==================== STEP 4: Feature Preprocessing ====================
    print("\n[STEP 4] Feature Preprocessing (StandardScaler + Encoding)")
    print("-" * 80)
    X_train_processed, X_test_processed, feature_names = preprocess_features(X_train, X_test)
    print(f"✓ Processed train features: {X_train_processed.shape}")
    print(f"✓ Processed test features:  {X_test_processed.shape}")
    print(f"✓ Total engineered features: {len(feature_names)}")
    print(f"  Sample feature names: {feature_names[:5]}")
    
    # ==================== STEP 5: SMOTE Balancing (Training Only) ====================
    print("\n[STEP 5] Applying SMOTE to Training Data ONLY")
    print("-" * 80)
    print("\n⚠️  CRITICAL DATA LEAKAGE PREVENTION:")
    print("   SMOTE will be applied ONLY to training set.")
    print("   Test set remains UNTOUCHED to reflect real-world imbalance.\n")
    
    X_train_balanced, y_train_balanced = apply_smote_balancing(
        X_train_processed, y_train.values, random_state=42
    )
    
    print(f"✓ SMOTE applied successfully")
    print(f"✓ Original training samples: {X_train_processed.shape[0]}")
    print(f"✓ Balanced training samples: {X_train_balanced.shape[0]} (synthetic samples added)")
    
    # Report balanced distribution
    y_train_balanced_series = pd.Series(y_train_balanced)
    balanced_dist = report_class_distribution(y_train_balanced_series, "Training Set (After SMOTE)")
    
    print(f"\nTraining set class distribution (AFTER SMOTE):")
    for cls, count in balanced_dist['class_counts'].items():
        pct = balanced_dist['class_percentages'][cls]
        print(f"  Class {cls}: {count:,} ({pct}%)")
    print(f"Imbalance ratio (after SMOTE): {balanced_dist['imbalance_ratio']:.3f}:1 ← Perfect balance achieved")
    
    # ==================== STEP 6: Summary ====================
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"""
✓ Dataset loaded:             {df.shape[0]:,} records × {df.shape[1]} features
✓ Features engineered:        {len(feature_names)} dimensions
✓ Stratified train/test:      {X_train.shape[0]:,} train / {X_test.shape[0]:,} test
✓ SMOTE balancing applied:    {X_train_balanced.shape[0]:,} training samples
✓ Zero data leakage enforced: Test set pristine, unbalanced (real-world)
✓ Ready for model training:   X_train_balanced, y_train_balanced, X_test_processed, y_test

TRAINING CONFIGURATION:
  • Scaler/Encoder: Fit on original training data, applied to both sets
  • SMOTE: Applied only to training set after feature preprocessing
  • Random state: 42 (reproducible across all stochastic operations)
  • Class ratio (before): {orig_dist['imbalance_ratio']:.3f}:1 → (after SMOTE): {balanced_dist['imbalance_ratio']:.3f}:1
""")
