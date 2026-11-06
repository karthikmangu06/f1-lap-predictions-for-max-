"""
train_model.py
==============
Phase 3: Machine Learning — Train & Export the XGBoost Lap-Time Predictor

This script reads the accumulated lap data from `max_verstappen_laps.csv`,
engineers a scikit-learn Pipeline that handles categorical encoding and
regression, evaluates model performance, and serialises the trained pipeline
to `max_lap_predictor.pkl` for use by the Streamlit app.

Usage:
    python train_model.py

Dependencies:
    pip install pandas scikit-learn xgboost joblib
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

INPUT_CSV      = r"D:\weekend project\max_verstappen_laps.csv"
OUTPUT_MODEL   = r"D:\weekend project\max_lap_predictor.pkl"

# Upgrade 2: Expanded feature set to match the 9-column schema produced
# by the upgraded extract_data.py. Column order must be identical here
# and in app.py's build_input_df() so predict() receives the right shape.
FEATURES = [
    'LapNumber',           # numeric
    'Compound',            # categorical → OHE
    'TyreLife',            # numeric
    'TyreLife_Squared',    # numeric (non-linear tyre degradation signal)
    'Track',               # categorical → OHE
    'Stint',               # numeric (stint number within the race)
    'FuelLapsRemaining',   # numeric (fuel load proxy: laps left in race)
    'IsClearTrack',        # numeric (binary flag 0/1)
    'IsPushing'            # numeric (binary flag 0/1)
]

NUMERIC_FEATURES     = ['LapNumber', 'TyreLife', 'TyreLife_Squared', 'Stint', 'FuelLapsRemaining', 'IsClearTrack', 'IsPushing']
CATEGORICAL_FEATURES = ['Compound', 'Track']
TARGET               = "LapTime_ms"

# Tyre compound categories expected by the encoder — order matters for
# deterministic one-hot encoding across train/predict calls.
COMPOUND_CATEGORIES = [["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]]

# Train / test split ratio
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# XGBoost hyper-parameters (as specified)
XGB_PARAMS = {
    "n_estimators"  : 100,
    "learning_rate" : 0.1,
    "random_state"  : RANDOM_STATE,
    "tree_method"   : "hist",     # fast histogram-based algorithm
    "verbosity"     : 0,          # suppress XGBoost output
}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the master lap CSV produced by extract_data.py.

    Raises:
        FileNotFoundError : If the CSV does not exist yet.
        ValueError        : If the CSV is empty or missing required columns.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found: '{filepath}'. "
            "Run extract_data.py first to generate the dataset."
        )

    if df.empty:
        raise ValueError(f"The data file '{filepath}' is empty. Add more sessions.")

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the CSV: {missing_cols}"
        )

    print(f"[INFO] Loaded {len(df)} rows from '{filepath}'.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPARE FEATURES & TARGET
# ──────────────────────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """
    Separate features (X) from the target variable (y) and perform
    light preprocessing, including dynamic outlier clipping.
    """
    df = df.copy()

    # Normalise compound labels
    df["Compound"] = df["Compound"].str.upper().str.strip()

    # Keep only rows whose Compound is one of the expected categories
    df = df[df["Compound"].isin(COMPOUND_CATEGORIES[0])]

    # Drop any final NaN rows across all required columns
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df.dropna(subset=all_features + [TARGET], inplace=True)

    # ── THE OUTLIER CLIPPER ──────────────────────────────────────────────────
    # Calculate the median lap time for each track.
    # Drop any lap that is more than 25% slower than the track's median.
    # This keeps wet weather laps (usually ~15% slower) but destroys 
    # 3-minute pit stops and 20-minute red flags that corrupt the model.
    track_medians = df.groupby('Track')[TARGET].transform('median')
    before_clip = len(df)
    df = df[df[TARGET] <= (track_medians * 1.25)]
    after_clip = len(df)
    print(f"[INFO] Outlier Clipper removed {before_clip - after_clip} corrupted laps.")
    # ─────────────────────────────────────────────────────────────────────────

    X = df[FEATURES]
    y = df[TARGET]

    print(f"[INFO] Feature matrix shape : {X.shape}")
    print(f"[INFO] Target series shape  : {y.shape}")
    
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD THE SCIKIT-LEARN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Construct a scikit-learn Pipeline that:

    1. ColumnTransformer
       ├─ OneHotEncoder  → applied to 'Compound' column
       │    categories   : [['SOFT', 'MEDIUM', 'HARD']]
       │    handle_unknown: 'ignore'  (gracefully handles unseen compounds)
       │    sparse_output: False      (dense array, compatible with pandas)
       └─ 'passthrough'  → applied to numeric features (no scaling needed for XGBoost)

    2. XGBRegressor
       Gradient-boosted decision trees — excels on tabular data without
       feature scaling, making them an ideal match for F1 lap-time regression.

    Returns:
        An unfitted scikit-learn Pipeline.
    """
    # Preprocessor: encode Compound, Track, and Rainfall → one-hot vectors.
    # Upgrade 2: All three categorical features are encoded in a single
    # transformer block for consistency.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "compound_encoder",
                OneHotEncoder(
                    categories=COMPOUND_CATEGORIES,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Compound"],
            ),
            (
                # Track: auto-inferred categories; unknown tracks get
                # all-zero OHE vectors — XGBoost falls back to globally
                # learned base patterns.
                "track_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Track"],
            ),
        ],
        remainder="passthrough",  # numeric columns pass through unchanged
    )

    # Full pipeline: preprocess → XGBoost regression
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor",    XGBRegressor(**XGB_PARAMS)),
        ]
    )

    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN, EVALUATE & EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Split data, fit the pipeline, compute evaluation metrics, and return
    the trained pipeline.

    Metrics reported:
      • R² (coefficient of determination) — higher is better; 1.0 is perfect.
      • MAE (Mean Absolute Error in ms)   — lower is better.

    Args:
        pipeline : Unfitted scikit-learn Pipeline.
        X        : Feature DataFrame.
        y        : Target Series (LapTime_ms).

    Returns:
        The fitted Pipeline.
    """
    print(f"\n[INFO] Splitting data: {int((1 - TEST_SIZE) * 100)}% train / "
          f"{int(TEST_SIZE * 100)}% test  (random_state={RANDOM_STATE})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing  samples : {len(X_test)}")

    print("\n[INFO] Training XGBoost pipeline …")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    results = X_test.copy()
    results['Actual'] = y_test.values
    results['Predicted'] = y_pred
    results['Error_ms'] = results['Predicted'] - results['Actual']
    results['Error_sec'] = results['Error_ms'] / 1000
    results['AbsError_sec'] = results['Error_sec'].abs()
    r2  = r2_score(y_test, y_pred)


    mae = mean_absolute_error(y_test, y_pred)

    # Convert MAE from ms to seconds for a more intuitive reading
    mae_sec = mae / 1000.0

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  R2 Score                       : {r2:.4f}   (1.00 = perfect)")
    # Upgrade 2: MAE displayed in the requested format
    print(f"  Mean Absolute Error (MAE)      : {mae_sec:.3f} seconds per lap")
    print(f"  Mean Absolute Error (MAE raw)  : {mae:.1f} ms")
    print("=" * 50)
    print("\n[INFO] Sample predictions:")
    print(results[['LapNumber','Compound','TyreLife','FuelLapsRemaining','Actual','Predicted','Error_sec']].head(10))
    print("\n[INFO] Worst predictions (top 10):")
    print(results.sort_values('AbsError_sec', ascending=False).head(10))
    print("\n[INFO] Error by Compound:")
    print(results.groupby('Compound')['AbsError_sec'].mean())
    results['RacePhase'] = pd.cut(
    results['LapNumber'],
    bins=[0, 20, 40, 70],
    labels=['Early', 'Mid', 'Late']
    )

    print("\n[INFO] Error by Race Phase:")
    print(results.groupby('RacePhase')['AbsError_sec'].mean())
    return pipeline





def export_model(pipeline: Pipeline, filepath: str) -> None:
    """
    Serialise the trained pipeline to disk using joblib (efficient for
    numpy-heavy objects like sklearn estimators).

    Args:
        pipeline : The fitted scikit-learn Pipeline to save.
        filepath : Destination .pkl file path.
    """
    joblib.dump(pipeline, filepath)
    print(f"\n[INFO] Model saved to '{filepath}'")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the full training pipeline."""
    print("=" * 60)
    print("  MAX VERSTAPPEN LAP TIME PREDICTOR — MODEL TRAINING")
    print("=" * 60)

    # 1. Load raw data
    df = load_data(INPUT_CSV)

    # 2. Prepare features and target
    X, y = prepare_xy(df)

    if len(X) < 10:
        raise ValueError(
            "Fewer than 10 usable laps found. Run extract_data.py for "
            "more sessions before training."
        )

    # 3. Build the ML pipeline
    pipeline = build_pipeline()

    # 4. Train, evaluate and retrieve fitted pipeline
    fitted_pipeline = train_and_evaluate(pipeline, X, y)

    # 5. Persist to disk
    export_model(fitted_pipeline, OUTPUT_MODEL)

    print("\n[DONE] Training complete. Run 'streamlit run app.py' to launch the simulator.")

# Extract Feature Importances
    print("\n[INFO] Feature Importances:")
    try:
        feature_names = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = fitted_pipeline.named_steps['regressor'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

if __name__ == "__main__":
    main()
