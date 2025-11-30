import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
RANDOM_CSV = os.path.join("../data/random_faults.csv")

ARTIFACT_DIR = "artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_model.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_cols.pkl")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "training_report.txt")
CM_PATH = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
FI_PATH = os.path.join(ARTIFACT_DIR, "feature_importances.png")

# Default CSV paths for Option B (can be overridden by CLI)
DEFAULT_CSV_INPUT = "realtime_stream.csv"
DEFAULT_CSV_OUTPUT = "realtime_with_preds.csv"

CLASS_NAMES = {0: "normal", 5: "fault_both", 6: "fault_clogged", 7: "fault_filling"}

FEATURE_COLS = [
    "level_real",
    "flow_in_real",
    "flow_out_real",
    "pressure_real",
    "pump_current",
    "valve_position",
    "pump_state",
    "controller_setpoint",
    "is_valve_clogged",
    "is_filling",
    "filling_rate",
]


def ensure_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_datasets(random_path: str = RANDOM_CSV) -> pd.DataFrame:
    if not os.path.exists(random_path):
        raise FileNotFoundError(random_path)
    return pd.read_csv(random_path)


def preprocess_df(
    df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLS
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = df.copy()
    if "label" not in df.columns:
        raise KeyError("label missing")

    # Convert numeric-looking strings to proper numeric
    df = df.apply(pd.to_numeric, errors="ignore")

    # Ensure all feature columns exist
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0.0

    df[feature_cols] = df[feature_cols].fillna(0.0)

    X = df[feature_cols].astype(float).values
    y = df["label"].astype(int).values
    return X, y, feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    save_artifacts: bool = True,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int = 15,
):
    ensure_artifact_dir()
    X, y, feature_cols = preprocess_df(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(
        model,
        scaler.transform(X),
        y,
        cv=cv,
        scoring="f1_macro",
    )

    if save_artifacts:
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(feature_cols, FEATURES_PATH)

        with open(REPORT_PATH, "w") as fh:
            fh.write(report + "\n")
            fh.write(str(cm) + "\n")
            fh.write(str(cv_scores) + "\n")

    _plot_confusion_matrix_and_save(cm)
    _plot_feature_importances_and_save(model, feature_cols)

    print(report)
    print(cv_scores)

    return model, scaler, feature_cols


def _plot_confusion_matrix_and_save(cm: np.ndarray):
    n = cm.shape[0]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(i) for i in range(n)],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)


def _plot_feature_importances_and_save(
    model: RandomForestClassifier,
    feature_cols: List[str],
):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    ordered_names = [feature_cols[i] for i in idx]
    ordered_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(ordered_vals)), ordered_vals[::-1])
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names[::-1])
    plt.tight_layout()
    fig.savefig(FI_PATH, dpi=150)
    plt.close(fig)


def load_artifacts():
    if not (
        os.path.exists(SCALER_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(FEATURES_PATH)
    ):
        raise FileNotFoundError("Artifacts missing")
    return (
        joblib.load(SCALER_PATH),
        joblib.load(MODEL_PATH),
        joblib.load(FEATURES_PATH),
    )


def predict_single(
    data_point: dict,
    scaler=None,
    model=None,
    feature_cols=None,
) -> dict:
    if scaler is None or model is None or feature_cols is None:
        scaler, model, feature_cols = load_artifacts()

    x = np.array([float(data_point.get(f, 0.0)) for f in feature_cols]).reshape(1, -1)

    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    prob = float(model.predict_proba(x_scaled).max())

    return {
        "predicted_class": pred,
        "class_name": CLASS_NAMES.get(pred, str(pred)),
        "confidence": prob,
    }


# ---------------------------------------------------------------------
# Option B: CSV inference pipeline
# ---------------------------------------------------------------------
def csv_inference(
    csv_input: str = DEFAULT_CSV_INPUT,
    csv_output: str = DEFAULT_CSV_OUTPUT,
):
    """
    Read a CSV of sensor data, run the RF model, and write predictions.

    The input CSV must contain the same feature columns as FEATURE_COLS.
    The output CSV will add:
        - predicted_class
        - class_name
        - confidence
        - anomaly_flag (0 = normal, 1 = any fault)
    """
    if not os.path.exists(csv_input):
        raise FileNotFoundError(csv_input)

    scaler, model, feature_cols = load_artifacts()
    df = pd.read_csv(csv_input)

    if df.empty:
        print(f"[INFO] {csv_input} is empty, nothing to process.")
        return

    # Ensure required columns exist, but do not modify original values
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0.0

    # Build feature matrix
    X = df[feature_cols].astype(float).values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled).max(axis=1)

    df["predicted_class"] = preds.astype(int)
    df["class_name"] = [CLASS_NAMES.get(int(p), str(int(p))) for p in preds]
    df["confidence"] = probs.astype(float)
    df["anomaly_flag"] = (df["predicted_class"] != 0).astype(int)

    df.to_csv(csv_output, index=False)
    print(f"[OK] CSV inference complete. Saved to {csv_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--csv-infer", action="store_true")

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--csv-input", type=str, default=DEFAULT_CSV_INPUT)
    parser.add_argument("--csv-output", type=str, default=DEFAULT_CSV_OUTPUT)

    args = parser.parse_args()

    if args.train:
        df = load_datasets()
        train_and_evaluate(
            df,
            save_artifacts=True,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    elif args.evaluate_only:
        df = load_datasets()
        X, y, feature_cols = preprocess_df(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        scaler, model, _ = load_artifacts()
        preds = model.predict(scaler.transform(X_test))
        print(classification_report(y_test, preds, zero_division=0))

    elif args.csv_infer:
        csv_inference(csv_input=args.csv_input, csv_output=args.csv_output)


if __name__ == "__main__":
    main()
