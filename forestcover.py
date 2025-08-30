# forest_cover_type_classification.py
# End-to-end script with visuals, optional hyperparameter tuning,
# and automatic saving of figures as PNG files.

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Force interactive backend
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

SEED = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 20000  # for faster run, set None for full dataset
DO_TUNING = False    # set True for hyperparameter tuning

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

np.random.seed(SEED)


def load_data(sample_size=SAMPLE_SIZE, seed=SEED):
    print("Loading UCI Covertype dataset...")
    data = fetch_covtype(as_frame=True)
    X = data.data
    y = data.target

    if sample_size is not None and sample_size < len(X):
        print(f"Downsampling to {sample_size} rows...")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
        idx = next(sss.split(X, y))[0]
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    feature_names = X.columns.tolist()
    class_names = [f"Type {c}" for c in sorted(np.unique(y))]
    print(f"Train={len(X_train)}, Test={len(X_test)}, Features={len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names, class_names


def train_random_forest(X_train, y_train, do_tuning=DO_TUNING, seed=SEED):
    if not do_tuning:
        print("Training RandomForest (baseline)...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=seed
        )
        rf.fit(X_train, y_train)
        return rf

    print("Hyperparameter tuning RandomForest...")
    rf_base = RandomForestClassifier(n_jobs=-1, random_state=seed)
    param_dist = {
        "n_estimators": randint(200, 400),
        "max_depth": randint(10, 40),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }
    rf_search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    print("Best RF params:", rf_search.best_params_)
    return rf_search.best_estimator_


def train_xgboost(X_train, y_train, X_valid, y_valid, do_tuning=DO_TUNING, seed=SEED):
    if not HAS_XGB:
        print("XGBoost not installed. Skipping.")
        return None

    if not do_tuning:
        print("Training XGBoost (baseline)...")
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=20
        )
        return xgb_model

    print("Hyperparameter tuning XGBoost...")
    xgb_base = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": randint(200, 400),
        "learning_rate": uniform(0.05, 0.15),
        "max_depth": randint(6, 14),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    }

    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=3,
        random_state=seed,
        n_jobs=-1,
    )
    xgb_search.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=20
    )
    print("Best XGB params:", xgb_search.best_params_)
    return xgb_search.best_estimator_


def evaluate_and_plot(model, X_test, y_test, class_names, feature_names, model_label):
    print(f"Evaluating {model_label}...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, normalize='true', display_labels=class_names, xticks_rotation=45
    )
    plt.title(f"{model_label} — Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_label.replace(' ', '_')}_confusion.png", dpi=300)
    plt.show(block=True)

    # Feature importances
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is not None:
        top_k = min(15, len(importances))
        idx = np.argsort(importances)[-top_k:][::-1]
        plt.barh(range(top_k), importances[idx][::-1])
        plt.yticks(range(top_k), [feature_names[i] for i in idx][::-1])
        plt.xlabel("Importance")
        plt.title(f"{model_label} — Top {top_k} Features")
        plt.tight_layout()
        plt.savefig(f"{model_label.replace(' ', '_')}_features.png", dpi=300)
        plt.show(block=True)

    return {"accuracy": acc, "macro_f1": f1m}


def main():
    X_train, X_test, y_train, y_test, feature_names, class_names = load_data()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    rf_model = train_random_forest(X_tr, y_tr, do_tuning=DO_TUNING)
    rf_scores = evaluate_and_plot(rf_model, X_test, y_test, class_names, feature_names, "Random Forest")

    if HAS_XGB:
        xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, do_tuning=DO_TUNING)
        xgb_scores = evaluate_and_plot(xgb_model, X_test, y_test, class_names, feature_names, "XGBoost")
    else:
        xgb_scores = None

    print("Comparison:")
    print(f"Random Forest -> Acc: {rf_scores['accuracy']:.4f}, Macro-F1: {rf_scores['macro_f1']:.4f}")
    if xgb_scores is not None:
        print(f"XGBoost -> Acc: {xgb_scores['accuracy']:.4f}, Macro-F1: {xgb_scores['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
