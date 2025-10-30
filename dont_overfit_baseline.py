#!/usr/bin/env python3
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

SEED = 42
np.random.seed(SEED)

# ---------- Utility ----------
def find_cols(df):
    cols = [c.lower() for c in df.columns]
    id_col = next((df.columns[i] for i, c in enumerate(cols) if c in ["id","id_code","row_id"]), None)
    tgt_col = next((df.columns[i] for i, c in enumerate(cols) if c in ["target","label","y"]), None)
    return id_col, tgt_col

def split_xy(train, target_col):
    y = train[target_col].values
    X = train.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y

# ---------- Model builder ----------
def make_lr_pipeline(n_features=25, n_pca=8, C=0.0005, seed=SEED):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=n_features)),
        ("pca", PCA(n_components=n_pca, random_state=seed)),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            C=C,
            max_iter=10000,
            random_state=seed
        )),
    ])
    return pipe


# ---------- Visualization Utilities ----------
def plot_cv_auc_distribution(cv_scores, out_path="cv_auc_distribution.png"):
    plt.figure(figsize=(6,4))
    sns.histplot(cv_scores, bins=8, kde=True, color="skyblue")
    plt.title("Distribution of Cross-Validation AUC across Monte Carlo Seeds")
    plt.xlabel("ROC-AUC")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def plot_pca_variance(X, out_path="pca_explained_variance.png"):
    pca = PCA().fit(X)
    plt.figure(figsize=(7,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def plot_feature_correlation(X, out_path="correlation_heatmap.png"):
    plt.figure(figsize=(9,7))
    sns.heatmap(X.sample(30, axis=1).corr().abs(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (Random 30 Features)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

def plot_learning_curve(pipe, X, y, out_path="learning_curve.png"):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=10, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training AUC")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="s", label="Validation AUC")
    plt.xlabel("Training Samples")
    plt.ylabel("ROC-AUC")
    plt.title("Learning Curve (Logistic Regression)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# ---------- Bagging + CV ----------
def monte_carlo_ensemble(X, y, X_test, seeds=10):
    preds = np.zeros((len(X_test), seeds))
    cv_scores = []

    for i, seed in enumerate(range(SEED, SEED + seeds)):
        np.random.seed(seed)
        X_noisy = X + np.random.normal(0, 0.01, X.shape)  # data noise
        pipe = make_lr_pipeline(seed=seed)

        cv = StratifiedKFold(n_splits=30, shuffle=True, random_state=seed)
        cv_score = cross_val_score(pipe, X_noisy, y, cv=cv, scoring="roc_auc").mean()
        cv_scores.append(cv_score)
        print(f"[seed {seed}] CV AUC={cv_score:.4f}")

        pipe.fit(X_noisy, y)
        preds[:, i] = pipe.predict_proba(X_test)[:, 1]

    # --- Visualization: CV AUC distribution ---
    plot_cv_auc_distribution(cv_scores)

    print(f"\nAverage CV AUC across seeds: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    return preds.mean(axis=1), np.mean(cv_scores), np.std(cv_scores)


# ---------- Main ----------
def main(train_path, test_path, out_path="submission.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    id_col, tgt_col = find_cols(train)
    X, y = split_xy(train, tgt_col)
    X_test = test[X.columns].copy()

    # clip to avoid outliers
    X = X.clip(-3, 3)
    X_test = X_test.clip(-3, 3)

    # --- Train ensemble ---
    preds, mean_auc, std_auc = monte_carlo_ensemble(X, y, X_test, seeds=10)

    # --- Extra visualizations ---
    plot_pca_variance(X)
    plot_feature_correlation(X)
    pipe = make_lr_pipeline()
    plot_learning_curve(pipe, X, y)

    # --- Save submission ---
    ids = test[id_col] if id_col in test.columns else np.arange(len(test))
    sub = pd.DataFrame({id_col or "id": ids, "target": preds})
    sub.to_csv(out_path, index=False)
    print(f"[ok] Saved submission to {out_path}")
    print(f"[report] Mean CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")

if __name__ == "__main__":
    train_csv = os.environ.get("TRAIN_CSV", "../mnt/data/train.csv")
    test_csv  = os.environ.get("TEST_CSV", "../mnt/data/test.csv")
    out_csv   = os.environ.get("SUBMISSION_CSV", "../dont_overfit/submission.csv")
    sys.exit(main(train_csv, test_csv, out_csv))
