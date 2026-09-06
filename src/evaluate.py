"""
evaluate.py
===========

Task 3 - Evaluation & Performance Analysis
-------------------------------------------
Evaluates the 6 already-trained spam-detection models from
`Optimized_Spam_Detection_System` on the SAME held-out test set that was
used in `Models.ipynb`. No model is retrained, no vectorizer is refit, and
no new train/test split is created.

Models evaluated (vectorizer + classifier):
    1. CountVectorizer + Multinomial Naive Bayes
    2. CountVectorizer + Logistic Regression
    3. CountVectorizer + LinearSVC
    4. TF-IDF + Multinomial Naive Bayes
    5. TF-IDF + Logistic Regression
    6. TF-IDF + LinearSVC

Positive class definition
--------------------------
Spam = 1 (positive class), Ham = 0 (negative class).
This matches the label mapping used in `Models.ipynb`
(`y = df["label"].map({"ham": 0, "spam": 1})`), so Precision/Recall/F1
below are reported with respect to detecting SPAM correctly.

How the test set is reproduced
-------------------------------
`Models.ipynb` does not persist X_test/y_test to disk. To use the exact
same, untouched test set without creating a new split, this script
reloads `data/cleaned_spam.csv` and repeats the identical
`train_test_split(..., test_size=0.2, random_state=42, stratify=y)` call
used in `Models.ipynb`. Because scikit-learn's split is fully
deterministic given the same data, same order, and same random_state,
this reproduces the identical test indices/rows - it does not create a
new split. This was verified against the original notebook (see
README/analysis notes).

Outputs written to `../outputs/`:
    - model_comparison_results.csv   (metrics table for all 6 models)
    - confusion_matrices.png         (2x3 grid of confusion matrices)
    - metrics_comparison.png         (grouped bar chart of the 4 metrics)
    - misclassified_examples.csv     (sample of FP/FN messages per model)
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# 0. Paths & constants
# --------------------------------------------------------------------------
DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
POSITIVE_LABEL = 1  # spam is the positive class

# (comparison-table name, feature family, classifier, vectorizer file, model file)
MODEL_SPECS = [
    ("CountVectorizer + Multinomial NB", "CountVectorizer", "MultinomialNB",
     "count_vectorizer.pkl", "countvectorizer_multinomialnb.pkl"),
    ("CountVectorizer + Logistic Regression", "CountVectorizer", "LogisticRegression",
     "count_vectorizer.pkl", "countvectorizer_logisticregression.pkl"),
    ("CountVectorizer + LinearSVC", "CountVectorizer", "LinearSVC",
     "count_vectorizer.pkl", "countvectorizer_linearsvc.pkl"),
    ("TF-IDF + Multinomial NB", "TF-IDF", "MultinomialNB",
     "tfidf_vectorizer.pkl", "tf_idf_multinomialnb.pkl"),
    ("TF-IDF + Logistic Regression", "TF-IDF", "LogisticRegression",
     "tfidf_vectorizer.pkl", "tf_idf_logisticregression.pkl"),
    ("TF-IDF + LinearSVC", "TF-IDF", "LinearSVC",
     "tfidf_vectorizer.pkl", "tf_idf_linearsvc.pkl"),
]


# --------------------------------------------------------------------------
# 1. Reconstruct the exact test set used in Models.ipynb
# --------------------------------------------------------------------------
def load_test_set():
    """Reproduce the identical held-out test set from Models.ipynb.

    This mirrors Models.ipynb exactly: same source file, same dropna,
    same label mapping, same split call/order/random_state. No new split
    is created - this deterministically reconstructs the original one
    since X_test/y_test were not saved to disk by Models.ipynb.
    """
    df = pd.read_csv(DATA_DIR / "cleaned_spam.csv")
    df = df.dropna(subset=["clean_message", "label"]).reset_index(drop=True)

    X = df["clean_message"].astype(str)
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Load vectorizers and models (no fitting / training happens here)
# --------------------------------------------------------------------------
def load_vectorizers():
    return {
        "CountVectorizer": joblib.load(MODELS_DIR / "count_vectorizer.pkl"),
        "TF-IDF": joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl"),
    }


def load_models():
    models = {}
    for display_name, _feat, _clf, _vec_file, model_file in MODEL_SPECS:
        models[display_name] = joblib.load(MODELS_DIR / model_file)
    return models


# --------------------------------------------------------------------------
# 3. Evaluate every model on the SAME test set
# --------------------------------------------------------------------------
def evaluate_all_models(X_test, y_test, vectorizers, models):
    """Transform-only (never fit) the test text with the matching vectorizer,
    predict with the matching pre-trained model, and score."""
    results = []
    predictions = {}          # display_name -> y_pred
    confusion_matrices = {}   # display_name -> cm array

    for display_name, feature_name, clf_name, _vec_file, _model_file in MODEL_SPECS:
        vectorizer = vectorizers[feature_name]
        model = models[display_name]

        X_test_features = vectorizer.transform(X_test)  # transform only, never fit
        y_pred = model.predict(X_test_features)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        rec = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": display_name,
            "Vectorizer": feature_name,
            "Classifier": clf_name,
            "Accuracy": round(acc, 4),
            "Precision (Spam)": round(prec, 4),
            "Recall (Spam)": round(rec, 4),
            "F1-score (Spam)": round(f1, 4),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
        })

        predictions[display_name] = y_pred
        confusion_matrices[display_name] = cm

    results_df = pd.DataFrame(results).sort_values(
        "F1-score (Spam)", ascending=False
    ).reset_index(drop=True)
    return results_df, predictions, confusion_matrices


# --------------------------------------------------------------------------
# 4. Error analysis: false positives / false negatives / misclassified text
# --------------------------------------------------------------------------
def collect_misclassified(X_test, y_test, predictions, max_examples=10):
    """Return a DataFrame of misclassified messages per model, tagged as
    False Positive (ham -> predicted spam) or False Negative
    (spam -> predicted ham)."""
    rows = []
    for display_name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred)
        y_true = y_test.to_numpy()

        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

        for i in fp_idx[:max_examples]:
            rows.append({
                "Model": display_name,
                "Error Type": "False Positive (ham -> spam)",
                "True Label": "ham",
                "Predicted": "spam",
                "Message": X_test.iloc[i],
            })
        for i in fn_idx[:max_examples]:
            rows.append({
                "Model": display_name,
                "Error Type": "False Negative (spam -> ham)",
                "True Label": "spam",
                "Predicted": "ham",
                "Message": X_test.iloc[i],
            })

    return pd.DataFrame(rows)


def print_error_summary(results_df):
    print("\nFalse Positive / False Negative counts per model")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(
            f"{row['Model']:<38} FP={row['False Positives']:<4} "
            f"FN={row['False Negatives']:<4}"
        )


# --------------------------------------------------------------------------
# 5. Visualizations
# --------------------------------------------------------------------------
def plot_confusion_matrices(confusion_matrices, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for ax, (display_name, cm) in zip(axes, confusion_matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(display_name, fontsize=10)

    fig.suptitle("Confusion Matrices - All 6 Models (Test Set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(results_df, save_path):
    metrics = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-score (Spam)"]
    plot_df = results_df.set_index("Model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(14, 7), width=0.8)
    ax.set_ylabel("Score")
    y_min = max(0.0, plot_df.to_numpy().min() - 0.05)
    ax.set_ylim(y_min, 1.01)
    ax.set_title("Model Comparison: Accuracy, Precision, Recall, F1 (Spam = positive class)")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_f1_ranking(results_df, save_path):
    plt.figure(figsize=(10, 6))
    order = results_df.sort_values("F1-score (Spam)")
    sns.barplot(data=order, y="Model", x="F1-score (Spam)", hue="Model",
                palette="viridis", legend=False)
    plt.xlim(0.8, 1.0)
    plt.title("Models Ranked by F1-score (Spam as positive class)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# 6. Best model selection
# --------------------------------------------------------------------------
def select_best_model(results_df):
    """Pick the best model using F1-score as the primary criterion (not
    accuracy), since the dataset is imbalanced (~87% ham / ~13% spam) and
    F1 balances precision and recall for the spam class. Recall is used
    as a tiebreaker because missing spam (false negative) is generally
    worse than an occasional false positive in this use case discussion."""
    best_row = results_df.sort_values(
        by=["F1-score (Spam)", "Recall (Spam)"], ascending=False
    ).iloc[0]
    return best_row


def print_comparison_table(results_df):
    display_cols = ["Model", "Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-score (Spam)"]
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE (test set, spam = positive class)")
    print("=" * 90)
    print(results_df[display_cols].to_string(index=False))


def print_best_model_explanation(best_row, results_df):
    print("\n" + "=" * 90)
    print("BEST MODEL")
    print("=" * 90)
    print(f"Selected: {best_row['Model']}")
    print(f"  Accuracy : {best_row['Accuracy']}")
    print(f"  Precision: {best_row['Precision (Spam)']}")
    print(f"  Recall   : {best_row['Recall (Spam)']}")
    print(f"  F1-score : {best_row['F1-score (Spam)']}")
    print(f"  False Positives: {best_row['False Positives']}  "
          f"False Negatives: {best_row['False Negatives']}")

    acc_leader = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    if acc_leader["Model"] != best_row["Model"]:
        print(
            f"\nNote: '{acc_leader['Model']}' has the highest raw Accuracy "
            f"({acc_leader['Accuracy']}), but on this imbalanced dataset "
            "(~87% ham / ~13% spam) accuracy alone is misleading - a model "
            "that predicts mostly 'ham' can still score high on accuracy "
            "while missing spam messages. F1-score (the harmonic mean of "
            "precision and recall for the spam class) and recall give a "
            "fairer picture of how well each model actually catches spam "
            "without excessively misflagging real messages, which is why "
            f"'{best_row['Model']}' is selected as the best overall model."
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading reconstructed test set (same split as Models.ipynb)...")
    X_test, y_test = load_test_set()
    print(f"Test set size: {len(X_test)} messages "
          f"({(y_test == 1).sum()} spam / {(y_test == 0).sum()} ham)")

    print("\nLoading pre-trained vectorizers and models (no fitting)...")
    vectorizers = load_vectorizers()
    models = load_models()

    print("Evaluating all 6 models on the identical test set...")
    results_df, predictions, confusion_matrices = evaluate_all_models(
        X_test, y_test, vectorizers, models
    )

    print_comparison_table(results_df)
    print_error_summary(results_df)

    misclassified_df = collect_misclassified(X_test, y_test, predictions)

    best_row = select_best_model(results_df)
    print_best_model_explanation(best_row, results_df)

    # Save outputs
    results_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    misclassified_df.to_csv(OUTPUT_DIR / "misclassified_examples.csv", index=False)
    plot_confusion_matrices(confusion_matrices, OUTPUT_DIR / "confusion_matrices.png")
    plot_metrics_comparison(results_df, OUTPUT_DIR / "metrics_comparison.png")
    plot_f1_ranking(results_df, OUTPUT_DIR / "f1_ranking.png")

    print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
