"""
Train and evaluate the offer-success model.
LightGBM claasifier is to estimate P(success | customer, offer).
The next best offer for a customer is the offer with the highest predicted probability of success.
For evaluation, the model's performance is compared against two baseline strategies:
- the historical/random offer assignment success rate,
- the best single-offer strategy, where the same offer is assigned to every customer.

The resulting uplift shows the added value of personalized offer recommendations compared to simpler marketing approaches.
Input: processed.csv
Output: Classification metrics, Feature Importance, NBA metrics
"""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Use a non-interactive backend for matplotlib to avoid issues in headless environments
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GroupShuffleSplit
from preprocessing import CUSTOMER_FEATURES, DATA_DIR, OFFER_FEATURES, TARGET, load_portfolio, model_matrix


ROOT = DATA_DIR.parent
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "model.joblib"
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)


def score_all_offers(model, customers: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Calculate the probability of every offer for every customer."""
    rows = customers[CUSTOMER_FEATURES].reset_index(drop=True).rename_axis("row").reset_index()
    grid = rows.merge(offers[["offer_id"] + OFFER_FEATURES], how="cross")
    grid["p"] = model.predict_proba(model_matrix(grid))[:, 1]
    return grid.pivot(index="row", columns="offer_id", values="p")


def evaluate_nba(test, recommended, best_single_offer) -> dict:
    """Offers were assigned roughly at random, so the observed success rate of customers
    who happened to receive the model's pick estimates how the model's policy would do."""
    got = test["offer_id"].to_numpy() # real marketing offer
    y = test[TARGET].to_numpy() # target variable
    match = got == recommended # do the recommended and real offer march
    return {
        "success_rate_random_offer": float(y.mean()), # the rate of previous and random offers
        "success_rate_best_single_offer": float(y[got == best_single_offer].mean()), # if the best solution is offered for everyone
        "success_rate_model_pick": float(y[match].mean()), # rate of nba model
        "rows_where_model_pick_was_sent": int(match.sum()), # the predicted and the real offer is the same
    }

def feature_importance(model) -> None:
    """Print the feature importance of the trained model."""
    importance = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(importance.index, importance.to_numpy())
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlim(0, importance.max() * 1.12)
    ax.set_xlabel("Number of splits")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return None

def roc_auc_curve(model, test) -> None:
    """Plot the ROC AUC curve of the trained model."""

    y_true = test[TARGET]
    y_score = model.predict_proba(model_matrix(test))[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_curve.png", dpi=150)
    plt.close(fig)

def confusion_matrix(model, test) -> None:
    """Plot the confusion matrix of the trained model."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_true = test[TARGET]
    y_pred = model.predict(model_matrix(test))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    df = pd.read_csv(DATA_DIR / "processed.csv")
    offers = load_portfolio()

    #Split by customer, so the same person never appears in both train and test
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(split.split(df, groups=df["person"]))
    train = df.iloc[tr_idx]
    test = df.iloc[te_idx].reset_index(drop=True)

    #Build ML model
    model = LGBMClassifier(**PARAMS).fit(model_matrix(train), train[TARGET])

    #Classification metrics
    p = model.predict_proba(model_matrix(test))[:, 1]
    pred = (p >= 0.5).astype(int)
    y = test[TARGET]
    classification = {
        "roc_auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "accuracy_baseline_majority_class": max(y.mean(), 1 - y.mean()),
    }

    #Next Best Offer evaluation vs. simple baselines
    picks = score_all_offers(model, test, offers).idxmax(axis=1) # the offer with the highest probability
    best_single_offer = train.groupby("offer_id")[TARGET].mean().idxmax()
    nba = evaluate_nba(test, picks.to_numpy(), best_single_offer)

    print("\nRecommended offer distribution:")
    print(picks.value_counts(normalize=True).round(3))

    metrics = {"classification": classification, "next_best_offer": nba}
    print(json.dumps(metrics, indent=2, default=float))

    feature_importance(model)
    roc_auc_curve(model, test)
    confusion_matrix(model, test)
    joblib.dump(model, MODEL_PATH)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()