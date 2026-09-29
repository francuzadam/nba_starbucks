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
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from preprocessing import CUSTOMER_FEATURES, DATA_DIR, OFFER_FEATURES, TARGET, load_portfolio, model_matrix

ROOT = DATA_DIR.parent
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


def main():
    df = pd.read_csv(DATA_DIR / "processed.csv")
    offers = load_portfolio()

    # split by customer, so the same person never appears in both train and test
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(split.split(df, groups=df["person"]))
    train = df.iloc[tr_idx]
    test = df.iloc[te_idx].reset_index(drop=True)

    model = LGBMClassifier(**PARAMS).fit(model_matrix(train), train[TARGET])

    # 1. Classification metrics
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

    # 2. Next Best Offer evaluation vs. simple baselines
    picks = score_all_offers(model, test, offers).idxmax(axis=1) # the offer with the highest probability
    best_single_offer = train.groupby("offer_id")[TARGET].mean().idxmax()
    nba = evaluate_nba(test, picks.to_numpy(), best_single_offer)

    print("\nRecommended offer distribution:")
    print(picks.value_counts(normalize=True).round(3))

    metrics = {"classification": classification, "next_best_offer": nba}
    print(json.dumps(metrics, indent=2, default=float))

    importance = pd.Series(model.feature_importances_, index=model.feature_name_)
    print("\nFeature importance:")
    print(importance.sort_values(ascending=False))

    joblib.dump(model, MODEL_PATH)
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()