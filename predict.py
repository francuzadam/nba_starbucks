"""Next best offer for new customers:  python predict.py

Input: raw customer rows in the profile.json format
       (id, age, gender, income, became_member_on).
Output: the recommended offer and its predicted success probability per customer.
"""
import joblib
import pandas as pd

from preprocessing import clean_profile, load_portfolio, membership_reference_date
from train import MODEL_PATH, score_all_offers


def predict_next_best_action(customers: pd.DataFrame, min_probability: float = 0.0) -> pd.DataFrame:
    """Score all offers for each customer and return the best one.
    If even the best offer is below `min_probability`, recommend 'no_offer'."""
    model = joblib.load(MODEL_PATH)
    offers = load_portfolio().set_index("offer_id")

    X = clean_profile(customers, membership_reference_date())
    probs = score_all_offers(model, X, offers.reset_index())

    best_offer = probs.idxmax(axis=1)
    best_prob = probs.max(axis=1)
    result = pd.DataFrame({
        "customer_id": customers["id"].to_numpy(),
        "recommended_offer": best_offer.where(best_prob >= min_probability, "no_offer").to_numpy(),
        "success_probability": best_prob.round(3).to_numpy(),
    })
    details = offers[["offer_type", "difficulty", "reward", "duration_days"]]
    return result.join(details, on="recommended_offer")


if __name__ == "__main__":
    new_customers = pd.DataFrame([
        {"id": "new_1", "age": 34, "gender": "F", "income": 85000.0, "became_member_on": 20170301},
        {"id": "new_2", "age": 62, "gender": "M", "income": 45000.0, "became_member_on": 20180601},
        {"id": "new_3", "age": 118, "gender": None, "income": None, "became_member_on": 20160115},
    ])
    print(predict_next_best_action(new_customers, min_probability=0.3).to_string(index=False))