"""Preprocessing for the Starbucks Next Best Offer prototype.

Reads the 3 raw JSON files from data/, cleans them, builds the target variable from the
event log and writes one modelling table: one row per (customer, received offer).

    python preprocessing.py   ->   data/processed.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT = DATA_DIR / "processed.csv"

CHANNELS = ["email", "mobile", "social", "web"]
CUSTOMER_FEATURES = ["age", "gender", "income", "income_missing", "membership_days"]
OFFER_FEATURES = ["offer_type", "difficulty", "reward", "duration_days"] + [f"ch_{c}" for c in CHANNELS]
FEATURES = CUSTOMER_FEATURES + OFFER_FEATURES
TARGET = "success"
CATEGORIES = {"gender": ["F", "M", "O", "U"], "offer_type": ["bogo", "discount", "informational"]}


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the model. Fixed categories so training and prediction match
    (LightGBM handles pandas categoricals and NaN natively)."""
    X = df[FEATURES].copy()
    for col, cats in CATEGORIES.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


def read_json(name: str) -> pd.DataFrame:
    return pd.read_json(DATA_DIR / name, orient="records", lines=True)


# --------------------------------------------------------------------------- #
# 1. Offers
# --------------------------------------------------------------------------- #
def load_portfolio() -> pd.DataFrame:
    df = read_json("portfolio.json").rename(columns={"id": "offer_id", "duration": "duration_days"})
    for c in CHANNELS:  # list column -> one binary flag per channel
        df[f"ch_{c}"] = df["channels"].apply(lambda x, c=c: int(c in x))
    return df.drop(columns="channels")


# --------------------------------------------------------------------------- #
# 2. Customers: missing values, noisy age, membership length
# --------------------------------------------------------------------------- #
def clean_profile(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Clean raw profile rows (also used for new customers at prediction time)."""
    df = df.copy()

    # age == 118 is a placeholder used when the customer gave no data -> missing
    df["age"] = df["age"].replace(118, np.nan).clip(18, 100)

    # missing gender -> own category; missing income -> keep NaN (LightGBM handles it) + flag
    df["gender"] = df["gender"].fillna("U")
    df["income_missing"] = df["income"].isna().astype(int)

    # became_member_on is an int like 20170815 -> membership length in days
    member_since = pd.to_datetime(df["became_member_on"].astype(str), format="%Y%m%d")
    df["membership_days"] = (reference_date - member_since).dt.days

    return df.drop(columns="became_member_on")


def membership_reference_date() -> pd.Timestamp:
    """Latest membership date in the training data, used as 'today' for membership_days."""
    raw = read_json("profile.json")["became_member_on"].astype(str)
    return pd.to_datetime(raw, format="%Y%m%d").max()


def load_profile() -> pd.DataFrame:
    df = read_json("profile.json").rename(columns={"id": "person"})
    return clean_profile(df, membership_reference_date())


# --------------------------------------------------------------------------- #
# 3. Event log -> events by type
# --------------------------------------------------------------------------- #
def load_transcript() -> pd.DataFrame:
    df = read_json("transcript.json")
    # noisy key naming: 'offer id' for received/viewed, 'offer_id' for completed
    df["offer_id"] = df["value"].apply(lambda v: v.get("offer id", v.get("offer_id")))
    df["amount"] = df["value"].apply(lambda v: v.get("amount"))
    return df.drop(columns="value")


# --------------------------------------------------------------------------- #
# 4. Target: was the offer viewed and then acted on within its validity window?
# --------------------------------------------------------------------------- #
def build_labels(events: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """One row per received offer with success = 1 if
    - the customer viewed it within the validity window, AND
    - after viewing (still within the window) they
        * completed it (bogo / discount), or
        * made any transaction (informational offers have no 'completed' event).
    Completing an offer without viewing it is NOT a success: the customer would
    have bought anyway, so the offer did not influence them."""
    rec = events[events["event"] == "offer received"][["person", "offer_id", "time"]]
    rec = rec.rename(columns={"time": "t_received"}).reset_index(drop=True)
    rec["row_id"] = rec.index
    rec = rec.merge(offers[["offer_id", "offer_type", "duration_days"]], on="offer_id")
    rec["t_end"] = rec["t_received"] + rec["duration_days"] * 24  # time is in hours

    def first_in_window(ev, start_col, keys):
        """First event of `ev` per received offer between start_col and t_end."""
        m = rec.merge(ev, on=keys)
        m = m[(m["time"] >= m[start_col]) & (m["time"] <= m["t_end"])]
        return m.groupby("row_id")["time"].min()

    viewed = events[events["event"] == "offer viewed"][["person", "offer_id", "time"]]
    rec["t_viewed"] = rec["row_id"].map(first_in_window(viewed, "t_received", ["person", "offer_id"]))

    completed = events[events["event"] == "offer completed"][["person", "offer_id", "time"]]
    t_completed = first_in_window(completed, "t_viewed", ["person", "offer_id"])

    transactions = events[events["event"] == "transaction"][["person", "time"]]
    t_transaction = first_in_window(transactions, "t_viewed", ["person"])

    is_info = rec["offer_type"] == "informational"
    acted = np.where(is_info, rec["row_id"].isin(t_transaction.index), rec["row_id"].isin(t_completed.index))
    rec[TARGET] = (rec["t_viewed"].notna() & acted).astype(int)

    return rec[["person", "offer_id", "t_received", TARGET]]


# --------------------------------------------------------------------------- #
# 5. Put it together
# --------------------------------------------------------------------------- #
def build_dataset() -> pd.DataFrame:
    offers, customers, events = load_portfolio(), load_profile(), load_transcript()
    labels = build_labels(events, offers)
    df = labels.merge(customers, on="person").merge(offers, on="offer_id")
    return df[["person", "offer_id", "t_received"] + FEATURES + [TARGET]]


if __name__ == "__main__":
    data = build_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data):,} rows to {OUTPUT}")
    print("\nSuccess rate by offer type:")
    print(data.groupby("offer_type")[TARGET].mean().round(3))
    print("\nMissing values:")
    print(data[FEATURES].isna().sum()[lambda s: s > 0])