"""
Preprocessing for the Next Best Action based on Starbucks dataset .
Reads the 3 raw JSON files from data folder, cleans them, builds the target variable from the event log and writes one modelling table: one row per customer, received offer.
Input: python preprocessing.py 
Output: data/processed.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT = DATA_DIR / "processed.csv"

CHANNELS = ["email", "mobile", "social", "web"]
HISTORY_FEATURES = ["past_tx_count", "past_spend_total", "past_spend_mean", "past_spend_max", "days_since_last_tx", "past_offers_received", "past_views", "past_view_rate"]
CUSTOMER_FEATURES = ["age", "gender", "income", "income_missing", "membership_days"] + HISTORY_FEATURES
OFFER_FEATURES = ["offer_type", "difficulty", "reward", "duration_days"] + [f"ch_{c}" for c in CHANNELS]
FEATURES = CUSTOMER_FEATURES + OFFER_FEATURES
TARGET = "success"
CATEGORIES = {"gender": ["F", "M", "O", "U"], "offer_type": ["bogo", "discount", "informational"]}


# Compare the spending habit of the customer to the conditions of the offer
def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    difficulty = X["difficulty"].replace(0, np.nan)  # informational offers have 0 difficulty, division would fail
    X["spend_vs_diff"] = X["past_spend_mean"] / difficulty  # can the customer reach the spending limit
    X["reward_ratio"] = X["reward"] / difficulty  # how generous is the offer
    X["diff_per_day"] = X["difficulty"] / X["duration_days"]  # how much has to be spent per day
    return X

def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for the model. Fixed categories so training and prediction match (LightGBM handles pandas categoricals and NaN natively)."""
    X = df[FEATURES].copy()
    for col, cats in CATEGORIES.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return add_interactions(X)


def read_json(name: str) -> pd.DataFrame:
    """Import JSON files from DATA_DIR into a DataFrame."""
    return pd.read_json(DATA_DIR / name, orient="records", lines=True)


# Import porfolio JSON applying One-Hot Encoding
def load_portfolio() -> pd.DataFrame:
    df = read_json("portfolio.json").rename(columns={"id": "offer_id", "duration": "duration_days"})
    for c in CHANNELS:
        df[f"ch_{c}"] = df["channels"].apply(lambda x: int(c in x))
    return df.drop(columns="channels")


# Handle missing, noisy values, and outliers
def clean_profile(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Clean raw profile rows (also used for new customers at prediction time)."""
    df = df.copy()

    # In case of age 118 means that it is missing, replace with NaN and filter the customer between 18 and 100 because these could be relevant
    df["age"] = df["age"].replace(118, np.nan).clip(18, 100)

    # For gender I use a unique category for missing values called U, the missing income could be relevant information so I create a new feature for it and keep tha NaN
    df["gender"] = df["gender"].fillna("U")
    df["income_missing"] = df["income"].isna().astype(int)

    # Membership length is essential for ML model and calculate from registration date
    member_since = pd.to_datetime(df["became_member_on"].astype(str), format="%Y%m%d")
    df["membership_days"] = (reference_date - member_since).dt.days

    return df.drop(columns="became_member_on")


# Membership reference date is the maximum of became_member_on feature
def membership_reference_date() -> pd.Timestamp:
    """Latest membership date in the training data, used as 'today' for membership_days."""
    raw = read_json("profile.json")["became_member_on"].astype(str)
    return pd.to_datetime(raw, format="%Y%m%d").max()


# Load customer data using clean_profile and membership_reference_date funcions
def load_profile() -> pd.DataFrame:
    df = read_json("profile.json").rename(columns={"id": "person"})
    return clean_profile(df, membership_reference_date())


# Import Event logs
def load_transcript() -> pd.DataFrame:
    df = read_json("transcript.json")

    # value column is a dictionary, I have to collect the offer_id and amount features
    # noisy key naming: 'offer id' for received/viewed, 'offer_id' for completed
    df["offer_id"] = df["value"].apply(lambda v: v.get("offer id", v.get("offer_id")))
    df["amount"] = df["value"].apply(lambda v: v.get("amount"))

    return df.drop(columns="value")


# Create target variable
def build_labels(events: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """One row per received offer.
    Success = 1 if
    - the customer viewed it within the validity window, AND
    - after viewing they
        * completed it (bogo / discount), or
        * made any transaction (informational offers have no 'completed' event).
    """
    rec = events[events["event"] == "offer received"][["person", "offer_id", "time"]]
    rec = rec.rename(columns={"time": "t_received"}).reset_index(drop=True)
    rec["row_id"] = rec.index # Create unique id because a customer can get the same offer multiple times
    rec = rec.merge(offers[["offer_id", "offer_type", "duration_days"]], on="offer_id")
    rec["t_end"] = rec["t_received"] + rec["duration_days"] * 24  # time is in hours

    def first_in_window(ev, start_col, keys):
        """First event of `ev` per received offer between start_col and t_end."""
        m = rec.merge(ev, on=keys)
        m = m[(m["time"] >= m[start_col]) & (m["time"] <= m["t_end"])]
        return m.groupby("row_id")["time"].min()

    viewed = events[events["event"] == "offer viewed"][["person", "offer_id", "time"]]
    rec["t_viewed"] = rec["row_id"].map(first_in_window(viewed, "t_received", ["person", "offer_id"])) # first viewed log within the validity window, else NaN

    completed = events[events["event"] == "offer completed"][["person", "offer_id", "time"]]
    t_completed = first_in_window(completed, "t_viewed", ["person", "offer_id"]) # is completed within the validity window

    transactions = events[events["event"] == "transaction"][["person", "time"]]
    t_transaction = first_in_window(transactions, "t_viewed", ["person"]) # is any transaction within the validity window

    is_info = rec["offer_type"] == "informational"
    acted = np.where(is_info, rec["row_id"].isin(t_transaction.index), rec["row_id"].isin(t_completed.index)) # if the offer is informational I check the t_transaction, else t_completed
    rec[TARGET] = (rec["t_viewed"].notna() & acted).astype(int) # 1 if t_viewed is not null and acted is True

    return rec[["person", "offer_id", "t_received", TARGET]]


# What do I know about the customer at the moment when the offer arrives
def add_history(labels: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rec = labels.reset_index(drop=True)
    rec["row_id"] = rec.index
    keys = rec[["row_id", "person", "t_received"]]

    # Past transactions. Only the events before t_received, otherwise it would be leakage
    tx = events[events["event"] == "transaction"][["person", "time", "amount"]]
    past_tx = keys.merge(tx, on="person")
    past_tx = past_tx[past_tx["time"] < past_tx["t_received"]]
    g = past_tx.groupby("row_id")

    rec["past_tx_count"] = rec["row_id"].map(g.size()).fillna(0) # amount of previous transactions, 0 if the customer never bought before
    rec["past_spend_total"] = rec["row_id"].map(g["amount"].sum()).fillna(0) # total amount spent in previous transactions, 0 if the customer never bought before
    rec["past_spend_mean"] = rec["row_id"].map(g["amount"].mean()) # Avarage amount spent in previous transactions, NaN if the customer never bought before
    rec["past_spend_max"] = rec["row_id"].map(g["amount"].max()) # Maximum amount spent in previous transactions, NaN if the customer never bought before

    last_tx = rec["row_id"].map(g["time"].max())
    rec["days_since_last_tx"] = (rec["t_received"] - last_tx) / 24 # time is in hours

    # How many offers the customer opened before
    viewed = events[events["event"] == "offer viewed"][["person", "time"]]
    past_viewed = keys.merge(viewed, on="person")
    past_viewed = past_viewed[past_viewed["time"] < past_viewed["t_received"]]
    rec["past_views"] = rec["row_id"].map(past_viewed.groupby("row_id").size()).fillna(0)

    # How many offers the customer got before, and what part of them was opened
    rec = rec.sort_values(["person", "t_received"])
    rec["past_offers_received"] = rec.groupby("person").cumcount()
    rec["past_view_rate"] = rec["past_views"] / rec["past_offers_received"].replace(0, np.nan)

    return rec.drop(columns="row_id")


# Create final dataset
def build_dataset() -> pd.DataFrame:
    offers, customers, events = load_portfolio(), load_profile(), load_transcript() # Load JSON data
    labels = build_labels(events, offers) # Create target variable based on event logs and offers
    labels = add_history(labels, events) # Add the customer history features
    df = labels.merge(customers, on="person").merge(offers, on="offer_id") # Create final df
    return df[["person", "offer_id", "t_received"] + FEATURES + [TARGET]]


if __name__ == "__main__":
    data = build_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data):,} rows to {OUTPUT}")

    print("\n-----")
    print("Success rate by offer type:")
    print(data.groupby('offer_type')[TARGET].mean().round(3))

    print("\n-----")
    missing = data[FEATURES].isna().sum()
    print("Missing values:")
    print(missing[missing > 0])