# Next Best Action: Starbucks Offers

Recommends the promotional offer each Starbucks rewards customer is most likely to act on.

A LightGBM classifier estimates `P(success | customer, offer)`. For each customer, the model scores all 10 offers and recommends the one with the highest probability. If even the best offer falls below a threshold, it recommends `no_offer`.

## Project structure

```
nba-starbucks/
├── data/
│   ├── raw/                  # portfolio.json, profile.json, transcript.json
│   └── processed/            # processed.csv, created by preprocessing
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis of the raw datasets
├── output/
│   ├── metrics.json          # classification and next-best-offer metrics
│   ├── model.joblib          # trained model
│   ├── feature_importance.png
│   ├── roc_auc_curve.png
│   └── confusion_matrix.png
├── src/
│   ├── preprocessing.py      # cleaning, target variable, customer history features
│   ├── train.py              # training, evaluation, plots
│   └── predict.py            # next best offer for new customers
├── run.py                    # runs the pipeline
└── requirements.txt
```

## Data

The project uses the Starbucks rewards app dataset, which simulates customer behaviour. It was downloaded from Kaggle: https://www.kaggle.com/datasets/mexwell/starbucks-offers-advertisement-data

The three raw files are included in `data/raw/`:

| File | Content |
|---|---|
| `portfolio.json` | The 10 offers: type (BOGO, discount, informational), difficulty, reward, duration and channels |
| `profile.json` | Customer demographics: age, gender, income, membership date |
| `transcript.json` | Event log: offer received, offer viewed, offer completed and transactions |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    
pip install -r requirements.txt
```

## Usage

```bash
python run.py                          # preprocessing, training and prediction demo
python run.py --step preprocess        # only builds data/processed/processed.csv
python run.py --step train             # only trains and evaluates the model
python run.py --step train predict     # several steps; they always run in pipeline order
```

The processed dataset and the trained model are already included, so `python run.py --step predict` works straight away. Running `preprocess` or `train` overwrites these files and the contents of `output/`.

The scripts in `src/` can also be run on their own, for example `python src/train.py`.

## Method

**Target variable.** One row is one received offer. The offer counts as a success when the customer viewed it within its validity window and then:
- completed it (BOGO and discount offers), or
- made any transaction (informational offers, which have no completion event).

**Features.**
- Customer: age, gender, income, a flag for missing income, and membership length. An age of 118 means the age is missing. A missing gender gets its own category, `U`.
- Customer history: only events before the offer arrived, so the model can't see the future. This covers past transaction count, total, mean and max spend, days since the last transaction, offers received before, and the share of past offers the customer viewed.
- Offer: type, difficulty, reward, duration and one-hot channels.
- Interactions: typical spend compared with the offer's difficulty, reward per unit of difficulty, and required spend per day.

**Evaluation.** The train/test split is grouped by customer, so no customer appears in both sets. Offers in the data were sent roughly at random. So the success rate on test rows where the customer happened to get the model's pick estimates how the model's recommendations would perform.

## Results

From `output/metrics.json`:

| Classification | Value |
|---|---|
| ROC AUC | 0.804 |
| Accuracy | 0.738 |
| Precision | 0.686 |
| Recall | 0.619 |

| Offer strategy | Success rate |
|---|---|
| Offers as sent in the data (roughly random) | 39.4% |
| Best single offer sent to everyone | 64.7% |
| **Model's recommended offer** | **67.7%** |

The model's pick is evaluated on 1,485 test rows where the offer sent matched the recommendation.
