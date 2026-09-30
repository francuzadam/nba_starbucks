"""
Entry point for the whole pipeline.

Usage:
    python run.py                     # preprocessing + training + prediction demo
    python run.py --step train        # only training
    python run.py --step train predict
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import preprocessing
import train
import predict

STEPS = ["preprocess", "train", "predict"]
MAINS = {"preprocess": preprocessing.main, "train": train.main, "predict": predict.main}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", nargs="+", choices=STEPS + ["all"], default=["all"])
    args = parser.parse_args()
    steps = STEPS if "all" in args.step else args.step

    for step in STEPS:              # fixed order, no matter how the user listed them
        if step in steps:
            print("\n" + "=" * 50)
            print(step.upper())
            print("=" * 50)
            MAINS[step]()


if __name__ == "__main__":
    main()