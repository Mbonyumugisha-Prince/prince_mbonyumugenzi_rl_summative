from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def explore_dataset():
    files = {
        "Sales": "sales_train_validation.csv",
        "Prices": "sell_prices.csv",
        "Calendar": "calendar.csv",
    }

    for name, filename in files.items():
        file_path = DATA_DIR / filename

        print(f"\n{'=' * 50}")
        print(f"EXPLORING: {name.upper()}")
        print(f"{'=' * 50}")

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        # Read a small sample first
        df = pd.read_csv(file_path, nrows=5)

        print(f"File: {filename}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumn names:\n{df.columns.tolist()}")
        print(f"\nFirst five rows:\n{df.head()}")


if __name__ == "__main__":
    explore_dataset()
