from pathlib import Path
import pandas as pd 

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

TOP_N = 10

def select_products():
    df = pd.read_csv(
        PROCESSED_DIR / "inventory_data.csv",
        parse_dates= ["date"],
    )

    stats = df.groupby("item_id").agg(
        days = ("sales", "size"),
        total_units=("sales", "sum"),
        zero_days_pct=("sales", lambda s: (s == 0).mean() * 100),
    )

    full_days = df["date"].nunique()
    stats = stats[stats["days"] >= 0.9 * full_days]

    top_products = (
        stats.sort_values("total_units", ascending=False)
        .head(TOP_N)
        .index
        .tolist()
    )

    print(f"Selected {len(top_products)} products:")
    for item in top_products:
        print(f" - {item}")

    selected = df[df["item_id"].isin(top_products)].copy() 
    selected = selected.sort_values(["item_id", "date"])

    output_file = PROCESSED_DIR / "selected_products.csv"
    selected = selected.sort_values(["item_id", "date"])

    output_file = PROCESSED_DIR / "selected_products.csv"
    selected.to_csv(output_file, index=False)

    print(f"\nRecords: {len(selected):,}")
    print(f"Saved to : {output_file}")




if __name__ == "__main__":
    select_products()

