from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    ROOT / "data" / "processed" / "inventory_data.csv",
    parse_dates=["date"],
)

print("Columns:" , df.columns.to_list())

stats = df.groupby("item_id").agg(
    days=("sales", "size"),
    total_units = ("sales", "sum"),
    mean_daily = ("sales", "mean"),
    std_daily=("sales", "std"),
    zero_days_pct =("sales", lambda s: (s==0).mean() * 100),
    max_daily = ("sales", "max"),
    mean_price=("sell_price", "mean")
)

full_days = df["date"].nunique()
stats = stats[stats["days"] >= 0.9 * full_days]
stats = stats.sort_values("total_units", ascending=False)

print(f"\nProducts with near-ful history : {len(stats)}")

pd.set_option("display.width" , 140)

print("\nTop 15 by total units sold:")
print(stats.head(15).round(2))