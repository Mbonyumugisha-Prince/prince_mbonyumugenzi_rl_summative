from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "data" / "processed" / "inventory_data.csv", parse_dates=["date"])
df = df.sort_values(["item_id", "date"])

print(f"Total records: {len(df):,}")
print(f"Products: {df['item_id'].nunique()}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Price missing: {df['sell_price'].isna().mean():.2%}")

# Has the product had a price at least once up to this day?
launched = df["sell_price"].notna().groupby(df["item_id"]).cummax()
pre_launch = df["sell_price"].isna() & ~launched
mid_gap = df["sell_price"].isna() & launched

print(f"\nNaN before launch:            {pre_launch.sum():,}")
print(f"NaN after launch (real gaps): {mid_gap.sum():,}")
print(f"Units sold on pre-launch rows: {df.loc[pre_launch, 'sales'].sum():,}")

never_priced = df.groupby("item_id")["sell_price"].apply(lambda s: s.isna().all())
print(f"Products never priced: {never_priced.sum()}")