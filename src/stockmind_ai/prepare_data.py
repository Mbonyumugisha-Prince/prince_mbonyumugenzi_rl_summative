from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT  / "data" / "processed"


def prepare_dataset():

    OUTPUT_DIR.mkdir(parents=True , exist_ok=True)

    print("Loading datasets ..........")

    sales = pd.read_csv(DATA_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")


    print("Filterin products .......")

    sales = sales[
        (sales["store_id"] == "CA_1") &
        (sales["cat_id"] == "HOBBIES")
    ].copy()


    print (f"Selected  product series : {len(sales)}")

    daily_columns = [
        col for col in sales.columns
        if col.startswith("d_")
    ]

    sales_long = sales.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id"
        ],
        value_vars=daily_columns,
        var_name="d",
        value_name="sales"
    )

    print("Marging calender information......")

    dataset = sales_long.merge(
        calendar,
        on="d",
        how="left",
    )


    print("Merging produc prices .........")

    dataset = dataset.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["item_id", "date"])

    # Remove days before each product launched (no price = not on sale yet)
    launched = dataset["sell_price"].notna().groupby(dataset["item_id"]).cummax()
    dataset = dataset[launched].copy()

    print(f"Records after removing pre-launch days: {len(dataset):,}")
    print(f"Remaining missing prices: {dataset['sell_price'].isna().sum()}")

    output_file = OUTPUT_DIR / "inventory_data.csv"

    dataset.to_csv(output_file, index=False)

    print("\nDataset preparation completed!")
    print(f"Total records: {len(dataset):,}")
    print(f"Saved to: {output_file}")

    print("\nPreview:") 

    print(
        dataset[
            ["date", "item_id", "sales", "sell_price"]
        ].head(10)
    )




if __name__ == "__main__":
    prepare_dataset()

