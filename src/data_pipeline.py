from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
DB = ROOT / "data" / "instamart.db"


def audit_frame(frame: pd.DataFrame) -> dict:
    missing = frame.isna().sum()
    return {
        "shape": list(frame.shape),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {
            column: {
                "count": int(missing[column]),
                "percent": round(float(missing[column] / max(len(frame), 1) * 100), 4),
            }
            for column in frame.columns
        },
    }


def audit_raw() -> dict:
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    report = {
        "customers": audit_frame(customers),
        "products": audit_frame(products),
        "orders": audit_frame(orders),
        "tickets": audit_frame(tickets),
        "checkpoints": {
            "raw_order_rows": int(len(orders)),
            "exact_duplicate_order_rows": int(orders.duplicated().sum()),
            "pending_orders": int(orders["status"].eq("pending").sum()),
            "distinct_order_ids": int(orders["order_id"].nunique()),
            "non_numeric_amounts": int(amount.isna().sum()),
            "negative_delivery_times": int(pd.to_numeric(orders["delivery_minutes"], errors="coerce").lt(0).sum()),
            "blank_delivery_times": int(orders["delivery_minutes"].isna().sum()),
            "blank_ages": int(customers["age"].isna().sum()),
            "ages_outside_10_100": int((age.notna() & ~age.between(10, 100)).sum()),
            "raw_city_spellings": int(customers["city"].nunique()),
            "nonblank_raw_gender_spellings": int(customers["gender"].dropna().nunique()),
            "blank_genders": int(customers["gender"].isna().sum()),
        },
    }
    return report


def clean_data() -> dict:
    CLEAN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    if set(customers["city"].unique()) != {
        "Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"
    }:
        raise ValueError("Customer city normalization did not yield the six supported cities.")
    gender = customers["gender"].astype(str).str.strip().str.lower()
    customers["gender"] = gender.map(
        {"male": "male", "m": "male", "female": "female", "f": "female", "": "unknown"}
    ).fillna("unknown")
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    missing_or_invalid_ages = int(
        (customers["age"].isna() | ~customers["age"].between(10, 100)).sum()
    )
    customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    ).astype(int)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    original_order_rows = len(orders)
    orders = orders.drop_duplicates().copy()
    exact_duplicates_removed = original_order_rows - len(orders)
    orders = orders.sort_values(["order_id", "order_ts"], kind="mergesort")
    orders = orders.drop_duplicates(subset="order_id", keep="last").copy()
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    price_map = products.set_index("product_id")["price_inr"]
    expected_amount = (
        orders["product_id"].map(price_map)
        * pd.to_numeric(orders["quantity"], errors="raise")
        * (1 - pd.to_numeric(orders["discount_pct"], errors="raise") / 100)
    ).round(2)
    missing_amounts = int(orders["order_amount"].isna().sum())
    orders["order_amount"] = orders["order_amount"].fillna(expected_amount).round(2)
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_delivery = int(orders["delivery_minutes"].lt(0).sum())
    orders.loc[orders["delivery_minutes"].lt(0), "delivery_minutes"] = np.nan
    category_map = products.set_index("product_id")["category"]
    orders["_category"] = orders["product_id"].map(category_map)
    completed = orders["status"].eq("completed")
    completed_missing_delivery_before_fill = int(
        (completed & orders["delivery_minutes"].isna()).sum()
    )
    category_median = orders.loc[completed].groupby("_category")["delivery_minutes"].transform("median")
    orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(category_median)
    orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")
    orders = orders.drop(columns="_category")
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    for frame in (customers, products, orders):
        if "signup_date" in frame:
            frame["signup_date"] = frame["signup_date"].dt.strftime("%Y-%m-%d")
        if "order_ts" in frame:
            frame["order_ts"] = frame["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)
    with sqlite3.connect(DB) as connection:
        customers.to_sql("customers", connection, if_exists="replace", index=False)
        products.to_sql("products", connection, if_exists="replace", index=False)
        orders.to_sql("orders", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_orders_status_ts ON orders(status, order_ts)")

    completed_orders = orders.loc[orders["status"].eq("completed")]
    cancelled_orders = orders.loc[orders["status"].eq("cancelled")]
    result = {
        "exact_duplicates_removed": exact_duplicates_removed,
        "invalid_amounts_filled": missing_amounts,
        "negative_delivery_times_set_missing": negative_delivery,
        "completed_missing_delivery_filled": completed_missing_delivery_before_fill,
        "customer_ages_imputed": missing_or_invalid_ages,
        "customers": int(len(customers)),
        "customer_city_counts": customers["city"].value_counts().to_dict(),
        "customer_gender_counts": customers["gender"].value_counts().to_dict(),
        "customer_missing_ages": int(customers["age"].isna().sum()),
        "customer_age_min": int(customers["age"].min()),
        "customer_age_max": int(customers["age"].max()),
        "orders": int(len(orders)),
        "order_status_counts": orders["status"].value_counts().to_dict(),
        "missing_amounts": int(orders["order_amount"].isna().sum()),
        "completed_missing_delivery": int(completed_orders["delivery_minutes"].isna().sum()),
        "cancelled_missing_delivery": int(cancelled_orders["delivery_minutes"].isna().sum()),
        "completed_revenue_inr": round(float(completed_orders["order_amount"].sum()), 2),
    }
    return result


def run_sql_and_analysis() -> dict:
    query_path = ROOT / "sql" / "queries.sql"
    query_text = query_path.read_text(encoding="utf-8")
    queries = {}
    current = None
    for line in query_text.splitlines():
        if line.startswith("-- Q"):
            current = line.split("—", 1)[0].strip().replace("-- ", "")
            queries[current] = []
        elif current:
            queries[current].append(line)
    sql_results = {}
    with sqlite3.connect(DB) as connection:
        for name, lines in queries.items():
            sql = "\n".join(lines).strip()
            sql_results[name] = pd.read_sql_query(sql, connection)

    with sqlite3.connect(DB) as connection:
        orders = pd.read_sql_query("SELECT * FROM orders", connection, parse_dates=["order_ts"])
        customers = pd.read_sql_query("SELECT * FROM customers", connection)
        products = pd.read_sql_query("SELECT * FROM products", connection)
    completed = orders.loc[orders["status"].eq("completed")].copy()
    joined = completed.merge(customers[["customer_id", "city"]], on="customer_id").merge(
        products[["product_id", "category"]], on="product_id"
    )
    pivot = pd.pivot_table(
        joined, values="order_amount", index="city", columns="category",
        aggfunc="sum", fill_value=0, margins=True, margins_name="All",
    )
    city_stats = completed.merge(customers[["customer_id", "city"]], on="customer_id").groupby(
        "city", as_index=False
    ).agg(
        sum_inr=("order_amount", "sum"),
        mean_inr=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    amount = completed["order_amount"]
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    corr = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    pairs = [
        (left, right, float(corr.loc[left, right]))
        for i, left in enumerate(corr.columns)
        for right in corr.columns[i + 1 :]
    ]
    strongest = max(pairs, key=lambda item: abs(item[2]))
    analysis = {
        "sql": {
            name: {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "records": frame.replace({np.nan: None}).to_dict(orient="records"),
            }
            for name, frame in sql_results.items()
        },
        "pivot_grand_total_inr": round(float(pivot.loc["All", "All"]), 2),
        "city_stats": city_stats.round({"sum_inr": 2, "mean_inr": 2}).to_dict(orient="records"),
        "completed_order_amounts": {
            "mean": float(amount.mean()),
            "median": float(amount.median()),
            "skewness": float(amount.skew()),
            "q1": float(q1),
            "q3": float(q3),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outliers_below": int(amount.lt(lower).sum()),
            "outliers_above": int(amount.gt(upper).sum()),
        },
        "correlation_matrix": corr.round(6).to_dict(),
        "largest_absolute_correlation": {
            "left": strongest[0], "right": strongest[1], "value": strongest[2]
        },
        "pivot": pivot.round(2).reset_index().to_dict(orient="records"),
    }
    (OUT / "part_a_report.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    return analysis


def validate_checkpoints(clean_report: dict, raw_report: dict, analysis: dict) -> None:
    raw = raw_report["checkpoints"]
    assert raw["raw_order_rows"] == 13084
    assert raw["exact_duplicate_order_rows"] == 119
    assert raw["pending_orders"] == 656
    assert raw["distinct_order_ids"] == 12309
    assert raw["non_numeric_amounts"] == 120
    assert clean_report["orders"] == 12309
    assert clean_report["order_status_counts"] == {"completed": 11110, "cancelled": 1199}
    assert clean_report["completed_revenue_inr"] == 2998668.80
    assert clean_report["customer_missing_ages"] == 0
    assert clean_report["customer_city_counts"] == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191,
        "Hyderabad": 191, "Mumbai": 184,
    }
    assert analysis["sql"]["Q1"]["rows"] == 30
    assert analysis["sql"]["Q2"]["rows"] == 30
    assert analysis["sql"]["Q3"]["rows"] == 11
    assert analysis["sql"]["Q4"]["rows"] == 25
    assert analysis["sql"]["Q5"]["rows"] == 550
    assert analysis["sql"]["Q7"]["rows"] == 183
    print("ALL CHECKPOINTS PASSED")


def run() -> dict:
    raw_report = audit_raw()
    clean_report = clean_data()
    analysis = run_sql_and_analysis()
    validate_checkpoints(clean_report, raw_report, analysis)
    combined = {"raw_audit": raw_report, "cleaning": clean_report, "analysis": analysis}
    (OUT / "part_a_full_report.json").write_text(
        json.dumps(combined, indent=2, default=str), encoding="utf-8"
    )
    return combined


if __name__ == "__main__":
    run()
