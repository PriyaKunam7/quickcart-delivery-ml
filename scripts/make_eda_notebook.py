"""One-time script to scaffold the EDA notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Delivery Time EDA\n"
        "Exploratory analysis of `data/raw/orders.csv` for the QuickCart delivery prediction platform."
    )
)

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "../data/raw/orders.csv",
    dtype={"origin_zip": str, "destination_zip": str},
    parse_dates=["order_timestamp"],
)
df.shape"""))

cells.append(nbf.v4.new_markdown_cell("## Missing Values"))
cells.append(nbf.v4.new_code_cell("df.isnull().sum()"))

cells.append(nbf.v4.new_markdown_cell("## Cardinality of Categorical Fields"))
cells.append(
    nbf.v4.new_code_cell(
        """for col in ["carrier_code", "item_category", "origin_zip", "destination_zip"]:
    print(col, "->", df[col].nunique(), "unique values")"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Target Distribution"))
cells.append(nbf.v4.new_code_cell("""df["actual_delivery_days"].describe()"""))
cells.append(nbf.v4.new_code_cell("""df["actual_delivery_days"].hist(bins=50)
plt.xlabel("actual_delivery_days")
plt.ylabel("count")
plt.title("Target Distribution")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## Delivery Days by Carrier"))
cells.append(
    nbf.v4.new_code_cell(
        """df.groupby("carrier_code")["actual_delivery_days"].mean().sort_values().plot(kind="bar")
plt.ylabel("mean actual_delivery_days")
plt.title("Delivery Days by Carrier")
plt.show()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Delivery Days by Item Category"))
cells.append(
    nbf.v4.new_code_cell(
        """df.groupby("item_category")["actual_delivery_days"].mean().sort_values().plot(kind="bar")
plt.ylabel("mean actual_delivery_days")
plt.title("Delivery Days by Item Category")
plt.show()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Distance vs Target"))
cells.append(
    nbf.v4.new_code_cell("""print(df[["distance_miles", "actual_delivery_days"]].corr())
sample_idx = df["distance_miles"].sample(2000, random_state=42).index
plt.scatter(df.loc[sample_idx, "distance_miles"], df.loc[sample_idx, "actual_delivery_days"], alpha=0.3, s=5)
plt.xlabel("distance_miles")
plt.ylabel("actual_delivery_days")
plt.title("Distance vs Delivery Days")
plt.show()""")
)

cells.append(nbf.v4.new_markdown_cell("## Weather Impact"))
cells.append(
    nbf.v4.new_code_cell(
        """df["weather_bin"] = pd.cut(df["weather_score"], bins=[0, 3, 7, 10], labels=["low", "medium", "high"])
df.groupby("weather_bin", observed=True)["actual_delivery_days"].mean().plot(kind="bar")
plt.ylabel("mean actual_delivery_days")
plt.title("Delivery Days by Weather Severity")
plt.show()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Holiday Impact"))
cells.append(
    nbf.v4.new_code_cell(
        """df.groupby("holiday_flag")["actual_delivery_days"].mean()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Warehouse Processing Impact"))
cells.append(
    nbf.v4.new_code_cell(
        """df[["warehouse_processing_hours", "actual_delivery_days"]].corr()"""
    )
)

cells.append(nbf.v4.new_markdown_cell("## Outlier Ranges (IQR method)"))
cells.append(
    nbf.v4.new_code_cell(
        """for col in ["distance_miles", "warehouse_processing_hours", "actual_delivery_days"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: IQR bounds=[{lower:.2f}, {upper:.2f}]  outliers={n_outliers}")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## Summary\\n\\nSee `reports/eda_summary.md` for the written findings, including target leakage notes."
    )
)

nb["cells"] = cells

import os

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_delivery_eda.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created at notebooks/01_delivery_eda.ipynb")
