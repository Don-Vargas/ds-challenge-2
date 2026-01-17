import pandas as pd

def pandas_summary(df, report_path: str):

    summary = pd.DataFrame({
        "Type": df.dtypes,
        "Non-Null Count": df.count(),
        "Missing": df.isna().sum(),
        "% Missing": df.isna().mean(),
        "Unique Values": df.nunique(),
    })

    # Numeric summaries
    numeric_desc = df.describe(include="number").T[
        ["min", "max", "mean", "std"]
    ]

    # Categorical summaries
    categorical_desc = df.describe(include="object").T[
        ["top", "freq"]
    ].rename(columns={"top": "Mode", "freq": "Mode Count"})

    # Merge everything
    summary = summary.join(numeric_desc, how="left")
    summary = summary.join(categorical_desc, how="left")

    summary.reset_index(names="Column", inplace=True)
    summary.sort_values(
        by=["Mode", "Unique Values"],
        key=lambda col: col.notna() if col.name == "Mode" else col,
        ascending=[False, True],
        inplace=True,
        )
    summary.to_csv(report_path, index=False)
