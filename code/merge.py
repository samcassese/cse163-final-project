from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
csv_dir = base_dir / "datasets"
csv_files = sorted(csv_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_dir}")

merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
merged_df.to_csv(csv_dir / "merged.csv", index=False)