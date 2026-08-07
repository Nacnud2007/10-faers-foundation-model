"""Identify first-occurrence drug combinations from processed FAERS data."""
import pandas as pd
from collections import Counter

# ============================================================
# FILE PATHS

FAERS_FILE = "data/processed/faers_combined_cleaned_pure_reactions.csv"
MAPPING_FILE = "data/processed/drug_name_mapping_updated.csv"

OUTPUT_COMBOS = "output/new_drug_combinations_2022Q1_2026Q1.csv"
OUTPUT_REPORTS = "output/reports_with_new_drug_combinations.csv"

# Load data
print("Loading data...")

faers = pd.read_csv(FAERS_FILE, low_memory=False)
mapping = pd.read_csv(MAPPING_FILE, low_memory=False)

# Build dictionary

messy_col = mapping.columns[0]
standard_col = mapping.columns[2]

name_map = {}

for _, row in mapping.iterrows():

    messy = str(row[messy_col]).strip().upper()

    standard = str(row[standard_col]).strip()

    if messy != "" and standard != "":
        name_map[messy] = standard

print(f"Loaded {len(name_map):,} standardized drug names.")

combo_col = "Drug Combination"

def standardize_combo(combo):

    if pd.isna(combo):
        return None

    drugs = []

    for drug in str(combo).split(","):

        drug = drug.strip()

        if drug == "":
            continue

        standardized = name_map.get(drug.upper(), drug)

        drugs.append(standardized)

    # remove duplicates
    drugs = sorted(set(drugs))

    return "|".join(drugs)

print("Standardizing drug combinations...")

faers["combo_key"] = faers[combo_col].apply(standardize_combo)

quarter_col = "Quarter"

def quarter_number(q):

    q = str(q).strip().upper()

    year = int(q[:4])

    quarter = int(q[-1])

    return year * 4 + quarter

faers["quarter_num"] = faers[quarter_col].apply(quarter_number)

# FIND FIRST APPEARANCE
print("Finding first appearance of every combination...")

first_seen = (
    faers.groupby("combo_key")
         .agg(
             first_quarter_num=("quarter_num", "min"),
             reports=("combo_key", "size")
         )
         .reset_index()
)

# recover readable quarter

reverse_lookup = (
    faers[["quarter_num", quarter_col]]
    .drop_duplicates()
)

first_seen = first_seen.merge(
    reverse_lookup,
    left_on="first_quarter_num",
    right_on="quarter_num",
    how="left"
)

# KEEP ONLY 2022Q1 -> 2026Q1
start = quarter_number("2022 Q1")
end = quarter_number("2026 Q1")

new_combos = first_seen[
    (first_seen.first_quarter_num >= start) &
    (first_seen.first_quarter_num <= end)
].copy()

print(f"\nUnique new combinations: {len(new_combos):,}")

reports = faers.merge(
    new_combos[["combo_key"]],
    on="combo_key",
    how="inner"
)

print(f"Total reports involving new combinations: {len(reports):,}")

print("\nNew combinations by first quarter:")

summary = (
    new_combos.groupby(quarter_col)
              .size()
              .sort_index()
)

print(summary)

new_combos = new_combos.rename(
    columns={
        quarter_col: "First Quarter",
        "reports": "Number of Reports"
    }
)

new_combos.to_csv(
    OUTPUT_COMBOS,
    index=False
)

reports.to_csv(
    OUTPUT_REPORTS,
    index=False
)

print("\nDone!")

print(f"Saved:")
print(f"  {OUTPUT_COMBOS}")
print(f"  {OUTPUT_REPORTS}")
