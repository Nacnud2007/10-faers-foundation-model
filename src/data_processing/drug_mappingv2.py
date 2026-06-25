import pandas as pd
import pubchempy as pcp
import re
import time

MAPPING_CSV = "data/drug_name_mapping.csv"
OUTPUT_CSV = "data/drug_name_mapping_updated.csv"


def clean_drug_name(name):
    """Clean FAERS drug names before PubChem lookup."""

    name = str(name).strip()
    name = name.rstrip('.')
    name = re.sub(r'/\d+/', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    name = ' '.join(name.split())

    return name.strip()


print("Loading existing mapping file...")
df = pd.read_csv(MAPPING_CSV)

failed_mask = df["pubchem_cid"].isna()

failed_drugs = (
    df.loc[failed_mask, "messy_name"]
    .dropna()
    .unique()
    .tolist()
)

print(f"Found {len(failed_drugs):,} failed drug names to retry.")

updates = {}

for i, drug in enumerate(failed_drugs):
    clean_name = clean_drug_name(drug)

    if len(clean_name) < 2:
        continue

    try:
        results = pcp.get_compounds(clean_name, "name")

        if results:
            cid = int(results[0].cid)

            updates[drug] = {
                "pubchem_cid": cid,
                "standardized_name": clean_name.lower()
            }

            print(f"SUCCESS: {drug} -> {cid}")

    except Exception as e:
        print(f"ERROR: {drug} -> {e}")

    time.sleep(0.2)

    if i % 100 == 0 and i > 0:
        print(f"Processed {i}/{len(failed_drugs)}")


print("\nApplying updates...")

for original_name, info in updates.items():

    mask = df["messy_name"] == original_name

    df.loc[mask, "pubchem_cid"] = info["pubchem_cid"]
    df.loc[mask, "standardized_name"] = info["standardized_name"]


df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved updated file to: {OUTPUT_CSV}")

print(f"Recovered {len(updates):,} additional PubChem mappings.")