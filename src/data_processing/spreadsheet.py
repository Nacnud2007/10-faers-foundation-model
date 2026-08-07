"""Create a frequency spreadsheet of common drugs and ADRs from FAERS."""
import pandas as pd
from collections import Counter

INPUT_FILE = "faers_combined_cleaned_pure_reactions.csv"
OUTPUT_FILE = "frequent_drugs_ADRs.csv"

def generate_frequency_spreadsheet():
    print (f"Loadng '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df = df.dropna(subset=['drug_combination', 'adrs']) # drop missing data 
    all_drugs = []
    all_adrs = []

    print("Counting terms")
    for drugs_str in df['drug_combination']:
        all_drugs.extend([d.strip() for d in str(drugs_str).split(';') if d.strip()])

    for adrs_str in df['adrs']:
        all_adrs.extend([a.strip() for a in str(adrs_str).split(';') if a.strip()])

    # Finding top 2,000 most common drugs and top 1,000 ADRs
    top_drugs = Counter(all_drugs).most_common(2000)
    top_adrs = Counter(all_adrs).most_common(1000)

    # Create data frame with both top_drugs and top_ADRs
    df_top_drugs = pd.DataFrame(top_drugs, columns=['Drug', 'Frequency'])
    df_top_adrs = pd.DataFrame(top_adrs, columns=['ADR', 'Frequency'])

    spreadsheet = pd.concat([df_top_drugs, df_top_adrs], axis=1)

    # Save spreadsheet
    spreadsheet.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved '{OUTPUT_FILE}' with top drugs and ADRs.")

if __name__ == "__main__":
    generate_frequency_spreadsheet()
