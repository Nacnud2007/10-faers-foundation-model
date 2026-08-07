"""Generate distribution plots for drugs and ADRs in the cleaned FAERS data."""
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

INPUT_FILE = "faers_combined_cleaned.csv"

def analyze_and_plot():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Make sure it is in the same folder.")
        return

    df = df.dropna(subset=['drug_combination', 'adrs'])
    print("Splitting strings...")
    all_drugs = []
    all_adrs = []

    for drugs_str in df['drug_combination']:
        all_drugs.extend([d.strip() for d in str(drugs_str).split(';') if d.strip()])

    for adrs_str in df['adrs']:
        all_adrs.extend([a.strip() for a in str(adrs_str).split(';') if a.strip()])

    # Enumerate drugs and ADRs
    unique_drugs = set(all_drugs)
    unique_adrs = set(all_adrs)

    print(f"Unique Drugs Found: {len(unique_drugs):,}")
    print(f"Unique ADRs Found:  {len(unique_adrs):,}")

    # Frequency counts for the histograms
    drug_frequencies = list(Counter(all_drugs).values())
    adr_frequencies = list(Counter(all_adrs).values())

    # Plot
    print("Generating histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Drug Distribution
    axes[0].hist(drug_frequencies, bins=50, color='skyblue', edgecolor='black', log=True)
    axes[0].set_title("Distribution of Drug Frequencies")
    axes[0].set_xlabel("Drug Frequency")
    axes[0].set_ylabel("Unique Drugs (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # ADR Distribution
    axes[1].hist(adr_frequencies, bins=50, color='salmon', edgecolor='black', log=True)
    axes[1].set_title("Distribution of ADR Frequencies")
    axes[1].set_xlabel("ADR Frequency")
    axes[1].set_ylabel("Unique ADRs (Log Scale)")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    
    # Save
    output_image = "faers_histogram_results.png"
    plt.savefig(output_image, dpi=300)
    print(f"Successfully saved")
    
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()
