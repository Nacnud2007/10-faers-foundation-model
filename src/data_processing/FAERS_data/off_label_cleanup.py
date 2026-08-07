"""Remove off-label use terms from the cleaned FAERS dataset."""
import pandas as pd

INPUT_FILE = "faers_combined_cleaned.csv"
OUTPUT_FILE = "faers_combined_cleaned.csv"


def clean_off_label(text):
    if pd.isna(text):
        return text

    # Split items by semicolon and strip whitespace
    items = [item.strip() for item in str(text).split(";")]

    # Filter out 'Off label use'
    cleaned_items = [item for item in items if item.lower() != "off label use"]

    # Re-join with a semicolon or return None if empty
    return "; ".join(cleaned_items) if cleaned_items else None


def clean_existing_csv():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        initial_rows = len(df)
        print(f"Loaded {initial_rows} rows successfully.")

        print("Removing 'Off label use' instances...")
        # Apply the cleaning function to both target data columns
        df["drug_combination"] = df["drug_combination"].apply(clean_off_label)
        df["adrs"] = df["adrs"].apply(clean_off_label)

        # Drop rows where it was just "Off label use"
        print("Dropping rows that are now empty...")
        df.dropna(subset=["drug_combination", "adrs"], inplace=True)

        final_rows = len(df)
        rows_removed = initial_rows - final_rows

        print(f"Saving cleaned dataset to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n--- Done! ---")
        print(f"Total rows remaining: {final_rows}")
        print(f"Rows completely removed (due to being empty): {rows_removed}")

    except FileNotFoundError:
        print(
            f"Error: Could not find '{INPUT_FILE}'. Make sure this script is in the same folder as your CSV."
        )



if __name__ == "__main__":
    clean_existing_csv()
