import pandas as pd

INPUT_FILE = "faers_combined_cleaned.csv"
OUTPUT_FILE = "faers_combined_cleaned_pure_reactions.csv"

NON_REACTION_TERMS = {
    'Drug ineffective',
    'Product dose omission issue',
    'Product use in unapproved indication',
    'Inappropriate schedule of product administration',
    'Wrong technique in product usage process',
    'Product use issue',
    'Incorrect dose administered',
    'Overdose',
    'No adverse event',
    'Intentional product use issue',
    'Drug dose omission',
    'Accidental exposure to product',
    'Drug dose omission by device',
    'Therapeutic product effect incomplete',
    'Device issue',
    'Product storage error',
    'Treatment failure',
    'Intentional product misuse',
    'Product quality issue',
    'Adverse drug reaction',
    'Therapy interrupted',
    'Product dose omission',
    'Adverse event',
    'Underdose',
    'Drug abuse',
    'Device difficult to use',
    'Device malfunction',
    'Unevaluable event',
    'Therapeutic product effect decreased',
    'Drug ineffective for unapproved indication',
    'Wrong technique in device usage process',
    'Device leakage',
    'Intentional overdose',
    'Inappropriate schedule of drug administration',
    'Contraindicated product administered',
    'Device breakage',
    'Product substitution issue',
    'Device expulsion',
    'Extra dose administered',
    'Product dose omission in error',
    'Device delivery system issue',
    'Needle issue',
    'Drug effect incomplete',
    'Intentional dose omission',
    'Device dislocation',
    'Circumstance or information capable of leading to medication error',
    'Drug effective for unapproved indication',
    'Product adhesion issue',
    'Expired product administered',
    'Product availability issue',
    'Medication error',
    'Product prescribing error',
    'Incorrect dose administered by device',
    'Product complaint',
    'Drug effect decreased',
    'Accidental overdose',
    'Prescribed underdose',
    'Product dispensing error',
    'Product preparation error',
    'Product administration error',
    'Product prescribing issue',
    'Incorrect product administration duration',
    'Syringe issue',
    'Device use issue',
    'Incorrect route of product administration',
    'Insurance issue',
    'Product administered at inappropriate site',
    'Device defective',
    'Device mechanical issue'
}

NON_REACTION_TERMS = {x.lower() for x in NON_REACTION_TERMS} # Standardize drug names to lowercase

def clean_non_reactions(text):
    if pd.isna(text):
        return text
    
    # Split items by semicolon and strip whitespace
    items = [item.strip() for item in str(text).split(";")]
    
    # Keep item only if its lowercase version is not in non-reaction list
    cleaned_items = [item for item in items if item.lower() not in NON_REACTION_TERMS]
    
    # Re-join with a semicolon or return None if the string is now completely empty
    return "; ".join(cleaned_items) if cleaned_items else None

def main():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        initial_rows = len(df)
        
        print("Filtering out device malfunctions and non-reaction terms...")
        # Apply the filter to the ADR column
        df["adrs"] = df["adrs"].apply(clean_non_reactions)
        
        # Drop rows where a patient had no real medical reactions
        print("Dropping rows that are now empty...")
        df.dropna(subset=["adrs"], inplace=True)
        
        final_rows = len(df)
        print(f"Saving pure dataset to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Total rows remaining: {final_rows}")
        print(f"Rows completely removed: {initial_rows - final_rows}")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'.")

if __name__ == "__main__":
    main()