import pandas as pd

input_file = '/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv'
chunk_size = 100000
max_drugs_found = 0

print("Finding the maximum number of concurrent drugs...")

# Read the file in chunks
for chunk in pd.read_csv(input_file, dtype=str, usecols=['drug_combination'], chunksize=chunk_size, low_memory=False):
    # Drop rows without drugs, split by comma, and find the length of each list
    lengths = chunk['drug_combination'].dropna().apply(lambda x: len([d for d in x.split(',') if d.strip()]))
    
    if not lengths.empty:
        chunk_max = lengths.max()
        if chunk_max > max_drugs_found:
            max_drugs_found = chunk_max

print(f"\nThe maximum number of drugs taken by a single patient is: {max_drugs_found}")