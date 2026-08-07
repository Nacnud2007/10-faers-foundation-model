"""Build an initial FAERS drug-name-to-PubChem mapping file."""
import pandas as pd
import pubchempy as pcp
import time
import os

INPUT_CSV = "faers_combined_cleaned_pure_Reactions.csv"
MAPPING_CSV = "data/drug_name_mapping.csv"
DRUG_COLUMN_NAME = "drug_combination"

def extract_true_top_20000(file_path, col_name):
    """Reads the master CSV in chunks, splits semicolon combinations, and finds top 20,000 individual drugs."""
    print(f"Scanning {file_path} and parsing drug combinations...")
    
    global_counts = pd.Series(dtype='int64')
    chunk_size = 50000
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=[col_name], low_memory=False):
        active_series = chunk[col_name].dropna().astype(str)
        
        individual_drugs = active_series.str.split('; ').explode()
        
        individual_drugs = individual_drugs.str.strip()
        
        chunk_counts = individual_drugs.value_counts()
        global_counts = global_counts.add(chunk_counts, fill_value=0)
        
    top_20000 = global_counts.nlargest(20000).index.tolist()
    print(f"Successfully extracted the top {len(top_20000):,} unique individual drugs from the combinations.")
    return top_20000

def query_pubchem(drug_list):
    """Queries PubChem, keeps biologics as text, cleans trailing dots, and handles integers."""
    mapping_data = []
    total = len(drug_list)
    
    print(f"Starting PubChem API lookups for {total:,} drugs...")
    
    for idx, messy_name in enumerate(drug_list):
        # Progress indicator & auto-save checkpoint every 100 entries
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{total} ({idx/total*100:.1f}%)")
            checkpoint_df = pd.DataFrame(mapping_data)
            if 'pubchem_cid' in checkpoint_df.columns:
                checkpoint_df['pubchem_cid'] = checkpoint_df['pubchem_cid'].astype('Int64')
            checkpoint_df.to_csv(MAPPING_CSV, index=False)
            
        if not messy_name or len(str(messy_name).strip()) < 2:
            continue
            
        # Clean up whitespace and fix the trailing period bug!
        raw_string = str(messy_name).strip()
        clean_search_term = raw_string.strip('.')
            
        try:
            results = pcp.get_compounds(clean_search_term, 'name')
            
            if results:
                best_match = results[0]
                cid = int(best_match.cid)  # Force explicit integer
                standard_name = clean_search_term.lower()
                
                mapping_data.append({
                    "messy_name": raw_string,
                    "pubchem_cid": cid,
                    "standardized_name": standard_name.lower()
                })
            else:
                # For biologics: Keep name as lowercase string, leave CID blank
                mapping_data.append({
                    "messy_name": raw_string,
                    "pubchem_cid": None,
                    "standardized_name": clean_search_term.lower()
                })
        except Exception as e:
            print(f"ERROR for {clean_search_term}: {e}")

            mapping_data.append({
                "messy_name": raw_string,
                "pubchem_cid": None,
                "standardized_name": clean_search_term.lower()
            })
            
        # 200ms delay for pubchem.py to work
        time.sleep(0.2)
        
    return mapping_data

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    top_drugs = extract_true_top_20000(INPUT_CSV, DRUG_COLUMN_NAME)
    results_list = query_pubchem(top_drugs)
    mapping_df = pd.DataFrame(results_list)
    mapping_df['pubchem_cid'] = mapping_df['pubchem_cid'].astype('Int64')  # Nullable integer type
    
    mapping_df.to_csv(MAPPING_CSV, index=False)
    print(f"\nSuccess! Your master 20,000 drug dictionary is saved to {MAPPING_CSV}")
