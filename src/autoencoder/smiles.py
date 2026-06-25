import os
import pandas as pd
import numpy as np
import pubchempy as pcp
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

# File Paths
MAPPING_FILE = '/Users/duncanpark/10-faers-foundation-model/data/processed/drug_name_mapping_updated.csv'
PATIENT_FILE = '/Users/duncanpark/10-faers-foundation-model/data/processed/faers_combined_cleaned_pure_reactions.csv' 
CACHE_FILE = 'cid_to_smiles_cache.csv'

# Output configs - NOTE: X is now a sparse .npz file!
X_OUTPUT_FILE = 'X_train_sparse.npz'
Y_OUTPUT_FILE = 'Y_train_sparse.npz'
ADR_VOCAB_FILE = 'adr_vocabulary.txt'

# Temporary coordinate files
TMP_X_ROWS = 'tmp_x_rows.npy'
TMP_X_COLS = 'tmp_x_cols.npy'
TMP_Y_ROWS = 'tmp_y_rows.npy'
TMP_Y_COLS = 'tmp_y_cols.npy'

print("Loading mapping file...")
mapping_df = pd.read_csv(MAPPING_FILE)
name_to_cid = dict(zip(mapping_df['messy_name'].str.lower().str.strip(), mapping_df['pubchem_cid']))

# Local Smiles Lookup
cid_to_smiles = {}
if os.path.exists(CACHE_FILE):
    print(f"Loading cached SMILES from local file: {CACHE_FILE}")
    cache_df = pd.read_csv(CACHE_FILE)
    cid_to_smiles = dict(zip(cache_df['cid'].astype(int), cache_df['smiles']))

unique_cids = mapping_df['pubchem_cid'].dropna().unique().astype(int).tolist()
cids_to_fetch = [cid for cid in unique_cids if cid not in cid_to_smiles]

if len(cids_to_fetch) > 0:
    print(f"Found {len(cids_to_fetch)} new CIDs to fetch from PubChem...")
    chunk_size = 20
    for i in range(0, len(cids_to_fetch), chunk_size):
        chunk = cids_to_fetch[i:i + chunk_size]
        try:
            compounds = pcp.get_compounds(chunk, 'cid', timeout=10)
            for comp in compounds:
                if comp.smiles:
                    cid_to_smiles[comp.cid] = comp.smiles
            pd.DataFrame(list(cid_to_smiles.items()), columns=['cid', 'smiles']).to_csv(CACHE_FILE, index=False)
        except Exception:
            continue

# Pre-compute fingerprints in memory
print("Pre-computing molecular fingerprints for all unique drug names...")
name_to_fingerprint = {}
for messy_name in mapping_df['messy_name'].dropna().unique():
    clean_name = str(messy_name).strip().lower()
    cid = name_to_cid.get(clean_name)
    drug_vector = np.zeros(3095, dtype=np.uint8)
    if cid and not pd.isna(cid):
        smiles = cid_to_smiles.get(int(cid))
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=np.uint8)[1:]
                morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol).astype(np.uint8)
                pubchem = np.zeros(881, dtype=np.uint8)
                drug_vector = np.concatenate([maccs, pubchem, morgan])
    name_to_fingerprint[clean_name] = drug_vector

# Build ADR Vocab
print("Checking file columns...")
sample_df = pd.read_csv(PATIENT_FILE, nrows=1)

target_column = 'adrs'
if target_column not in sample_df.columns:
    possible_matches = [col for col in sample_df.columns if 'adverse' in col.lower() or 'reaction' in col.lower() or 'event' in col.lower()]
    target_column = possible_matches[0] if possible_matches else "adrs"

drug_column = 'Drug_Combination'
if drug_column not in sample_df.columns:
    possible_drug_matches = [col for col in sample_df.columns if 'drug' in col.lower() or 'combo' in col.lower()]
    drug_column = possible_drug_matches[0] if possible_drug_matches else "Drug_Combination"

print(f"Scanning unique ADRs using column '{target_column}'...")
unique_adrs = set()
for chunk in pd.read_csv(PATIENT_FILE, dtype=str, usecols=[target_column], chunksize=100000, low_memory=False):
    for val in chunk[target_column].dropna():
        for adr in val.split(','):
            unique_adrs.add(adr.strip())

all_adrs = sorted(list(unique_adrs))
adr_to_index = {adr: idx for idx, adr in enumerate(all_adrs)}
num_unique_adrs = len(all_adrs)
print(f"Found {num_unique_adrs:,} unique adverse events in the dataset.")

with open(ADR_VOCAB_FILE, 'w') as f:
    f.write('\n'.join(all_adrs))

print("Processing rows with strict memory caps...")

# Clean up stale files
for f in [X_OUTPUT_FILE, Y_OUTPUT_FILE, TMP_X_ROWS, TMP_X_COLS, TMP_Y_ROWS, TMP_Y_COLS]:
    if os.path.exists(f): os.remove(f)

lines_processed = 0
chunk_size_streaming = 50000

# Coordinate trackers for sparse streaming
x_rows_acc, x_cols_acc = [], []
y_rows_acc, y_cols_acc = [], []

MAX_DRUGS = 5
FP_SIZE = 3095
TOTAL_X_FEATURES = MAX_DRUGS * FP_SIZE # 15,475

# Add an explicit zeros vector to use as padding for empty slots
name_to_fingerprint['__pad__'] = np.zeros(FP_SIZE, dtype=np.uint8)

for chunk in pd.read_csv(PATIENT_FILE, dtype=str, chunksize=chunk_size_streaming, low_memory=False):
    for idx, row in chunk.iterrows():
        # 1. Process Drugs (X)
        raw_drugs = [d.strip().lower() for d in str(row[drug_column]).split(',') if d.strip()]
        truncated_drugs = raw_drugs[:MAX_DRUGS]
        
        # Build the 10,355 vector entry by entry, tracking only where the 1s are
        for slot_idx in range(MAX_DRUGS):
            if slot_idx < len(truncated_drugs):
                drug = truncated_drugs[slot_idx]
                drug_vector = name_to_fingerprint.get(drug, name_to_fingerprint['__pad__'])
            else:
                drug_vector = name_to_fingerprint['__pad__']
            
            # Find index locations of 1s inside this specific 2,071 slot
            one_indices = np.where(drug_vector == 1)[0]
            for local_col in one_indices:
                # Map local 2071 fingerprint coordinate to the global 10355 width layout
                global_col = (slot_idx * FP_SIZE) + local_col
                x_rows_acc.append(lines_processed)
                x_cols_acc.append(global_col)
        
        # 2. Process ADR Targets (Y)
        raw_adrs = [a.strip() for a in str(row[target_column]).split(',')]
        for adr in raw_adrs:
            if adr in adr_to_index:
                y_rows_acc.append(lines_processed)
                y_cols_acc.append(adr_to_index[adr])
                
        lines_processed += 1

    # Stream out the coordinates to disk at the end of each chunk block
    with open(TMP_X_ROWS, 'ab') as f_xr, open(TMP_X_COLS, 'ab') as f_xc, open(TMP_Y_ROWS, 'ab') as f_yr, open(TMP_Y_COLS, 'ab') as f_yc:
        np.save(f_xr, np.array(x_rows_acc, dtype=np.int32))
        np.save(f_xc, np.array(x_cols_acc, dtype=np.int32))
        np.save(f_yr, np.array(y_rows_acc, dtype=np.int32))
        np.save(f_yc, np.array(y_cols_acc, dtype=np.int32))
        
    # Clear the temporary batch trackers
    x_rows_acc, x_cols_acc = [], []
    y_rows_acc, y_cols_acc = [], []
    print(f"   Progress: {lines_processed:,} / 14,806,532 rows logged.")

print("\nAll rows logged! Building final compressed matrices from coordinate maps...")

# Finalize X Sparse Matrix
final_x_rows = np.load(TMP_X_ROWS)
final_x_cols = np.load(TMP_X_COLS)
final_x_data = np.ones(len(final_x_rows), dtype=np.uint8)
X_sparse_matrix = sparse.csr_matrix((final_x_data, (final_x_rows, final_x_cols)), shape=(lines_processed, TOTAL_X_FEATURES), dtype=np.uint8)

print(f"Saving sparse features matrix to: {X_OUTPUT_FILE}")
sparse.save_npz(X_OUTPUT_FILE, X_sparse_matrix)

# Free up memory/disk
del final_x_rows, final_x_cols, final_x_data, X_sparse_matrix
os.remove(TMP_X_ROWS)
os.remove(TMP_X_COLS)

# Finalize Y Sparse Matrix
final_y_rows = np.load(TMP_Y_ROWS)
final_y_cols = np.load(TMP_Y_COLS)
final_y_data = np.ones(len(final_y_rows), dtype=np.uint8)
Y_sparse_matrix = sparse.csr_matrix((final_y_data, (final_y_rows, final_y_cols)), shape=(lines_processed, num_unique_adrs), dtype=np.uint8)

print(f"Saving sparse targets matrix to: {Y_OUTPUT_FILE}")
sparse.save_npz(Y_OUTPUT_FILE, Y_sparse_matrix)

# Clean up remaining temp files
os.remove(TMP_Y_ROWS)
os.remove(TMP_Y_COLS)

print("\n Successfully Completed All Rows! ")