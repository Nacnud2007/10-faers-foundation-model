'''
Check # ADRs in the file
'''
from pathlib import Path
import numpy as np
from scipy import sparse

project_root = Path(__file__).resolve().parent

y_path = '/Users/duncanpark/10-faers-foundation-model/Y_train_sparse.npz'
vocab_path = '/Users/duncanpark/10-faers-foundation-model/adr_vocabulary.txt'

print("=" * 50)
print("Checking ADR Dataset Dimensions")
print("=" * 50)

y_matrix = sparse.load_npz(y_path)
n_rows, n_adrs = y_matrix.shape
print(f"Y_train_sparse.npz Shape: {n_rows:,} rows x {n_adrs:,} ADR columns")
    
# Calculate non-zero signals across all rows
total_signals = y_matrix.nnz
print(f"Total Positive ADR Signals Recorded: {total_signals:,}")

vocab_lines = vocab_path.read_text().splitlines()
print(f"adr_vocabulary.txt Total Entries: {len(vocab_lines):,}")
print("\nFirst 5 ADRs in vocabulary:")
for adr in vocab_lines[:5]:
    print(f"  - {adr}")
