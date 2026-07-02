import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp

print('Loading...')
adata = sc.read_h5ad('sciplex.h5ad')

print('Pseudobulking without loading full matrix...')
drugs = adata.obs['perturbation'].unique()
print(f'{len(drugs)} unique drugs')

# Use integer positions instead of gene names to avoid duplicate index issues
results = []
drug_names = []

for i, drug in enumerate(drugs):
    mask = adata.obs['perturbation'] == drug
    subset = adata[mask]
    if subset.n_obs == 0:
        continue
    X = subset.X
    if sp.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).flatten()
    else:
        mean_expr = X.mean(axis=0).flatten()
    results.append(mean_expr)
    drug_names.append(drug)
    if i % 10 == 0:
        print(f'{i}/{len(drugs)} done')

print('Saving...')
gene_names = adata.var['ensembl_id'].values
pseudobulk = pd.DataFrame(results, columns=gene_names)
pseudobulk.insert(0, 'drug_name', drug_names)
pseudobulk.to_csv('pseudobulk_perturb.csv', index=False)
print(f'Done. Shape: {pseudobulk.shape}')

