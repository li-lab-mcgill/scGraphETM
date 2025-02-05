import numpy as np
import pandas as pd
import anndata as ad
import pickle
import scanpy as sc
import xgboost as xgb
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix, bmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pyarrow.feather as feather
import os
import argparse
from scipy.stats import pearsonr


def prepare_matrix(atac_adata):
    return {peak: idx for idx, peak in enumerate(atac_adata.var_names)}


def process_gene_to_matrix(gene_idx, rna_adata, atac_adata, gene_chrom, gene_start, distance, top, method,
                           peak_indices):
    connections = np.zeros(len(peak_indices))
    chrom_mask = atac_adata.var['chrom'] == gene_chrom
    same_chrom_peaks_adata = atac_adata[:, chrom_mask]
    same_chrom_peaks_names = atac_adata.var[chrom_mask].index
    mse, std = 0, 0

    if method in ['nearby', 'both']:
        # Calculate distances for all peaks
        distances = np.abs(same_chrom_peaks_adata.var['chromStart'].astype(int) - gene_start)
        # Filter peaks by distance
        nearby_peaks = distances[distances <= distance].index
        if len(nearby_peaks) > 0:
            numerical_indices = [peak_indices[name] for name in nearby_peaks]
            connections[numerical_indices] = 1

    if method in ['correlation', 'both']:
        # Calculate correlations for all peaks on same chromosome
        gene_expr = rna_adata.X[:, gene_idx].toarray().flatten() if sp.issparse(rna_adata.X) else rna_adata.X[:, gene_idx]
        peak_access = same_chrom_peaks_adata.X.toarray() if sp.issparse(same_chrom_peaks_adata.X) else same_chrom_peaks_adata.X

        # Calculate correlations and p-values for all peaks
        correlations = []
        p_values = []
        for i in range(peak_access.shape[1]):
            corr, p_val = pearsonr(gene_expr, peak_access[:, i])
            correlations.append(corr)
            p_values.append(p_val)

        # Set correlation thresholds
        corr_threshold = 0.3
        p_value_threshold = 0.05

        # Find significant peaks
        significant_peaks = [i for i in range(len(correlations))
                           if (abs(correlations[i]) > corr_threshold and
                               p_values[i] < p_value_threshold)]

        if significant_peaks:
            sig_peak_names = [same_chrom_peaks_names[i] for i in significant_peaks]
            numerical_indices = [peak_indices[name] for name in sig_peak_names]
            connections[numerical_indices] = 1

    if method == 'gbm':
        # GBM implementation
        X = same_chrom_peaks_adata.X
        y = rna_adata.X[:, gene_idx]
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        y_dense = y.toarray() if hasattr(y, "toarray") else y
        X_train, X_val, y_train, y_val = train_test_split(X_dense, y_dense, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.01, max_depth=6,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=10,
            random_state=42, device="cuda"
        )
        model.set_params(early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        std = np.std(y_pred)

        importances = model.feature_importances_
        top_peak_indices = np.argsort(importances)[-top:]
        top_original_names = [same_chrom_peaks_names[int_idx] for int_idx in top_peak_indices]
        top_numerical_indices = [peak_indices[string_index] for string_index in top_original_names]
        connections[top_numerical_indices] = 1

    return connections, mse, std


def create_connection_matrix(rna_adata, atac_adata, distance=1e6, top=5, method='both', save_path=None):
    peak_indices = prepare_matrix(atac_adata)
    gene_matrix = lil_matrix((rna_adata.shape[1], atac_adata.shape[1]), dtype=int)
    mse_list, std_list = [], []

    for gene_idx, gene_name in tqdm(enumerate(rna_adata.var_names), total=len(rna_adata.var_names)):
        connections, gene_mse, gene_std = process_gene_to_matrix(
            gene_idx, rna_adata, atac_adata, rna_adata.var.loc[gene_name, 'chrom'],
            rna_adata.var.loc[gene_name, 'chromStart'],
            distance, top, method, peak_indices
        )
        gene_matrix[gene_idx, :] = connections
        mse_list.append(gene_mse)
        std_list.append(gene_std)

    gene_matrix_csr = gene_matrix.tocsr()
    peak_connections = np.array(gene_matrix_csr.sum(axis=0)).flatten()
    connected_peak_mask = peak_connections > 0

    # Create filtered ATAC data
    filtered_atac = atac_adata[:, connected_peak_mask].copy()
    print(filtered_atac.shape, atac_adata.shape)
    if save_path:
        filtered_atac.write(save_path)

    filtered_gene_matrix = gene_matrix_csr[:, connected_peak_mask]

    return filtered_gene_matrix


from pybedtools import BedTool
import pandas as pd
from tqdm import tqdm


def match_cistarget(rna_adata, peak_adata, score_df, motif2tf_df, threshold=3):
    """
    Match cisTarget regions with peaks using pybedtools for efficient overlap calculation.

    Parameters:
    -----------
    rna_adata : AnnData
        RNA expression data
    peak_adata : AnnData
        Peak data with genomic coordinates
    score_df : pandas.DataFrame
        Score matrix with regions as columns
    motif2tf_df : pandas.DataFrame
        Motif to transcription factor mapping
    threshold : float, default=3
        Score threshold for significance

    Returns:
    --------
    tuple
        (filtered_score_df, overlapping_peak_indices, region_to_peaks)
    """
    # Create BED format DataFrame for peaks
    peak_bed_df = peak_adata.var[['chrom', 'chromStart', 'chromEnd']].copy()
    peak_bed_df['chrom'] = 'chr' + peak_bed_df['chrom'].astype(str)
    peak_bed_df['name'] = peak_bed_df.index

    peak_bedtool = BedTool.from_dataframe(peak_bed_df)

    print(peak_bedtool.head())
    significant_regions = []
    region_df_data = []

    for region in tqdm(score_df.columns, desc="Processing regions"):
        try:
            region_chr, region_loc = region.split(':')
            region_start, region_end = map(int, region_loc.split('-'))

            # Check if any score exceeds threshold for this region
            if any(score_df[region] > threshold):
                significant_regions.append(region)
                region_df_data.append({
                    'chrom': region_chr,
                    'start': region_start,
                    'end': region_end,
                    'name': region
                })
        except ValueError:
            continue

    if not region_df_data:
        return pd.DataFrame(), set(), {}

    region_df = pd.DataFrame(region_df_data)
    region_bedtool = BedTool.from_dataframe(region_df)

    print(region_bedtool.head(), peak_bedtool.head())
    # Find overlaps
    overlaps = region_bedtool.intersect(peak_bedtool, wa=True, wb=True).to_dataframe(
        names=['region_chrom', 'region_start', 'region_end', 'region_name',
               'peak_chrom', 'peak_start', 'peak_end', 'peak_name']
    )
    print(overlaps.head())

    # Process overlaps into return format
    region_to_peaks = {}
    overlapping_peak_indices = set()

    for _, row in overlaps.iterrows():
        region = row['region_name']
        peak_name = row['peak_name']

        if region not in region_to_peaks:
            region_to_peaks[region] = []
        region_to_peaks[region].append(peak_name)
        overlapping_peak_indices.add(peak_name)

    region_to_peaks = {k: set(v) for k, v in region_to_peaks.items()}
    filtered_score_df = score_df[list(region_to_peaks.keys())]

    return filtered_score_df, overlapping_peak_indices, region_to_peaks


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm


def create_cistarget_matrix(rna_adata, peak_adata, region_to_peak, score_df, motif2tf_df, threshold=3, percentile=0):
    """
    Create a gene-peak matrix based on significant motifs and their associated genes.

    Parameters:
    -----------
    rna_adata : AnnData
        RNA expression data
    peak_adata : AnnData
        Peak data
    region_to_peak : dict
        Mapping of regions to peaks (where values are sets of peaks)
    score_df : pandas.DataFrame
        Score matrix with motif scores
    motif2tf_df : pandas.DataFrame
        Motif to transcription factor mapping
    threshold : float, default=3
        Score threshold for significance
    percentile : float, default=0
        Optional percentile threshold

    Returns:
    --------
    scipy.sparse.csr_matrix
        Gene-peak matrix with significant interactions
    """
    # Initialize sparse matrix and mapping dictionaries
    n_genes, n_peaks = rna_adata.shape[1], peak_adata.shape[1]
    gene_peak_matrix = lil_matrix((n_genes, n_peaks), dtype=np.int8)

    # Create efficient lookups
    peak_indices = {peak: i for i, peak in enumerate(peak_adata.var.index)}
    gene_indices = {gene: i for i, gene in enumerate(rna_adata.var.index)}
    motif_genes = dict(zip(motif2tf_df['#motif_id'], motif2tf_df['gene_name']))

    # Filter score_df to relevant regions
    relevant_regions = score_df.columns.intersection(region_to_peak.keys())
    filtered_df = score_df[relevant_regions]

    # Calculate score percentile if needed
    score_threshold = np.percentile(filtered_df.values.flatten(), percentile) if percentile else threshold

    # Process each region and its scores
    for region, scores in tqdm(filtered_df.items(), desc='Processing regions'):
        # Process each peak in the set for this region
        for peak in region_to_peak[region]:
            peak_idx = peak_indices.get(str(peak))
            if peak_idx is None:
                continue

            # Get significant motifs
            mask = scores > (score_threshold if percentile else threshold)
            if percentile:
                mask &= (scores > threshold)

            # Process significant motifs
            for motif_id in scores[mask].index:
                motif = score_df.loc[motif_id, 'motifs']
                gene = motif_genes.get(motif)
                if gene and (gene_idx := gene_indices.get(gene)) is not None:
                    gene_peak_matrix[gene_idx, peak_idx] = 1

    return gene_peak_matrix.tocsr()


def combine_matrices(rna_adata, peak_adata, A, B, path):
    print(A.shape, B.shape)
    A_plus_B = (A + B > 0).astype(int)
    gene_num = rna_adata.shape[1]
    peak_num = peak_adata.shape[1]

    zero_block_gene = csr_matrix((gene_num, gene_num))
    zero_block_peak = csr_matrix((peak_num, peak_num))

    larger_matrix = bmat([[zero_block_gene, A_plus_B], [A_plus_B.T, zero_block_peak]], format='csr')

    with open(path, 'wb') as f:
        pickle.dump(larger_matrix, f)

    return larger_matrix.count_nonzero()


def run_preprocessing(args):
    distance_str = str(args.distance)
    data_folder = './data/'

    if args.use_hvg:
        paths = {
            'original_rna_path': f'{data_folder}{args.dataset}/RNA_count.h5ad',
            'original_atac_path': f'{data_folder}{args.dataset}/ATAC_count.h5ad',
            'rna_path': f'{data_folder}{args.dataset}/hvg_only_rna_count.h5ad',
            'atac_path': f'{data_folder}{args.dataset}/hvg_only_atac_count.h5ad',
            'tg_re_matrix_path': f'{data_folder}/{args.dataset}/{args.if_both}hvg_tg_re_{args.method}_dist_{distance_str}_matrix.pkl',
            'tg_re_atac_path': f'{data_folder}/{args.dataset}/{args.if_both}hvg_tg_re_{args.method}_dist_{distance_str}_ATAC.h5ad',
            'tf_re_matrix_path': f'{data_folder}/{args.dataset}/{args.if_both}hvg_tf_re_{args.method}_dist_{distance_str}_threshold_{args.threshold}_matrix_.pkl',
            'edge_path': f'{data_folder}/{args.dataset}/{args.if_both}hvg_{args.method}_{distance_str}_threshold{args.threshold}_GRN.pkl'
        }
    else:
        paths = {
            'original_rna_path': f'{data_folder}{args.dataset}/RNA_count.h5ad',
            'original_atac_path': f'{data_folder}{args.dataset}/ATAC_count.h5ad',
            'rna_path': f'{data_folder}{args.dataset}/coding_rna_count.h5ad',
            'atac_path': f'{data_folder}{args.dataset}/{args.dataset}_coding_tg_re_{args.method}_dist_{distance_str}_ATAC.h5ad',
            'tg_re_matrix_path': f'{data_folder}{args.dataset}/{args.dataset}_coding_tg_re_{args.method}_dist_{distance_str}_matrix.pkl',
            'tf_re_matrix_path': f'{data_folder}/{args.dataset}/TF_RE_coding_matrix_threshold_{args.threshold}.pkl',
            'edge_path': f'{data_folder}{args.dataset}/{args.tissue}_coding_{distance_str}_threshold{args.threshold}_GRN.pkl'
        }

    if not os.path.exists(paths['edge_path']):
        # Step 1: Process TG-RE relationships if needed
        if not os.path.exists(paths['tg_re_matrix_path']):
            rna_adata = ad.read_h5ad(paths['rna_path'])
            atac_adata = ad.read_h5ad(paths['atac_path'])
            tg_re_matrix = create_connection_matrix(
                rna_adata, atac_adata,
                distance=args.distance,
                top=5,
                method=args.method,
                save_path=paths['tg_re_atac_path']
            )
            with open(paths['tg_re_matrix_path'], 'wb') as f:
                pickle.dump(tg_re_matrix, f)

        # Step 2: Process TF-RE relationships if needed
        if not os.path.exists(paths['tf_re_matrix_path']):
            rna_adata = ad.read_h5ad(paths['rna_path'])
            atac_adata = ad.read_h5ad(paths['tg_re_atac_path'])
            score_df = feather.read_feather(args.score_path)
            motif2tf_df = pd.read_csv(args.motif2tf_path, delimiter='\t')
            motif2tf_df = motif2tf_df.loc[:, ['#motif_id', 'motif_name', 'gene_name']]

            filtered_score_df, overlapping_peaks, region_to_peaks_df = match_cistarget(
                rna_adata, atac_adata, score_df, motif2tf_df, threshold=args.threshold
            )

            tf_re_matrix = create_cistarget_matrix(
                rna_adata, atac_adata, region_to_peaks_df, score_df, motif2tf_df,
                threshold=args.threshold, percentile=args.percentile
            )
            with open(paths['tf_re_matrix_path'], 'wb') as f:
                pickle.dump(tf_re_matrix, f)

        # Step 3: Combine matrices
        with open(paths['tg_re_matrix_path'], 'rb') as f:
            tg_re_matrix = pickle.load(f)
        with open(paths['tf_re_matrix_path'], 'rb') as f:
            tf_re_matrix = pickle.load(f)

        # rna_adata = ad.read_h5ad(paths['rna_path'])
        # atac_adata = ad.read_h5ad(paths['atac_path'])
        num_edges = combine_matrices(rna_adata, atac_adata, tf_re_matrix, tg_re_matrix, paths['edge_path'])
        print(f"Total GRN edges: {num_edges}")

    return paths


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess GRN data')
    parser.add_argument('--dataset', type=str, default='PBMC', help='Dataset name')
    parser.add_argument('--tissue', type=str, default=None, help='Tissue type')
    parser.add_argument('--threshold', type=float, default=3, help='Threshold for filtering')
    parser.add_argument('--use_hvg', action='store_true', help='Use HVG genes')
    parser.add_argument('--if_both', type=str, default='', help='Both flag prefix')
    parser.add_argument('--method', type=str, default='nearby',
                       choices=['nearby', 'gbm', 'both', 'correlation'],
                       help='Method for connection matrix')
    parser.add_argument('--distance', type=float, default=1e6,
                       help='Distance threshold for nearby peaks')
    parser.add_argument('--score_path', type=str,
                        default='../cistarget/hg38_screen_v10_clust.regions_vs_motifs.scores.feather',
                        help='Path to score feather file')
    parser.add_argument('--motif2tf_path', type=str,
                        default='../cistarget/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',
                        help='Path to motif2tf table')
    parser.add_argument('--percentile', type=float, default=0,
                        help='Percentile threshold for filtering')
    args = parser.parse_args()

    if args.tissue is None:
        args.tissue = args.dataset

    return args


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(args)