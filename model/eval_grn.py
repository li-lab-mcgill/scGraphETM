# Import statements
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import pybedtools
import pickle
import random
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from scipy.stats import pearsonr

from scModel import (
    load_and_process_data,
    create_data_loader,
    initialize_model,
    get_model_save_path
)
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import random
import pybedtools
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from typing import Dict, List, Tuple, Optional
from scDataLoader import ScDataLoader, set_seed


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


class ChIPSeqAnalyzer:
    def __init__(self, rna_adata, atac_adata, rna_adata_gnn, atac_adata_gnn, args):
        """
        Initialize the ChIP-seq analyzer with filtered data.

        Args:
            rna_adata: AnnData object containing RNA data
            atac_adata: AnnData object containing ATAC data
            rna_adata_gnn: AnnData object containing RNA data for GNN
            atac_adata_gnn: AnnData object containing ATAC data for GNN
            args: Arguments containing configuration parameters
        """
        self.config = {
        'date': args.date,
        'data_folder': args.data_folder,
        'dataset': args.dataset,
        'tissue': args.tissue,
        'edge_path': None,
        'emb_size': args.emb_size,
        'num_of_topic': args.num_of_topic,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'device': args.device,
        'model_name': args.model_name
    }
        self.args = args
        self.device = args.device
        self.mlp = None


        # Filter cells by cell type first
        self.cell_indices = self.get_cell_type_indices(rna_adata, args.cell_type)
        if len(self.cell_indices) == 0:
            raise ValueError(f"No cells found for type {args.cell_type}")

        # Filter data to only include matched cells
        self.rna_adata = rna_adata[self.cell_indices]
        self.atac_adata = atac_adata[self.cell_indices]
        self.rna_adata_gnn = rna_adata_gnn[self.cell_indices]
        self.atac_adata_gnn = atac_adata_gnn[self.cell_indices]

        # Create data loader with filtered data
        self.data_loader = self.create_filtered_data_loader()

        # Initialize and load model
        self.model = self.initialize_model()
        self.load_model()

        # Store processed ChIP-seq data
        self.chipseq_data = None
        self.tf_regions = {}

    def get_cell_type_indices(self, adata, cell_type: str) -> np.ndarray:
        """
        Get indices of cells matching the specified cell type (case-insensitive).

        Args:
            adata: AnnData object containing cell type information
            cell_type: Target cell type string to match

        Returns:
            np.ndarray: Array of indices for matching cells
        """
        cell_type_lower = cell_type.lower()
        matching_mask = adata.obs['cell_type'].str.lower().str.contains(cell_type_lower)
        matched_types = adata.obs['cell_type'][matching_mask].unique()
        print(f"\nMatched cell types for '{cell_type}':")
        for t in matched_types:
            print(f"  - {t}")

        indices = np.where(matching_mask)[0]
        print(f"Total cells selected: {len(indices)}")
        return indices

    def create_filtered_data_loader(self):
        """Create a data loader with only the filtered cells using ScDataLoader."""
        return ScDataLoader(
            rna_adata=self.rna_adata,
            atac_adata=self.atac_adata,
            num_of_gene=len(self.rna_adata.var_names),
            num_of_peak=len(self.atac_adata.var_names),
            rna_adata_gnn=self.rna_adata_gnn,
            atac_adata_gnn=self.atac_adata_gnn,
            emb_size=self.args.emb_size,
            batch_size=self.args.batch_size,
            edge_path=None,
            feature_type='',
            # feature_type='node2vec',
            train_ratio=0.99,  # Use all data for cell type specific analysis
            cell_type="",
            num_workers=1,
            pin_memory=True
        )

    def initialize_model(self):
        """Initialize the model with filtered data."""

        node2vec = torch.load(f'./node2vec/{self.args.dataset}_{self.args.model_name}_{self.args.emb_size}_400.pt')
        return initialize_model(
            self.config,
            self.rna_adata,
            self.atac_adata,
            self.rna_adata_gnn,
            self.atac_adata_gnn,
            self.args,
            node2vec
        )

    def load_model(self):
        """Load the saved model weights."""
        model_path = get_model_save_path(self.args.tissue, self.args.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}")

        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialize_mlp(self, input_size):
        """Initialize MLP model for peak scoring."""
        hidden_size = 128
        output_size = 1
        self.mlp = MLP(input_size, hidden_size, output_size).to(self.device)

    def get_embeddings(self) -> torch.Tensor:
        """Get embeddings using the data loader's batching."""
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in self.data_loader.train_loader:
                # Get embeddings for the batch
                _, _, _, emb = self.model(batch, self.data_loader.train_loader)
                all_embeddings.append(emb)

        # Concatenate all embeddings
        return torch.stack(all_embeddings, dim=0).mean(dim=0)

    def load_chipseq_data(self, file_path: str) -> List[str]:
        """
        Read and process ChIP-seq file for all TFs.

        Args:
            file_path: Path to ChIP-seq file

        Returns:
            List of available TFs in the data
        """
        try:
            df = pd.read_csv(file_path, sep='\t', header=None,
                             names=['chrom', 'start', 'end', 'TF'])

            self.chipseq_data = df

            for tf in df['TF'].unique():
                tf_df = df[df['TF'] == tf]
                self.tf_regions[tf] = tf_df[['chrom', 'start', 'end']]
                print(f"Processed {len(tf_df)} regions for {tf}")

            return list(self.tf_regions.keys())

        except Exception as e:
            raise Exception(f"Error reading {file_path}: {str(e)}")

    def find_overlapping_peaks(self, tf_name: str) -> List[int]:
        """Find ATAC peaks that overlap with ChIP-seq regions for specific TF."""
        if tf_name not in self.tf_regions:
            raise ValueError(f"No ChIP-seq regions found for {tf_name}")

        chipseq_regions = self.tf_regions[tf_name]

        atac_peaks = []
        for idx, (chrom, start, end) in enumerate(zip(
                self.atac_adata.var['chrom'],
                self.atac_adata.var['chromStart'],
                self.atac_adata.var['chromEnd']
        )):
            atac_peaks.append([chrom, start, end, str(idx)])

        atac_bed = pybedtools.BedTool(atac_peaks)
        chipseq_bed = pybedtools.BedTool.from_dataframe(chipseq_regions)

        overlaps = atac_bed.intersect(chipseq_bed, wa=True)
        overlap_indices = [int(overlap[3]) for overlap in overlaps]

        return list(set(overlap_indices))

    def analyze_tf(self, tf_name: str, method: str = 'dot') -> Dict:
        """
        Analyze a single TF using pre-loaded ChIP-seq data, filtering for regions on the TF's chromosome.

        Args:
            tf_name: Name of transcription factor to analyze
            method: Scoring method ('dot', 'mlp', or 'pearson')

        Returns:
            Dictionary containing analysis results
        """
        if tf_name not in self.rna_adata.var_names:
            raise ValueError(f"TF {tf_name} not found in training data")

        if self.chipseq_data is None:
            raise ValueError("ChIP-seq data not loaded. Call load_chipseq_data first.")

        print(f"\nProcessing {tf_name}")

        # Get TF chromosome from RNA data
        tf_idx = self.rna_adata.var_names.get_loc(tf_name)
        tf_chromosome = self.rna_adata.var['chrom'][tf_idx]
        tf_chromosome = 'chr1'
        print(f"TF {tf_name} is located on chromosome {tf_chromosome}")

        # Get embeddings using data loader
        emb = self.get_embeddings()
        tf_emb = emb[tf_idx]

        # Find positive peaks only on TF's chromosome
        positive_indices = []
        for idx, (peak_chrom, start, end) in enumerate(zip(
                self.atac_adata.var['chrom'],
                self.atac_adata.var['chromStart'],
                self.atac_adata.var['chromEnd']
        )):
            if peak_chrom == tf_chromosome:
                positive_indices.append([peak_chrom, start, end, str(idx)])

        if not positive_indices:
            print(f"No peaks found on chromosome {tf_chromosome}")
            return None

        # Find overlaps with ChIP-seq data
        atac_bed = pybedtools.BedTool(positive_indices)
        tf_regions = self.tf_regions[tf_name]
        tf_chrom_regions = tf_regions[tf_regions['chrom'] == tf_chromosome]
        chipseq_bed = pybedtools.BedTool.from_dataframe(tf_chrom_regions)

        overlaps = atac_bed.intersect(chipseq_bed, wa=True)
        positive_indices = [int(overlap[3]) for overlap in overlaps]
        positive_indices = list(set(positive_indices))

        if not positive_indices:
            print(f"No overlapping peaks found for {tf_name} on chromosome {tf_chromosome}")
            return None

        print(f"Found {len(positive_indices)} peaks with ChIP-seq evidence on chromosome {tf_chromosome}")

        # Sample negative peaks only from TF's chromosome
        negative_indices = []
        for idx, peak_chrom in enumerate(self.atac_adata.var['chrom']):
            if peak_chrom == tf_chromosome and idx not in positive_indices:
                negative_indices.append(idx)

        if len(negative_indices) >= len(positive_indices):
            random.seed(42)
            negative_indices_selected = random.sample(negative_indices, len(positive_indices))
            random_indices = random.sample(negative_indices, len(positive_indices) * 2)

        # Calculate scores
        all_peak_emb = emb[len(self.rna_adata.var_names):]
        selected_indices = positive_indices + negative_indices_selected
        selected_peak_emb = all_peak_emb[selected_indices]

        if method == 'dot':
            scores = torch.mv(selected_peak_emb, tf_emb)
            scaled_scores = (scores - scores.mean()) / scores.std()
            scores = torch.sigmoid(scaled_scores).cpu()

        elif method == 'mlp':
            if self.mlp is None:
                self.initialize_mlp(2 * tf_emb.size(0))
            tf_expanded = tf_emb.unsqueeze(0).repeat(len(selected_peak_emb), 1)
            mlp_input = torch.cat([selected_peak_emb, tf_expanded], dim=1)
            scores = self.mlp(mlp_input).squeeze().cpu().numpy()
        elif method == 'pearson':
            scores = np.array([
                pearsonr(peak_emb.cpu().numpy(), tf_emb.cpu().numpy())[0]
                for peak_emb in selected_peak_emb
            ])
            scores = np.nan_to_num(scores)

        # Calculate metrics
        labels = np.ones(len(selected_indices))
        labels[:len(positive_indices)] = 0

        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)

        # Plot and save AUPRC curve
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label=f'AUPRC = {auprc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {tf_name} on {tf_chromosome}')
        plt.legend(loc='lower left')
        plt.ylim(0, 1)

        # Save plot
        plot_filename = f"./result/PBMC/Jan23/grn_eval/{tf_name}_{tf_chromosome}_{method}_auprc.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved AUPRC plot to {plot_filename}")

        results = {
            'scores': scores,
            'labels': labels,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'tf_chromosome': tf_chromosome,
            'positive_indices': positive_indices,
            'positive_regions': {
                'chrom': self.atac_adata.var['chrom'][positive_indices],
                'start': self.atac_adata.var['chromStart'][positive_indices],
                'end': self.atac_adata.var['chromEnd'][positive_indices]
            }
        }

        print(f"\nSummary for {tf_name}:")
        print(f"TF chromosome: {tf_chromosome}")
        print(f"Number of positive peaks: {len(positive_indices)}")
        print(f"Total number of peaks analyzed: {len(selected_indices)}")
        print(f"AUPRC: {auprc:.4f}")

        # Save detailed results to CSV
        output_data = {
            'chrom': self.atac_adata.var['chrom'][selected_indices],
            'start': self.atac_adata.var['chromStart'][selected_indices],
            'end': self.atac_adata.var['chromEnd'][selected_indices],
            'score': np.concatenate([scores[len(positive_indices):], scores[:len(positive_indices)]]),
            'is_positive': np.concatenate([labels[len(positive_indices):], labels[:len(positive_indices)]]),
        }

        df = pd.DataFrame(output_data)
        csv_filename = f"./result/PBMC/Jan22/grn_eval/{tf_name}_{tf_chromosome}_{method}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved detailed results to {csv_filename}")

        return results

def parse_args():
    parser = argparse.ArgumentParser(description='ChIP-seq analysis and model evaluation')

    # Model arguments
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='PBMC')
    parser.add_argument('--tissue', type=str, default=None)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--num_of_topic', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='')

    # Additional required scModel arguments
    parser.add_argument('--use_gnn', action='store_true')
    parser.add_argument('--use_graph_recon', action='store_true')
    parser.add_argument('--graph_recon_weight', type=float, default=1.0)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--shared_decoder', action='store_true')
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--distance', type=float, default=1e6)
    parser.add_argument('--preprocess_method', type=str, default='nearby')
    parser.add_argument('--if_both', type=str, default='')
    parser.add_argument('--use_xtrimo', action='store_true')
    parser.add_argument('--latent_size', type=int, default=100)

    # ChIP-seq specific arguments
    parser.add_argument('--tf_name', type=str)
    parser.add_argument('--chipseq_file', type=str, required=True)
    parser.add_argument('--cell_type', type=str, required=True)
    parser.add_argument('--scoring_method', type=str, default='dot', choices=['dot', 'mlp', 'pearson'])

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device('cpu')

    if args.tissue is None:
        args.tissue = args.dataset

    return args


def main():
    args = parse_args()

    # Set up paths
    paths = {
        'original_rna_path': f'./data/{args.dataset}/RNA_count.h5ad',
        'original_atac_path': f'./data/{args.dataset}/ATAC_count.h5ad',
        'rna_path': f'./data/{args.dataset}/hvg_only_rna_count.h5ad',
        'atac_path': f'./data/{args.dataset}/hvg_tg_re_{args.preprocess_method}_dist_{str(args.distance)}_ATAC.h5ad',
        'edge_path': f'./data//{args.dataset}/{args.if_both}hvg_{args.preprocess_method}_{str(args.distance)}_threshold{args.threshold}_GRN.pkl'
    }

    # Load and process data
    print("Loading and processing data...")
    rna_adata, atac_adata, rna_adata_gnn, atac_adata_gnn = load_and_process_data(paths)
    print(f"Data loaded - RNA shape: {rna_adata.shape}, ATAC shape: {atac_adata.shape}")

    # Create results directory
    results_dir = os.path.join("./result", args.tissue, args.date, "chipseq_analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("\nInitializing ChIP-seq analyzer...")
    analyzer = ChIPSeqAnalyzer(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        rna_adata_gnn=rna_adata_gnn,
        atac_adata_gnn=atac_adata_gnn,
        args=args
    )

    # Load ChIP-seq data once
    print("\nLoading ChIP-seq data...")
    available_tfs = analyzer.load_chipseq_data(args.chipseq_file)
    print(f"Found {len(available_tfs)} TFs in ChIP-seq data")

    # Determine which TFs to analyze
    if args.tf_name is not None:
        if args.tf_name not in available_tfs:
            raise ValueError(f"Specified TF {args.tf_name} not found in ChIP-seq data")
        tfs_to_analyze = [args.tf_name]
    else:
        tfs_to_analyze = available_tfs

    # Analyze each TF
    all_results = {}
    for tf in tfs_to_analyze:
        if tf in rna_adata.var_names:
            try:
                results = analyzer.analyze_tf(
                    tf_name=tf,
                    method=args.scoring_method
                )
                if results is not None:
                    all_results[tf] = results
            except Exception as e:
                print(f"Error analyzing TF {tf}: {str(e)}")
                continue
        else:
            print(f"Skipping {tf} - not found in training data")

    # Save results
    if all_results:
        results_file = os.path.join(
            results_dir,
            f"{'all_tfs' if args.tf_name is None else args.tf_name}_{args.cell_type}_results.pickle"
        )
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved to {results_file}")

        # Print summary
        print("\nAnalysis Summary:")
        for tf, results in all_results.items():
            print(f"\n{tf}:")
            print(f"  AUPRC: {results['auprc']:.4f}")
            print(f"  Number of positive peaks: {len(results['positive_indices'])}")
    else:
        print("\nNo results to save - no successful TF analyses.")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    main()