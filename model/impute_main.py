import os
import torch
import argparse
from typing import Dict, Any, Optional
import datetime
import anndata as ad

from model import ScModel
from scDataLoader import ScDataLoader, set_seed
from imputation import ImputationTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Single-cell model imputation configuration')

    # Data configuration
    parser.add_argument('--date', type=str, default=None, required=True,
                        help='Date identifier for the run')
    parser.add_argument('--data_folder', type=str, default='./data/',
                        help='Path to data folder')
    parser.add_argument('--dataset', type=str, choices=['PBMC', 'BMMC', 'cerebral_cortex'],
                        default='PBMC', help='Dataset to use')
    parser.add_argument('--tissue', type=str, default=None,
                        help='Tissue type (defaults to dataset if not specified)')

    # Model configuration
    parser.add_argument('--emb_size', type=int, default=512,
                        help='Embedding size')
    parser.add_argument('--num_of_topic', type=int, default=100,
                        help='Number of topics')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (-1 for CPU)')
    parser.add_argument('--model_name', type=str, default='',
                        help='Custom name to be added to the model file name')

    # Architecture options
    parser.add_argument('--use_gnn', action='store_true',
                        help='Enable GNN for feature transformation')
    parser.add_argument('--use_xtrimo', action='store_true',
                        help='Use GeneEmbedding and AtacEmbedding')
    parser.add_argument('--shared_decoder', action='store_true',
                        help='Share decoder parameters between modalities')

    # Imputation specific
    parser.add_argument('--imputation_weight', type=float, default=1.0,
                        help='Weight for imputation loss')
    parser.add_argument('--imputation_epochs', type=int, default=20,
                        help='Number of epochs for imputation training')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Path to save imputation plots')

    # Data processing
    parser.add_argument('--threshold', type=float, default=3,
                        help='Threshold for cistarget filtering')
    parser.add_argument('--distance', type=float, default=1e6,
                        help='Distance threshold in base pairs')
    parser.add_argument('--use_hvg', action='store_false',
                        help='Use HVG genes')
    parser.add_argument('--if_both', type=str, default='',
                        help='Both flag prefix')
    parser.add_argument('--preprocess_method', type=str, default='nearby',
                        choices=['nearby', 'gbm', 'both'],
                        help='Method for connection matrix')

    args = parser.parse_args()

    # Set tissue to dataset if not specified
    if args.tissue is None:
        args.tissue = args.dataset

    # Set device based on GPU argument
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device('cpu')

    return args


def get_model_save_path(tissue: str, model_name: str = '') -> str:
    """Generate the save path for the model with custom naming."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    base_name = f"best_model_{current_date}"
    if model_name:
        base_name += f"_{model_name}"

    save_dir = f'./weights/{tissue}'
    os.makedirs(save_dir, exist_ok=True)
    return f'{save_dir}/{base_name}.pth'


def load_and_process_data(paths: Dict[str, str]) -> Tuple[ad.AnnData, ad.AnnData,
ad.AnnData, ad.AnnData]:
    """Load and process RNA and ATAC data."""
    # Load original data
    rna_adata = ad.read(paths['original_rna_path'])
    atac_adata = ad.read(paths['original_atac_path'])

    # Load GNN data
    rna_adata_gnn = ad.read_h5ad(paths['rna_path'])
    atac_adata_gnn = ad.read_h5ad(paths['atac_path'])

    # Process common features
    common_genes = rna_adata.var_names.intersection(rna_adata_gnn.var_names)
    common_peaks = atac_adata.var_names.intersection(atac_adata_gnn.var_names)

    # Filter data
    rna_adata_gnn = rna_adata[:, common_genes].copy()
    rna_adata = rna_adata_gnn.copy()
    atac_adata_gnn = atac_adata[:, common_peaks].copy()
    atac_adata = atac_adata_gnn.copy()

    return rna_adata, atac_adata, rna_adata_gnn, atac_adata_gnn


def create_data_loader(config: Dict[str, Any], rna_adata: ad.AnnData,
                       atac_adata: ad.AnnData, rna_adata_gnn: ad.AnnData,
                       atac_adata_gnn: ad.AnnData) -> ScDataLoader:
    """Create data loader with the specified configuration."""
    return ScDataLoader(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        num_of_gene=rna_adata.n_vars,
        num_of_peak=atac_adata.n_vars,
        rna_adata_gnn=rna_adata_gnn,
        atac_adata_gnn=atac_adata_gnn,
        emb_size=config['emb_size'],
        batch_size=config['batch_size'],
        num_workers=1,
        pin_memory=True,
        edge_path=config['edge_path'],
        feature_type='',
        cell_type='',
        fm_save_path=config['model_name'],
    )


def initialize_model(config: Dict[str, Any], rna_adata: ad.AnnData,
                     atac_adata: ad.AnnData, rna_adata_gnn: ad.AnnData,
                     atac_adata_gnn: ad.AnnData, args: argparse.Namespace,
                     node2vec) -> ScModel:
    """Initialize the ScModel with proper configuration."""
    # Create result directories
    umap_dir = f"./result/{config['tissue']}/{config['date']}/figure/map"
    tsne_dir = f"./result/{config['tissue']}/{config['date']}/figure/tsne"
    os.makedirs(umap_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)

    return ScModel(
        num_of_gene=rna_adata.n_vars,
        num_of_peak=atac_adata.n_vars,
        num_of_gene_gnn=rna_adata_gnn.n_vars,
        num_of_peak_gnn=atac_adata_gnn.n_vars,
        emb_size=config['emb_size'],
        num_of_topic=config['num_of_topic'],
        device=config['device'],
        batch_size=config['batch_size'],
        result_save_path=(umap_dir, tsne_dir),
        gnn_conv='GCN' if args.use_gnn else None,
        lr=config['lr'],
        use_graph_recon=False,  # Not needed for imputation
        graph_recon_weight=1.0,
        pos_weight=1.0,
        latent_size=100,
        node2vec=node2vec,
        use_gnn=args.use_gnn,
        use_xtrimo=args.use_xtrimo,
        shared_decoder=args.shared_decoder,
    )


def main():
    """Main function to run model imputation."""
    # Parse arguments and set random seed
    args = parse_args()
    set_seed(42)

    # Setup paths and configuration
    paths = {
        'original_rna_path': f'./data/{args.dataset}/RNA_count.h5ad',
        'original_atac_path': f'./data/{args.dataset}/ATAC_count.h5ad',
        'rna_path': f'./data/{args.dataset}/hvg_only_rna_count.h5ad',
        'atac_path': f'./data/{args.dataset}/hvg_tg_re_{args.preprocess_method}_dist_{str(args.distance)}_ATAC.h5ad',
        'edge_path': f'./data//{args.dataset}/{args.if_both}hvg_{args.preprocess_method}_{str(args.distance)}_threshold{args.threshold}_GRN.pkl'
    }

    config = {
        'date': args.date,
        'data_folder': args.data_folder,
        'dataset': args.dataset,
        'tissue': args.tissue,
        'edge_path': None,  # No need for edge path in imputation
        'emb_size': args.emb_size,
        'num_of_topic': args.num_of_topic,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'device': args.device,
        'model_name': args.model_name
    }

    # Load and process data
    paths['best_model_path'] = get_model_save_path(args.tissue, args.model_name)
    rna_adata, atac_adata, rna_adata_gnn, atac_adata_gnn = load_and_process_data(paths)

    print(f"Data shapes: RNA: {rna_adata.shape}, ATAC: {atac_adata.shape}, "
          f"RNA GNN: {rna_adata_gnn.shape}, ATAC GNN: {atac_adata_gnn.shape}")

    # Initialize data loader and model
    data_loader = create_data_loader(config, rna_adata, atac_adata,
                                     rna_adata_gnn, atac_adata_gnn)

    # Initialize model
    if os.path.exists(paths['best_model_path']):
        print(f"Loading model from {paths['best_model_path']}")
        node2vec = torch.load(f'./node2vec/{args.dataset}_{args.model_name}_{args.emb_size}_400.pt')
        model = initialize_model(config, rna_adata, atac_adata, rna_adata_gnn,
                                 atac_adata_gnn, args, node2vec)
        model.load_best_model(paths['best_model_path'])
    else:
        raise FileNotFoundError(f"No saved model found at {paths['best_model_path']}. "
                                "Please train the model first.")

    # Create directories for saving results
    if args.plot_path:
        os.makedirs(args.plot_path, exist_ok=True)

    # Initialize imputation trainer and run imputation
    imputation_trainer = ImputationTrainer(model, args.device)

    print("\nStarting imputation training...")
    imputation_trainer.train_with_imputation(
        data_loader,
        epochs=args.imputation_epochs,
        imputation_weight=args.imputation_weight
    )

    # Run final imputation evaluation
    print("\nRunning final imputation evaluation...")
    with torch.no_grad():
        correlations, imputed_counts = imputation_trainer.impute(data_loader, args.plot_path)

    # Save final model
    base_path = paths['best_model_path'].rsplit('.', 1)[0]
    model.save(f"{base_path}_impute.pth")

    # Save correlation results
    if args.plot_path:
        results_file = os.path.join(args.plot_path, 'imputation_results.txt')
        with open(results_file, 'w') as f:
            f.write("=== Final Imputation Results ===\n")
            for modality in ['RNA', 'ATAC']:
                f.write(f"\n{modality} Imputation:\n")
                f.write(f"Pearson correlation: {correlations[modality]['pearson']:.4f}\n")
                f.write(f"Spearman correlation: {correlations[modality]['spearman']:.4f}\n")

        print(f"\nResults saved to {results_file}")

    print("Imputation completed successfully!")


if __name__ == '__main__':
    main()