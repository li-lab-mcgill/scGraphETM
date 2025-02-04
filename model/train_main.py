import os
import torch
import argparse
import datetime
from typing import Dict, Any
import anndata as ad

from model import ScModel
from scDataLoader import ScDataLoader, set_seed
from preprocess.preprocess import run_preprocessing
from utils import evaluate_clustering, save_plots, plot_expression_heatmaps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Single-cell model training configuration')

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
    parser.add_argument('--latent_size', type=int, default=100,
                        help='Size of latent tokens for attention mechanism')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (-1 for CPU)')

    # Architecture options
    parser.add_argument('--use_graph_recon', action='store_true',
                        help='Enable graph reconstruction loss')
    parser.add_argument('--graph_recon_weight', type=float, default=1.0,
                        help='Weight for graph reconstruction loss')
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='Positive weight for graph reconstruction loss')
    parser.add_argument('--use_gnn', action='store_true',
                        help='Enable GNN for feature transformation')
    parser.add_argument('--shared_decoder', action='store_true',
                        help='Share decoder parameters between modalities')
    parser.add_argument('--use_xtrimo', action='store_true',
                        help='Use GeneEmbedding and AtacEmbedding instead of average expression')

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

    # Output configuration
    parser.add_argument('--model_name', type=str, default='',
                        help='Custom name to be added to the model file name')

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
        use_graph_recon=args.use_graph_recon,
        graph_recon_weight=args.graph_recon_weight,
        pos_weight=args.pos_weight,
        latent_size=args.latent_size,
        node2vec=node2vec,
        use_gnn=args.use_gnn,
        use_xtrimo=args.use_xtrimo,
        shared_decoder=args.shared_decoder,
    )


def train_model(model: ScModel, data_loader: ScDataLoader, config: Dict[str, Any]):
    """Train the model."""
    from training import ModelTrainer

    trainer = ModelTrainer(model, config['device'])
    trainer.train_epoch(data_loader=data_loader, epochs=config['num_epochs'])


def main():
    """Main function to run the model training."""
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
        'emb_size': args.emb_size,
        'num_of_topic': args.num_of_topic,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'device': args.device,
        'model_name': args.model_name,
        'edge_path': paths['edge_path'],
    }

    # Run preprocessing if needed
    if not os.path.exists(paths['edge_path']):
        preprocess_args = argparse.Namespace(
            dataset=args.dataset,
            tissue=args.tissue,
            threshold=args.threshold,
            distance=args.distance,
            use_hvg=args.use_hvg,
            if_both=args.if_both,
            method=args.preprocess_method
        )
        paths = run_preprocessing(preprocess_args)

    # Load and process data
    paths['best_model_path'] = get_model_save_path(args.tissue, args.model_name)
    rna_adata, atac_adata, rna_adata_gnn, atac_adata_gnn = load_and_process_data(paths)

    print(f"Data shapes: RNA: {rna_adata.shape}, ATAC: {atac_adata.shape}, "
          f"RNA GNN: {rna_adata_gnn.shape}, ATAC GNN: {atac_adata_gnn.shape}")

    # Initialize data loader and model
    data_loader = create_data_loader(config, rna_adata, atac_adata,
                                     rna_adata_gnn, atac_adata_gnn)

    # Handle existing model
    if os.path.exists(paths['best_model_path']):
        print(f"Found existing model at {paths['best_model_path']}")
        print("Loading model parameters and continuing training...")
        node2vec = torch.load(f'./node2vec/{args.dataset}_{args.model_name}_{args.emb_size}_400.pt')
        config['edge_path'] = None
        model = initialize_model(config, rna_adata, atac_adata, rna_adata_gnn,
                                 atac_adata_gnn, args, node2vec)
        model.load_best_model(paths['best_model_path'])
    else:
        print("No existing model found. Starting training from scratch...")
        model = initialize_model(config, rna_adata, atac_adata, rna_adata_gnn,
                                 atac_adata_gnn, args,
                                 data_loader.train_loader.dataset.dataset.fm)

    # Train model
    train_model(model, data_loader, config)

    # Save final model
    model.save(paths['best_model_path'])


if __name__ == '__main__':
    main()