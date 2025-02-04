import os
import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
import gc
import shutil
import argparse
import anndata as ad
from typing import Dict, List, Tuple, Any, Optional
from preprocess.preprocess import run_preprocessing
from scDataLoader import ScDataLoader, set_seed
from model import ScModel


class ModelTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.2e-6)
        self.loss_scaler = GradScaler()

    def train_epoch(self, data_loader: ScDataLoader, epochs: int = 20, plot: bool = False):
        """Train the model for specified number of epochs."""
        train_loader = data_loader.train_loader
        test_loader = None if plot else data_loader.test_loader

        for epoch in range(epochs):
            self.model.epoch = epoch + 1
            print(f"Epoch {epoch + 1}/{epochs}")

            # Calculate weights for different loss components
            recon_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0.6, 8, True)
            kl_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 1e-2, False)
            edge_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 10, False)

            losses = self.train_single_epoch(train_loader, recon_loss_weight,
                                             kl_weight, edge_loss_weight)

            # Print training information
            print(f"recon_loss_weight: {recon_loss_weight}, "
                  f"kl_weight: {kl_weight}, "
                  f"edge_loss_weight: {edge_loss_weight}")
            print(f"Avg Recon Loss: {losses['recon_loss']:.4f}, "
                  f"Avg KL Loss: {losses['kl_loss']:.4f}, "
                  f"Avg edge_recon Loss: {losses['edge_recon_loss']:.4f}, "
                  f"Avg Total Loss: {losses['total_loss']:.4f}")

            if not plot:
                self.model.evaluate_and_save(data_loader)

            gc.collect()

        print(f"Best Train ARI: {self.model.best_train_ari:.4f}, "
              f"Best Test ARI: {self.model.best_test_ari:.4f}")

    def train_single_epoch(self, train_loader: torch.utils.data.DataLoader,
                           recon_loss_weight: float, kl_weight: float,
                           edge_loss_weight: float) -> Dict[str, float]:
        """Train for a single epoch and return losses."""
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_edge_recon_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            with autocast():
                self.optimizer.zero_grad()
                recon_loss, kl_loss, edge_recon_loss, emb = self.model.forward(batch, train_loader)
                loss = (recon_loss_weight * recon_loss +
                        kl_loss * kl_weight +
                        edge_loss_weight * edge_recon_loss)

                self.loss_scaler.scale(loss).backward()
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_edge_recon_loss += edge_recon_loss.item()
            total_loss += loss.item()

        return {
            'recon_loss': total_recon_loss / len(train_loader),
            'kl_loss': total_kl_loss / len(train_loader),
            'edge_recon_loss': total_edge_recon_loss / len(train_loader),
            'total_loss': total_loss / len(train_loader)
        }

    @staticmethod
    def calc_weight(epoch: int, n_epochs: int, cutoff_ratio: float,
                    warmup_ratio: float, min_weight: float, max_weight: float,
                    reverse: bool = False) -> float:
        """Calculate weight for loss components based on training progress."""
        if epoch < n_epochs * cutoff_ratio:
            return 0.0

        fully_warmup_epoch = n_epochs * warmup_ratio
        if warmup_ratio:
            if reverse:
                if epoch < fully_warmup_epoch:
                    return 1.0
                weight_progress = min(1.0, (epoch - fully_warmup_epoch) /
                                      (n_epochs - fully_warmup_epoch))
                weight = max_weight - weight_progress * (max_weight - min_weight)
            else:
                weight_progress = min(1.0, epoch / fully_warmup_epoch)
                weight = min_weight + weight_progress * (max_weight - min_weight)
            return max(min_weight, min(max_weight, weight))
        return max_weight


def get_model_save_path(tissue: str, model_name: str = '', current_date: str = None) -> str:
    """Generate the save path for the model with custom naming."""
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    base_name = f"best_model_{current_date}"

    if model_name:
        base_name += f"_{model_name}"

    return f'./weights/{tissue}/{base_name}.pth'


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


def initialize_model(config: Dict[str, Any], rna_adata: ad.AnnData,
                     atac_adata: ad.AnnData, rna_adata_gnn: ad.AnnData,
                     atac_adata_gnn: ad.AnnData, args: argparse.Namespace, node2vec) -> ScModel:
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


def train_main(args: argparse.Namespace):
    """Main function to run the model training."""
    # Set up configuration
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
    if hasattr(args, 'distance') or hasattr(args, 'threshold'):
        if not os.path.exists(paths['edge_path']):
            preprocess_args = argparse.Namespace(
                dataset=args.dataset,
                tissue=args.tissue,
                threshold=args.threshold,
                distance=args.distance,
                use_hvg=args.use_hvg,
                if_both=args.if_both,
                method=args.preprocess_method,
                score_path=args.score_path,
                motif2tf_path=args.motif2tf_path,
                percentile=args.percentile,
                top=5
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
        state_dict = torch.load(paths['best_model_path'])
        model.load_state_dict(state_dict)
    else:
        print("No existing model found. Starting training from scratch...")
        model = initialize_model(config, rna_adata, atac_adata, rna_adata_gnn,
                                 atac_adata_gnn, args,
                                 data_loader.train_loader.dataset.dataset.fm)

    # Initialize trainer and start training
    trainer = ModelTrainer(model, args.device)
    trainer.train_epoch(data_loader=data_loader, epochs=config['num_epochs'])

    # Save final model
    model.save(paths['best_model_path'])


if __name__ == '__main__':
    pass