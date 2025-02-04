import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import anndata as ad

from components import VAE, GNN, LDEC, xtrimoEmbedding
from utils import evaluate_clustering, save_plots, plot_expression_heatmaps


class ScModel(nn.Module):
    def __init__(self, num_of_gene: int, num_of_peak: int, num_of_gene_gnn: int,
                 num_of_peak_gnn: int, emb_size: int, num_of_topic: int, device: torch.device,
                 batch_size: int, result_save_path: Tuple[str, str], gnn_conv: Optional[str] = None,
                 lr: float = 0.001, best_model_path: Optional[str] = None,
                 use_graph_recon: bool = False, graph_recon_weight: float = 1.0,
                 pos_weight: float = 1.0, node2vec: Optional[torch.Tensor] = None,
                 latent_size: int = 100, use_gnn: bool = True, use_xtrimo: bool = False,
                 shared_decoder: bool = False):
        super().__init__()
        self.device = device
        self.use_graph_recon = use_graph_recon
        self.graph_recon_weight = graph_recon_weight
        self.pos_weight = pos_weight
        self.use_gnn = use_gnn
        self.shared_decoder = shared_decoder
        self.emb_size = emb_size

        # Initialize VAEs
        self.vae_rna = VAE(num_of_gene, emb_size, num_of_topic).to(device)
        self.vae_atac = VAE(num_of_peak, emb_size, num_of_topic).to(device)

        # Initialize GNN if needed
        self.gnn = GNN(emb_size, emb_size * 2, emb_size).to(device) if use_gnn else None

        # Initialize decoders with shared parameters if requested
        if shared_decoder:
            # Create shared alpha transformation
            shared_alphas = nn.Sequential(
                nn.Linear(emb_size, num_of_topic, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
            ).to(device)

            # Create decoders with shared alphas
            self.decoder_rna = LDEC(num_of_gene, emb_size, num_of_topic, batch_size,
                                    shared_module=shared_alphas).to(device)
            self.decoder_atac = LDEC(num_of_peak, emb_size, num_of_topic, batch_size,
                                     shared_module=shared_alphas).to(device)
        else:
            # Create independent decoders
            self.decoder_rna = LDEC(num_of_gene, emb_size, num_of_topic, batch_size).to(device)
            self.decoder_atac = LDEC(num_of_peak, emb_size, num_of_topic, batch_size).to(device)

        # Store all models in ModuleList for consistency
        self.models = nn.ModuleList([
            self.vae_rna,
            self.vae_atac,
            self.gnn,
            self.decoder_rna,
            self.decoder_atac,
        ])

        # Initialize embeddings and additional components
        self.setup_embeddings(num_of_gene, num_of_peak, node2vec, latent_size, use_xtrimo)
        self.setup_network_components(num_of_gene, num_of_peak, num_of_gene_gnn,
                                      num_of_peak_gnn, emb_size, batch_size)
        self.initialize_metrics()

        self.umap_dir, self.tsne_dir = result_save_path

    def setup_embeddings(self, num_of_gene: int, num_of_peak: int,
                         node2vec: Optional[torch.Tensor], latent_size: int, use_xtrimo: bool):
        """Setup embeddings for the model."""
        self.use_xtrimo = use_xtrimo
        self.node2vec = node2vec

        if node2vec is not None:
            self.gene_node2vec, self.peak_node2vec = self.split_tensor(node2vec, num_of_gene)
        else:
            self.gene_node2vec = torch.randn(num_of_gene, self.emb_size)
            self.peak_node2vec = torch.randn(num_of_peak, self.emb_size)

        if use_xtrimo:
            self.gene_embedding = xtrimoEmbedding(
                d=self.emb_size,
                b=latent_size,
                node2vec=self.gene_node2vec
            ).to(self.device)

            self.atac_embedding = xtrimoEmbedding(
                d=self.emb_size,
                b=latent_size,
                node2vec=self.peak_node2vec
            ).to(self.device)
        else:
            self.gene_embedding = None
            self.atac_embedding = None

    def setup_network_components(self, num_of_gene: int, num_of_peak: int,
                                 num_of_gene_gnn: int, num_of_peak_gnn: int,
                                 emb_size: int, batch_size: int):
        """Setup neural network components."""
        self.batch_to_emb = nn.Sequential(
            nn.Linear(batch_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.LeakyReLU(),
        ).to(self.device)

        self.lin_rna = nn.Sequential(
            nn.Linear(num_of_gene_gnn, num_of_gene),
            nn.BatchNorm1d(num_of_gene),
            nn.LeakyReLU(),
        ).to(self.device)

        self.lin_peak = nn.Sequential(
            nn.Linear(num_of_peak_gnn, num_of_peak),
            nn.BatchNorm1d(num_of_peak),
            nn.LeakyReLU(),
        ).to(self.device)

    def initialize_metrics(self):
        """Initialize model metrics."""
        self.best_emb = None
        self.best_model = None
        self.best_train_ari = 0
        self.best_test_ari = 0
        self.epoch = 0

    def forward(self, batch: List[torch.Tensor], train_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the model."""
        batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor)
                 else tensor for tensor in batch]

        (RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
         RNA_tensor_gnn, RNA_tensor_normalized_gnn,
         ATAC_tensor_gnn, ATAC_tensor_normalized_gnn, batch_indices) = batch

        encoder1, encoder2, gnn, decoder1, decoder2 = self.models

        # Process RNA data
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        z1 = self.reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        # Process ATAC data
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)
        z2 = self.reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        # Initialize variables
        rho, eta = None, None
        edge_recon_loss = torch.tensor(0.0, device=self.device)
        emb = None

        # GNN processing if enabled
        if self.use_gnn:
            edge_index = self.get_edge_index(train_loader)

            if self.use_xtrimo:
                # Use learned embeddings
                gene_embeddings = self.gene_embedding(RNA_tensor)
                atac_embeddings = self.atac_embedding(ATAC_tensor)
                fm = torch.cat((gene_embeddings, atac_embeddings), dim=0)
            else:
                specific_fm = torch.cat((ATAC_tensor_normalized.T, RNA_tensor_normalized.T), dim=0)
                specific_fm = self.batch_to_emb(specific_fm).to(self.device)
                fm = specific_fm * self.node2vec.to(self.device)

            # Pass through GNN
            emb = gnn(fm, edge_index)
            rho, eta = self.split_tensor(emb, RNA_tensor_gnn.shape[1])

            # Compute graph reconstruction loss if enabled
            if self.use_graph_recon:
                edge_recon_loss = self.compute_graph_reconstruction_loss(emb, edge_index)
                edge_recon_loss = edge_recon_loss * self.graph_recon_weight

        # Decode
        pred_RNA_tensor, rho = decoder1(theta1, rho if self.use_gnn else None)
        pred_ATAC_tensor, eta = decoder2(theta2, eta if self.use_gnn else None)

        if not self.use_gnn:
            emb = torch.cat((rho, eta), dim=0)

        # Calculate losses
        recon_loss1 = -(pred_RNA_tensor * RNA_tensor).sum(-1)
        recon_loss2 = -(pred_ATAC_tensor * ATAC_tensor).sum(-1)
        recon_loss = (recon_loss1 + recon_loss2).mean()
        kl_loss = (kl_theta1 + kl_theta2).mean()

        return recon_loss, kl_loss, edge_recon_loss, emb

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def compute_graph_reconstruction_loss(self, emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute graph reconstruction loss."""
        num_pos_edges = edge_index.shape[1]

        src_emb = emb[edge_index[0]]
        dst_emb = emb[edge_index[1]]
        pos_logits = torch.sum(src_emb * dst_emb, dim=1)

        num_nodes = emb.shape[0]
        neg_src = torch.randint(0, num_nodes, (num_pos_edges,), device=self.device)
        neg_dst = torch.randint(0, num_nodes, (num_pos_edges,), device=self.device)

        neg_src_emb = emb[neg_src]
        neg_dst_emb = emb[neg_dst]
        neg_logits = torch.sum(neg_src_emb * neg_dst_emb, dim=1)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.zeros(2 * num_pos_edges, dtype=torch.float32, device=self.device)
        labels[:num_pos_edges] = 1.0

        pos_weight = torch.tensor([self.pos_weight], device=self.device)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction='mean'
        )

        return loss

    def get_edge_index(self, train_loader: torch.utils.data.DataLoader,
                       edge_num: Optional[int] = None) -> torch.Tensor:
        """Get edge indices for GNN."""
        edge_index = train_loader.dataset.dataset.edge_index.to(self.device)
        if edge_num:
            selected_edge_index = torch.randperm(edge_index.size(1))[:edge_num]
            edge_index = edge_index[:, selected_edge_index]
        return edge_index

    @staticmethod
    def split_tensor(tensor: torch.Tensor, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into two parts at specified row."""
        if num_rows >= tensor.shape[0]:
            raise ValueError("num_rows should be less than tensor's number of rows")
        return tensor[:num_rows, :], tensor[num_rows:, :]

    def evaluate(self, x1: torch.Tensor, x2: torch.Tensor,
                 adata: ad.AnnData, is_train: bool = True) -> float:
        """Evaluate model performance using clustering metrics."""
        theta = self.get_theta(x1.to(self.device), x2.to(self.device))
        adata.obsm['cell_embed'] = theta.cpu().numpy()
        best_params = evaluate_clustering(adata)

        res = f"{best_params['n_neighbors']}_{best_params['method']}_{best_params['resolution']}"
        ari = round(best_params['ari'], 4)
        nmi = round(best_params['nmi'], 4)

        save_plots(adata, best_params, 'Train' if is_train else 'Test',
                   self.umap_dir, self.tsne_dir, self.epoch)
        print(f"{'Train' if is_train else 'Test'} Clustering Info: {res}, ARI: {ari}, NMI: {nmi}")

        return ari

    def get_theta(self, RNA_tensor_normalized: torch.Tensor,
                  ATAC_tensor_normalized: torch.Tensor) -> torch.Tensor:
        """Get theta values from RNA and ATAC data."""
        encoder1, encoder2, _, _, _ = self.models
        batch_size = RNA_tensor_normalized.shape[0]
        n_samples = RNA_tensor_normalized.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_thetas = []

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                rna_batch = RNA_tensor_normalized[start_idx:end_idx]
                atac_batch = ATAC_tensor_normalized[start_idx:end_idx]

                # Process RNA
                mu1, log_sigma1, _ = encoder1(rna_batch)
                z1 = self.reparameterize(mu1, log_sigma1)
                theta1 = F.softmax(z1, dim=-1)

                # Process ATAC
                mu2, log_sigma2, _ = encoder2(atac_batch)
                z2 = self.reparameterize(mu2, log_sigma2)
                theta2 = F.softmax(z2, dim=-1)

                # Average the thetas
                theta_batch = (z1 + z2) / 2
                all_thetas.append(theta_batch)

                # Clear GPU memory
                del mu1, log_sigma1, z1, theta1
                del mu2, log_sigma2, z2, theta2
                torch.cuda.empty_cache()

        # Concatenate all batches
        return torch.cat(all_thetas, dim=0)

    def evaluate_and_save(self, data_loader):
        """Evaluate model performance and save best model."""
        with torch.no_grad():
            train_ari = self.evaluate(
                data_loader.get_all_train_data()['X_rna_tensor_normalized'],
                data_loader.get_all_train_data()['X_atac_tensor_normalized'],
                data_loader.get_all_train_data()['rna_adata'],
                True
            )

            test_ari = self.evaluate(
                data_loader.get_all_test_data()['X_rna_tensor_normalized'],
                data_loader.get_all_test_data()['X_atac_tensor_normalized'],
                data_loader.get_all_test_data()['rna_adata'],
                False
            )

        if test_ari >= self.best_test_ari:
            self.best_test_ari = test_ari
            self.best_model = self.state_dict()

        if train_ari >= self.best_train_ari:
            self.best_train_ari = train_ari

    def save(self, path: str):
        """Save the best model state."""
        if self.best_model is not None:
            torch.save(self.best_model, path)
        else:
            torch.save(self.state_dict(), path)
            print("Saved the last model")

    def load_best_model(self, path: str):
        """Load the best model state."""
        self.best_model = torch.load(path)
        self.load_state_dict(self.best_model)
        print(f'Loaded best model from {path}')