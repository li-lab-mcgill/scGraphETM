import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import pickle


class ImputationTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_with_imputation(self, data_loader, epochs: int = 20,
                              imputation_weight: float = 1.0, plot: bool = False):
        """Train the model with additional imputation loss."""
        train_loader = data_loader.train_loader
        test_loader = None if plot else data_loader.test_loader

        for epoch in range(epochs):
            self.model.epoch = epoch + 1
            print(f"Epoch {epoch + 1}/{epochs}")

            # Calculate weights for different loss components
            recon_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0.6, 8, True)
            kl_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 1e-2, False)
            edge_loss_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, 10, False)
            imp_weight = self.calc_weight(epoch, epochs, 0, 2 / 4, 0, imputation_weight, False)

            losses = self.train_single_epoch(
                train_loader,
                recon_loss_weight,
                kl_weight,
                edge_loss_weight,
                imp_weight
            )

            # Print training information
            print(f"recon_loss_weight: {recon_loss_weight}, "
                  f"kl_weight: {kl_weight}, "
                  f"edge_loss_weight: {edge_loss_weight}, "
                  f"imputation_weight: {imp_weight}")
            print(f"Avg Recon Loss: {losses['recon_loss']:.4f}, "
                  f"Avg KL Loss: {losses['kl_loss']:.4f}, "
                  f"Avg Edge Recon Loss: {losses['edge_recon_loss']:.4f}, "
                  f"Avg Imputation Loss: {losses['imputation_loss']:.4f}, "
                  f"Avg Total Loss: {losses['total_loss']:.4f}")

            if not plot:
                self.model.evaluate_and_save(data_loader)

        print(f"Best Train ARI: {self.model.best_train_ari:.4f}, "
              f"Best Test ARI: {self.model.best_test_ari:.4f}")

    def train_single_epoch(self, train_loader, recon_loss_weight: float,
                           kl_weight: float, edge_loss_weight: float,
                           imputation_weight: float) -> Dict[str, float]:
        """Train for a single epoch with imputation loss."""
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_edge_recon_loss = 0
        total_imputation_loss = 0

        for batch in tqdm(train_loader, desc="Training", disable=True):
            with autocast():
                # Forward pass with original reconstruction and KL divergence
                recon_loss, kl_loss, edge_recon_loss, _ = self.model.forward(batch, train_loader)

                # Calculate imputation loss
                imputation_loss = self.calculate_imputation_loss(batch)

                # Combine all losses
                loss = (recon_loss * recon_loss_weight +
                        kl_loss * kl_weight +
                        edge_recon_loss * edge_loss_weight +
                        imputation_loss * imputation_weight)

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_edge_recon_loss += edge_recon_loss.item()
            total_imputation_loss += imputation_loss.item()
            total_loss += loss.item()

        num_batches = len(train_loader)
        return {
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'edge_recon_loss': total_edge_recon_loss / num_batches,
            'imputation_loss': total_imputation_loss / num_batches,
            'total_loss': total_loss / num_batches
        }

    def calculate_imputation_loss(self, batch) -> torch.Tensor:
        """Calculate imputation loss using cross-modal reconstruction."""
        batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor)
                 else tensor for tensor in batch]

        (RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
         RNA_tensor_gnn, RNA_tensor_normalized_gnn,
         ATAC_tensor_gnn, ATAC_tensor_normalized_gnn, batch_indices) = batch

        encoder1, encoder2, gnn, decoder1, decoder2 = self.model.models

        # RNA -> ATAC imputation
        mu1, log_sigma1, _ = encoder1(RNA_tensor_normalized)
        z1 = self.model.reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        # ATAC -> RNA imputation
        mu2, log_sigma2, _ = encoder2(ATAC_tensor_normalized)
        z2 = self.model.reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        # Get GNN embeddings if enabled
        rho, eta = None, None
        if self.model.use_gnn:
            edge_index = self.model.get_edge_index(None)

            if self.model.use_xtrimo:
                gene_embeddings = self.model.gene_embedding(RNA_tensor)
                atac_embeddings = self.model.atac_embedding(ATAC_tensor)
                fm = torch.cat((gene_embeddings, atac_embeddings), dim=0)
            else:
                specific_fm = torch.cat((ATAC_tensor_normalized.T, RNA_tensor_normalized.T), dim=0)
                specific_fm = self.model.batch_to_emb(specific_fm).to(self.device)
                fm = specific_fm * self.model.node2vec.to(self.device)

            emb = gnn(fm, edge_index)
            rho, eta = self.model.split_tensor(emb, RNA_tensor_gnn.shape[1])

        # Cross-modal reconstruction
        pred_RNA_from_ATAC, _ = decoder1(theta2, rho if self.model.use_gnn else None)
        pred_ATAC_from_RNA, _ = decoder2(theta1, eta if self.model.use_gnn else None)

        # Calculate reconstruction losses
        rna_imputation_loss = -(pred_RNA_from_ATAC * RNA_tensor).sum(-1)
        atac_imputation_loss = -(pred_ATAC_from_RNA * ATAC_tensor).sum(-1)

        return (rna_imputation_loss + atac_imputation_loss).mean()

    def impute(self, data_loader, impute_plot_path: Optional[str] = None) -> Tuple[
        Dict[str, Dict[str, float]], Dict[str, torch.Tensor]]:
        """Impute RNA and ATAC data and calculate correlations."""
        encoder1, encoder2, gnn, decoder1, decoder2 = self.model.models
        self.model.eval()

        # Initialize storage for predictions and ground truth
        all_predictions = {
            'RNA': {'original': [], 'imputed': [], 'norm': []},
            'ATAC': {'original': [], 'imputed': [], 'norm': []}
        }

        indices = []
        with torch.no_grad():
            for batch in tqdm(data_loader.train_loader, desc="Imputing"):
                batch = [tensor.to(self.device) if isinstance(tensor, torch.Tensor)
                         else tensor for tensor in batch]

                (RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
                 RNA_tensor_gnn, RNA_tensor_normalized_gnn,
                 ATAC_tensor_gnn, ATAC_tensor_normalized_gnn, batch_indices) = batch

                indices += batch_indices

                # Process RNA -> ATAC
                mu1, log_sigma1, _ = encoder1(RNA_tensor_normalized)
                z1 = self.model.reparameterize(mu1, log_sigma1)
                theta1 = F.softmax(z1, dim=-1)

                # Process ATAC -> RNA
                mu2, log_sigma2, _ = encoder2(ATAC_tensor_normalized)
                z2 = self.model.reparameterize(mu2, log_sigma2)
                theta2 = F.softmax(z2, dim=-1)

                rho, eta = None, None
                if self.model.use_gnn:
                    edge_index = data_loader.train_loader.dataset.dataset.edge_index.to(self.device)

                    if self.model.use_xtrimo:
                        gene_embeddings = self.model.gene_embedding(RNA_tensor_normalized)
                        atac_embeddings = self.model.atac_embedding(ATAC_tensor_normalized)
                        fm = torch.cat((gene_embeddings, atac_embeddings), dim=0)
                    else:
                        specific_fm = torch.cat((ATAC_tensor_normalized.T, RNA_tensor_normalized.T), dim=0)
                        specific_fm = self.model.batch_to_emb(specific_fm).to(self.device)
                        fm = specific_fm * self.model.node2vec.to(self.device)

                    emb = gnn(fm, edge_index)
                    rho, eta = self.model.split_tensor(emb, RNA_tensor_gnn.shape[1])

                # Imputation
                rna_imputed, _ = decoder1(theta2, rho if self.model.use_gnn else None)
                atac_imputed, _ = decoder2(theta1, eta if self.model.use_gnn else None)

                # Store results
                all_predictions['RNA']['original'].append(RNA_tensor.cpu().numpy())
                all_predictions['RNA']['imputed'].append(rna_imputed.cpu().numpy())
                all_predictions['RNA']['norm'].append(RNA_tensor_normalized.cpu().numpy())

                all_predictions['ATAC']['original'].append(ATAC_tensor.cpu().numpy())
                all_predictions['ATAC']['imputed'].append(atac_imputed.cpu().numpy())
                all_predictions['ATAC']['norm'].append(ATAC_tensor_normalized.cpu().numpy())

        # Process results and calculate correlations
        correlations, imputed_counts = self._process_imputation_results(
            all_predictions, indices, impute_plot_path)

        return correlations, imputed_counts

    def _process_imputation_results(self, all_predictions: Dict[str, Dict[str, list]],
                                    indices: list, impute_plot_path: Optional[str] = None) -> Tuple[
        Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        """Process imputation results and calculate correlations."""
        # Save indices
        with open('theta_batch_indices.pkl', 'wb') as f:
            pickle.dump(indices, f)

        # Concatenate all batches
        for modality in ['RNA', 'ATAC']:
            for data_type in ['original', 'imputed', 'norm']:
                all_predictions[modality][data_type] = np.concatenate(
                    all_predictions[modality][data_type], axis=0)

        correlations = {}
        imputed_counts = {}

        theta = self.model.get_theta(
            torch.tensor(all_predictions['RNA']['norm']).to(self.device),
            torch.tensor(all_predictions['ATAC']['norm']).to(self.device)
        )
        torch.save(theta, 'theta.pt')

        for modality in ['RNA', 'ATAC']:
            orig = all_predictions[modality]['original']
            imp = all_predictions[modality]['imputed']

            if impute_plot_path:
                # Calculate correlations
                pearson_corr, _ = pearsonr(orig.flatten(), imp.flatten())
                spearman_corr, _ = spearmanr(orig.flatten(), imp.flatten())

                correlations[modality] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr
                }

                imputed_counts[modality] = imp

                print(f"\n==== {modality} Imputation Results ====")
                print(f"Pearson correlation: {correlations[modality]['pearson']:.4f}")
                print(f"Spearman correlation: {correlations[modality]['spearman']:.4f}")

                self._plot_imputation_results(
                    orig.flatten(),
                    imp.flatten(),
                    impute_plot_path,
                    modality,
                    pearson_corr,
                    spearman_corr
                )

        return correlations, imputed_counts

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

    def _process_imputation_results(self, all_predictions: Dict[str, Dict[str, list]],
                                    indices: list, impute_plot_path: Optional[str] = None) -> Tuple[
        Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        """Process imputation results and calculate correlations."""
        from utils import plot_imputation_results

        # Save indices
        with open('theta_batch_indices.pkl', 'wb') as f:
            pickle.dump(indices, f)

        # Concatenate all batches
        for modality in ['RNA', 'ATAC']:
            for data_type in ['original', 'imputed', 'norm']:
                all_predictions[modality][data_type] = np.concatenate(
                    all_predictions[modality][data_type], axis=0)

        correlations = {}
        imputed_counts = {}

        theta = self.model.get_theta(
            torch.tensor(all_predictions['RNA']['norm']).to(self.device),
            torch.tensor(all_predictions['ATAC']['norm']).to(self.device)
        )
        torch.save(theta, 'theta.pt')

        for modality in ['RNA', 'ATAC']:
            orig = all_predictions[modality]['original']
            imp = all_predictions[modality]['imputed']

            if impute_plot_path:
                # Calculate correlations
                pearson_corr, _ = pearsonr(orig.flatten(), imp.flatten())
                spearman_corr, _ = spearmanr(orig.flatten(), imp.flatten())

                correlations[modality] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr
                }

                imputed_counts[modality] = imp

                print(f"\n==== {modality} Imputation Results ====")
                print(f"Pearson correlation: {correlations[modality]['pearson']:.4f}")
                print(f"Spearman correlation: {correlations[modality]['spearman']:.4f}")

                # Use plotting function from utils
                plot_imputation_results(
                    orig.flatten(),
                    imp.flatten(),
                    impute_plot_path,
                    modality,
                    pearson_corr,
                    spearman_corr
                )

        return correlations, imputed_counts