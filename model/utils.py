import scanpy as sc
import matplotlib.pyplot as plt
from typing import Dict, Any
import anndata as ad
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def save_plots(adata: ad.AnnData, best_params: Dict[str, Any], title: str,
               umap_dir: str, tsne_dir: str, epoch: int):
    """Save UMAP and t-SNE visualization plots."""
    sc.pp.neighbors(adata, use_rep='cell_embed', n_neighbors=best_params['n_neighbors'])

    # Generate and save UMAP plot
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[best_params['method'], 'cell_type'], show=False)
    plt.title(f"{title} - ARI: {best_params['ari']}, NMI: {best_params['nmi']}")
    plt.savefig(f"{umap_dir}/{title}_Epoch_{epoch}_ARI_{best_params['ari']}.png")
    plt.close()

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[best_params['method'], 'batch'], show=False)
    plt.title(f"{title} - ARI: {best_params['ari']}, NMI: {best_params['nmi']}")
    plt.savefig(f"{umap_dir}/{title}_Epoch_{epoch}_ARI_{best_params['ari']}_batch.png")
    plt.close()

    # Generate and save t-SNE plot
    sc.tl.tsne(adata, use_rep='cell_embed')
    sc.pl.tsne(adata, color=[best_params['method'], 'cell_type'], show=False)
    plt.title(f"{title} - ARI: {best_params['ari']}, NMI: {best_params['nmi']}")
    plt.savefig(f"{tsne_dir}/{title}_Epoch_{epoch}_ARI_{best_params['ari']}.png")
    plt.close()


def evaluate_clustering(adata: ad.AnnData) -> Dict[str, Any]:
    """Evaluate clustering performance using different methods and parameters."""
    clustering_methods = ["leiden"]
    resolutions = [0.2, 0.3, 0.5, 0.6]
    n_neighbors = [30, 60, 90, 120]

    best_params = {'resolution': 0, 'ari': 0, 'nmi': 0, 'method': None, 'n_neighbors': 0}

    for n_neighbor in n_neighbors:
        sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=n_neighbor)
        for method in clustering_methods:
            clustering_func = sc.tl.leiden if method == 'leiden' else sc.tl.louvain
            for resolution in resolutions:
                clustering_func(adata, resolution=resolution, key_added=method)
                ari = adjusted_rand_score(adata.obs['cell_type'], adata.obs[method])
                nmi = normalized_mutual_info_score(adata.obs['cell_type'], adata.obs[method])

                ari = round(ari, 4)
                nmi = round(nmi, 4)

                if ari > best_params['ari']:
                    best_params.update({
                        'resolution': resolution,
                        'ari': ari,
                        'method': method,
                        'n_neighbors': n_neighbor
                    })
                if nmi > best_params['nmi']:
                    best_params['nmi'] = nmi

    return best_params


def plot_expression_heatmaps(adata: ad.AnnData, true_exp: torch.Tensor,
                             imputed_data: torch.Tensor, save_path: str = None,
                             modality: str = None):
    """Plot heatmaps comparing true and imputed expression data."""
    # Get unique cell types and their indices
    cell_types = adata.obs['cell_type']

    # Perform hierarchical clustering on cell types
    from scipy.cluster import hierarchy
    linkage = hierarchy.linkage(true_exp.numpy(), method='ward')
    ordered_idx = hierarchy.leaves_list(linkage)

    # Order by clustering
    ordered_indices = adata.obs.index[ordered_idx]

    # Reorder matrices
    idx_map = {idx: pos for pos, idx in enumerate(adata.obs.index)}
    ordered_positions = [idx_map[idx] for idx in ordered_indices]

    true_exp_ordered = true_exp[ordered_positions]
    imputed_ordered = imputed_data.reshape(true_exp.shape)[ordered_positions]

    # Plot
    plt.rcParams.update({'font.size': 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Add cell type annotations
    cell_type_labels = cell_types[ordered_indices]

    plt.sca(ax1)
    plt.imshow(true_exp_ordered, cmap='Purples', aspect='auto')
    plt.title('True Expression', fontsize=20)
    plt.xlabel('Genes', fontsize=18)
    plt.ylabel('Cells', fontsize=18)
    plt.colorbar()

    plt.sca(ax2)
    plt.imshow(imputed_ordered, cmap='Purples', aspect='auto')
    plt.title('Imputed Expression', fontsize=20)
    plt.xlabel('Genes', fontsize=18)
    plt.ylabel('Cells', fontsize=18)
    plt.colorbar()

    if save_path:
        plt.savefig(f'{save_path}_{modality}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    return fig