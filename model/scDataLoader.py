import random
import numpy as np
import torch
import anndata as ad
import scanpy as sc
import pickleL
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import Node2Vec



def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScDataset(Dataset):
    def __init__(
            self,
            rna_adata,
            atac_adata,
            num_of_gene,
            num_of_peak,
            rna_adata_gnn,
            atac_adata_gnn,
            emb_size=512,
            edge_path=None,
            feature_type='node2vec',
            model_name=''
    ):
        self.indices = range(
            len(rna_adata))
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        self.rna_adata_gnn = rna_adata_gnn
        self.atac_adata_gnn = atac_adata_gnn
        self.edge_path = edge_path
        self.feature_type = feature_type

        self.num_of_gene = num_of_gene
        self.num_of_peak = num_of_peak

        self.fm_save_path = model_name

        self.rna_adata, self.X_rna_tensor, self.X_rna_tensor_normalized = (
            self.process_mod(self.rna_adata, self.num_of_gene, False))
        self.atac_adata, self.X_atac_tensor, self.X_atac_tensor_normalized = (
            self.process_mod(self.atac_adata, self.num_of_peak, False))

        self.rna_adata_gnn, self.X_rna_tensor_gnn, self.X_rna_tensor_normalized_gnn = (
            self.process_mod(self.rna_adata_gnn, None, True)
        )
        self.atac_adata_gnn, self.X_atac_tensor_gnn, self.X_atac_tensor_normalized_gnn = (
            self.process_mod(self.atac_adata_gnn, None, True)
        )

        feature_size = self.rna_adata_gnn.n_vars + self.atac_adata_gnn.n_vars

        if self.edge_path:
            with open(self.edge_path, 'rb') as f:
                edge_index = pickle.load(f)
            if sp.issparse(edge_index):
                edge_index_coo = edge_index.tocoo()
                row = edge_index_coo.row.astype(np.int64)
                col = edge_index_coo.col.astype(np.int64)
                self.edge_index = torch.tensor([row, col], dtype=torch.long)
            else:
                self.edge_index = torch.tensor(edge_index)
        else:
            self.edge_index = self.get_edge_index(feature_size)

        self.fm = self.get_feature_matrix(feature_size, emb_size)

    @staticmethod
    def process_mod(adata, num_of_mod, is_gnn_adata):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # if not is_gnn_adata:
        #     sc.pp.highly_variable_genes(adata, n_top_genes=num_of_mod)
        #     adata = adata[:, adata.var['highly_variable']][:, :num_of_mod]
        X = adata.X.toarray()

        X_tensor = torch.from_numpy(X).float()
        sums = X_tensor.sum(1).unsqueeze(1)
        X_tensor_normalized = X_tensor / sums

        return adata, X_tensor, X_tensor_normalized

    @staticmethod
    def get_edge_index(dim):
        num_edges = 100
        row = np.random.randint(0, dim, num_edges)
        col = np.random.randint(0, dim, num_edges)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        return edge_index

    def get_feature_matrix(self, dim, emb_size):
        if self.feature_type == 'node2vec':
            return self.get_node2vec_features(emb_size)
        else:
            return self.get_random_features(dim, emb_size)

    @staticmethod
    def get_random_features(dim, emb_size):
        fm = np.random.randn(dim, emb_size)
        fm = torch.tensor(fm, dtype=torch.float)
        return fm

    def get_node2vec_features(self, emb_size):
        Node2Vec_model = Node2Vec(self.edge_index, embedding_dim=emb_size, walk_length=15, context_size=10,
                                  walks_per_node=10)
        device = torch.device("cuda" if torch.cuda.is_available() else "")
        Node2Vec_model = Node2Vec_model.to(device)

        Node2Vec_model.train()
        optimizer = torch.optim.Adam(Node2Vec_model.parameters(), lr=0.01)
        loader = Node2Vec_model.loader(batch_size=256, shuffle=False, num_workers=0)
        for _ in tqdm(range(400)):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        node_embeddings = Node2Vec_model.embedding.weight.detach()
        print(node_embeddings.shape)
        torch.save(node_embeddings, f'./node2vec/pbmc_{self.fm_save_path}_{emb_size}_400.pt')

        return node_embeddings

    @staticmethod
    def custom_collate_fn(batch):
        (rna_batch, rna_batch_normalized,
         atac_batch, atac_batch_normalized,
         rna_batch_gnn, rna_batch_normalized_gnn,
         atac_batch_gnn, atac_batch_normalized_gnn, indices) = zip(*batch)

        rna_batch = torch.stack(rna_batch)
        rna_batch_normalized = torch.stack(rna_batch_normalized)
        atac_batch = torch.stack(atac_batch)
        atac_batch_normalized = torch.stack(atac_batch_normalized)

        rna_batch_gnn = torch.stack(rna_batch_gnn)
        rna_batch_normalized_gnn = torch.stack(rna_batch_normalized_gnn)
        atac_batch_gnn = torch.stack(atac_batch_gnn)
        atac_batch_normalized_gnn = torch.stack(atac_batch_normalized_gnn)
        indices = indices

        return (rna_batch, rna_batch_normalized,
                atac_batch, atac_batch_normalized,
                rna_batch_gnn, rna_batch_normalized_gnn,
                atac_batch_gnn, atac_batch_normalized_gnn, indices)

    def __len__(self):
        return self.rna_adata.shape[0]

    def __getitem__(self, idx):
        rna_item = self.X_rna_tensor[idx]
        rna_item_normalized = self.X_rna_tensor_normalized[idx]
        atac_item = self.X_atac_tensor[idx]
        atac_item_normalized = self.X_atac_tensor_normalized[idx]
        rna_item_gnn = self.X_rna_tensor_gnn[idx]
        rna_item_normalized_gnn = self.X_rna_tensor_normalized_gnn[idx]
        atac_item_gnn = self.X_atac_tensor_gnn[idx]
        atac_item_normalized_gnn = self.X_atac_tensor_normalized_gnn[idx]

        return (rna_item, rna_item_normalized,
                atac_item, atac_item_normalized,
                rna_item_gnn, rna_item_normalized_gnn,
                atac_item_gnn, atac_item_normalized_gnn, self.indices[idx])

    def get_all_data(self, indices):
        return {

            'X_rna_tensor_normalized': self.X_rna_tensor_normalized[indices],
            'X_atac_tensor_normalized': self.X_atac_tensor_normalized[indices],
            'X_rna_tensor_unnormalized': self.X_rna_tensor[indices],
            'X_atac_tensor_unnormalized': self.X_atac_tensor[indices],
            'rna_adata': self.rna_adata[indices],
            'atac_adata': self.atac_adata[indices],
            'indices': indices
        }


class ScDataLoader(DataLoader):
    def __init__(
            self,
            rna_adata,
            atac_adata,
            num_of_gene,
            num_of_peak,
            rna_adata_gnn,
            atac_adata_gnn,
            emb_size,
            batch_size=32,
            shuffle= False,
            num_workers=1,
            pin_memory=False,
            edge_path=None,
            feature_type='node2vec',
            train_ratio=0.8,
            cell_type='',
            fm_save_path=None,
    ):
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        self.num_of_gene = num_of_gene
        self.num_of_peak = num_of_peak
        self.rna_adata_gnn = rna_adata_gnn
        self.atac_adata_gnn = atac_adata_gnn
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.test_dataset = None
        self.test_loader = None
        self.train_loader = None
        self._log_hyperparams = None


        self.scdataset = ScDataset(
            rna_adata, atac_adata,
            num_of_gene, num_of_peak,
            rna_adata_gnn, atac_adata_gnn,
            emb_size=emb_size,
            edge_path=edge_path, feature_type=feature_type, model_name=fm_save_path,
        )

        if not cell_type:
            train_size = int(train_ratio * len(self.scdataset))
            test_size = len(self.scdataset) - train_size
            self.train_dataset, self.test_dataset = random_split(self.scdataset, [train_size, test_size])
        else:
            self.train_dataset, self.test_dataset = random_split(self.scdataset, [len(self.scdataset), 0])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.scdataset.custom_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.scdataset.custom_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )


    def get_all_train_data(self):
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'indices'):
            train_indices = self.train_dataset.indices
        else:
            train_indices = range(len(self.scdataset))
        return self.scdataset.get_all_data(train_indices)

    def get_all_test_data(self):
        if hasattr(self, 'test_loader'):
            test_indices = self.test_dataset.indices
            return self.scdataset.get_all_data(test_indices)
        else:
            return None

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
    edge_path = None
    num_of_gene = 20000
    num_of_peak = 100000

    rna_adata = ad.read_h5ad("../../data/10x-Multiome-Pbmc10k-RNA.h5ad")
    atac_adata = ad.read_h5ad("../../data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    rna_adata_gnn = ad.read_h5ad('../../data/PBMC/PBMC_processed/PBMC_filtered_nolog_rna_count.h5ad')
    atac_adata_gnn = ad.read_h5ad("../../data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    # edge_path = f'{data_folder}HOMER/hvg/homer_GRN_both_dist_150K_top_10_matrix.pkl'

    num_of_gene_gnn = rna_adata_gnn.n_vars
    num_of_peak_gnn = atac_adata_gnn.n_vars

    emb_size = 512
    num_of_topic = 100
    batch_size = 32
    num_epochs = 20
    lr = 0.001
    split_ratio = 0.8
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # best_model_path = f'./weights/best_model_{datetime.datetime.now()}.pth'

    # gnn_conv = 'GCN'
    gnn_conv = None

    data_loader = ScDataLoader(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        rna_adata_gnn=rna_adata_gnn,
        atac_adata_gnn=atac_adata_gnn,
        emb_size=emb_size,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        edge_path=edge_path,
        feature_type='',
    )

    print(data_loader.train_loader.dataset.dataset.fm)
