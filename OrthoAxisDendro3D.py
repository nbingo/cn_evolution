from agglomerate.agglomerate_3d import Agglomerate3D
import data.data_loader
from metrics.metric_utils import spearmanr_connectivity
from typing import Sequence, Optional, Literal, Union
import numpy as np
import pandas as pd
import time
import scanpy as sc
from anndata import AnnData
from scipy import stats, sparse
from sklearn.model_selection import StratifiedShuffleSplit
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
import matplotlib.pyplot as plt
import seaborn as sns

sc.settings.verbosity = 0  # Please tell me everything all the time

P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2
GENE_CORR_THRESH = 0.5
BATCH_SIZE = 500


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size, data_source):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        super().__init__(data_source)
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class SparseDataSet(Dataset):
    def __init__(self, anndata: AnnData, group_by: Literal['subregion', 'clusters']):
        # First need to get data into pytorch format
        # Make dict of label names to their id's, which will be used as one-hot positions
        label_to_id = {r: i for i, r in enumerate(np.unique(anndata.obs[group_by]))}
        self.n_obs = len(anndata.obs.index)
        self.n_labels = len(label_to_id)
        self.labels = anndata.obs[group_by].map(label_to_id)
        self.data = anndata.X.tocoo()
        v = torch.FloatTensor(self.data.data)
        i = torch.LongTensor(np.vstack((self.data.row, self.data.col)))
        shape = self.data.shape
        self.data = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def __getitem__(self, item):
        # Need to make region/cell type labels into one-hot format
        return self.data[item].to_dense(), self.labels.iloc[item]

    def __len__(self):
        return self.n_obs


class CTDataLoader(data.data_loader.DataLoader):

    def __init__(self, species: str, reprocess: Optional[bool] = False,
                 gene_selection_method: Optional[Literal['deg', 'lasso']] = 'deg',
                 lasso_cache_dir: Optional[str] = None,
                 l1_weight: Optional[Union[float, Sequence[float]]] = 1e-2, learning_rate: Optional[float] = 1e-3,
                 train_split: Optional[float] = 0.8, n_jobs: Optional[int] = 15,
                 remove_correlated: Optional[Literal['both', 'ct', 'region']] = None,
                 normalize: Optional[bool] = False,
                 dim_reduction: Optional[str] = None, n_components: Optional[int] = None):
        super().__init__()
        torch.set_num_threads(n_jobs)

        filename = f'{species}_ex_colors'

        self.l1_weight = l1_weight
        self.learning_rate = learning_rate
        self.device = 'cpu'

        # Used saved data if possible
        if not reprocess and os.path.exists(f'withcolors_preprocessed/{filename}.pickle'):
            with open(f'withcolors_preprocessed/{filename}.pickle', mode='rb') as file:
                data_dict = pickle.load(file)
                self.data = data_dict['data']
                self.ct_axis_mask = data_dict['ct_axis_mask']
                self.r_axis_mask = data_dict['r_axis_mask']
                # No need to do anything else
                return

        species_data = sc.read(f'withcolors/{filename}.h5ad')
        if dim_reduction is not None:
            sc.pp.pca(species_data, n_comps=n_components)
            sc.pp.highly_variable_genes(species_data)
            sc.pp.neighbors(species_data, n_pcs=n_components)
            if dim_reduction == 'pca':
                sc.tl.pca(species_data, n_comps=n_components)
            elif dim_reduction == 'umap':
                sc.tl.umap(species_data, n_components=n_components)
            elif dim_reduction == 'tsne':
                sc.tl.tsne(species_data, n_pcs=n_components)
            species_data = AnnData(species_data.obsm[f'X_{dim_reduction}'], obs=species_data.obs)
            species_data.var.index = pd.Index([f'{dim_reduction}{x}' for x in range(len(species_data.var.index))])

        # Label each observation with its subregion and species
        species_data.obs['clusters'] = species_data.obs['clusters'].apply(lambda s: species[0].upper() + '_' + s)
        species_data.obs['subregion'] = species_data.obs['clusters'].apply(lambda s: s.split('.')[0])
        self.n_var = len(species_data.var.index)
        self.n_subregions = len(np.unique(species_data.obs['subregion']))
        self.n_clusters = len(np.unique(species_data.obs['clusters']))

        if gene_selection_method == 'deg':
            self._deg_select(dim_reduction, species_data)
        elif gene_selection_method == 'lasso':
            for label in ['subregion', 'clusters']:
                num_labels = self.n_subregions if label == 'subregion' else self.n_clusters
                # define the model
                model = nn.Sequential(
                    # nn.BatchNorm1d(self.n_var),
                    nn.Linear(self.n_var, num_labels)
                )
                model_file = f'{lasso_cache_dir}_{label}.pt'
                if lasso_cache_dir is None or not os.path.exists(model_file):
                    print(f'\nTraining lasso on {label}.\n')
                    # Create the dataset and dataloader
                    ds = SparseDataSet(species_data, label)
                    train_size = int(train_split * len(ds))
                    val_size = len(ds) - train_size
                    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
                    train_dl = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
                    val_dl = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    # train
                    loss_history = self._train_model(model, train_dl, val_dl, optimizer, epochs=25)
                    # save the model
                    torch.save(model.state_dict(), model_file)
                    plt.plot(loss_history[:, 0], label='train loss')
                    plt.plot(loss_history[:, 1], label='val loss')
                    plt.legend()
                    plt.show()
                else:
                    model.load_state_dict(torch.load(model_file))
                # Get the max weight per gene to see whether it's relevant to at least one subregion
                with torch.no_grad():
                    max_weight_per_gene = torch.abs(model[-1].weight).max(dim=0)[0]
                    with torch.no_grad():
                        sns.distplot(max_weight_per_gene)
                        plt.show()
                if label == 'subregion':
                    self.r_axis_mask = max_weight_per_gene > 1e-4
                else:
                    self.ct_axis_mask = max_weight_per_gene > 1e-4

        if remove_correlated is not None:
            self._remove_r_ct_correlated(remove_correlated, species_data)

        # Average transcriptomes within each cell type and put into data frame with cell types as rows and genes as cols
        ct_names = np.unique(species_data.obs['clusters'])
        ct_avg_data = [species_data[species_data.obs['clusters'] == ct].X.mean(axis=0) for ct in ct_names]
        self.data = pd.concat([pd.DataFrame(data.reshape((1, -1)), columns=species_data.var.index, index=[cluster_name])
                               for data, cluster_name in zip(ct_avg_data, ct_names)])
        # Divide each row by mean, as in Tosches et al, rename columns,
        # and transpose so that column labels are genes and rows are cell types
        # Divide each row by mean
        if normalize:
            self.data = self.data.div(self.data.mean(axis=0).to_numpy(), axis=1)        # noqa

        # Save data
        data_dict = {'data': self.data, 'ct_axis_mask': self.ct_axis_mask, 'r_axis_mask': self.r_axis_mask}
        with open(f'withcolors_preprocessed/{filename}.pickle', mode='wb') as file:
            pickle.dump(data_dict, file)

    def _calc_val_loss(self, loader: DataLoader, model: nn.Module):
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            losses = []
            for x, y in loader:
                x = x.to(device=self.device)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)
                scores = model(x)
                loss = F.cross_entropy(scores, y)
                for param in model.parameters():
                    loss += F.l1_loss(param, torch.zeros_like(param)) * self.l1_weight
                losses.append(loss.item())
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            return np.mean(losses)

    def _train_model(self, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
                     optimizer: optim.Optimizer, epochs: int = 1, loss_chang_lim: float = 1e-3,
                     print_every: int = 1):
        loss_history = []
        for epoch in range(epochs):
            model.train()
            epoch_loss_history = []
            for t, (x, y) in enumerate(train_dl):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                scores = model(x).squeeze()
                loss = F.cross_entropy(scores, y)
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += F.l1_loss(param, torch.zeros_like(param)) * self.l1_weight
                loss += reg_loss
                epoch_loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % print_every == 0:
                    print('Iteration %d, Epoch %d, loss = %.4f' % (t + len(train_dl) * epoch, epoch, loss.item()))
            mean_epoch_loss = np.mean(epoch_loss_history)
            if (len(loss_history) > 1) and (np.abs(loss_history[-1][0] - loss_history[-2][0]) < loss_chang_lim):
                break
            loss_history.append([mean_epoch_loss, self._calc_val_loss(val_dl, model)])
        return np.array(loss_history)

    def _deg_select(self, dim_reduction, species_data):
        # Split different regions into separate AnnData-s
        species_data_region_split = [species_data[species_data.obs['subregion'] == sr] for sr in
                                     np.unique(species_data.obs['subregion'])]
        # Compute DEGs between different subregions to get region axis mask
        sc.tl.rank_genes_groups(species_data, groupby='subregion', method='wilcoxon')
        # Filter by adjusted p value and log fold change
        # noinspection PyTypeChecker
        r_axis_name_mask = (pd.DataFrame(species_data.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH)
        if dim_reduction is None:
            # noinspection PyTypeChecker
            r_axis_name_mask &= pd.DataFrame(species_data.uns['rank_genes_groups']['logfoldchanges']) \
                                > AVG_LOG_FC_THRESH
        # Our current mask is actually for names sorted by their z-scores, so have to get back to the original ordering
        r_axis_filtered_names = pd.DataFrame(species_data.uns['rank_genes_groups']['names'])[r_axis_name_mask]
        # Essentially take union between DEGs of different regions
        r_axis_filtered_names = r_axis_filtered_names.to_numpy().flatten()
        # remove nans
        r_axis_filtered_names = r_axis_filtered_names[~pd.isnull(r_axis_filtered_names)]
        # Now go through genes in their original order and check if they are in our list of genes
        self.r_axis_mask = species_data.var.index.isin(r_axis_filtered_names)
        # Compute DEGs between cell types within subregions to get cell type axis mask
        ct_degs_by_subregion = []
        # Iterate over regions in this species
        for sr in species_data_region_split:
            # Need to have at least one cell type in the region
            if len(np.unique(sr.obs['clusters'])) > 1:
                # Compute DEGs for cell types in this region
                sc.tl.rank_genes_groups(sr, groupby='clusters', method='wilcoxon')
                # Filter by adjusted p value and log fold change
                # noinspection PyTypeChecker
                deg_names_mask = pd.DataFrame(sr.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH
                if dim_reduction is None:
                    # noinspection PyTypeChecker
                    deg_names_mask &= pd.DataFrame(sr.uns['rank_genes_groups']['logfoldchanges']) > AVG_LOG_FC_THRESH

                # Get the names
                ct_degs_by_subregion.append(pd.DataFrame(sr.uns['rank_genes_groups']['names'])[deg_names_mask])
        # Construct mask of genes in original ordering
        # Essentially take union between DEGs of different regions
        ct_axis_filtered_names = np.concatenate([degs.to_numpy().flatten() for degs in ct_degs_by_subregion])
        # Get rid of nans
        ct_axis_filtered_names = ct_axis_filtered_names[~pd.isnull(ct_axis_filtered_names)]
        # Now go through genes in their original order and check if they are in our list of genes
        self.ct_axis_mask = species_data.var.index.isin(ct_axis_filtered_names)

    def _remove_r_ct_correlated(self, remove_correlated, species_data):
        # Find correlated genes between ct_axis_mask and r_axis_mask and remove them from both
        # First remove genes that appear in both masks since they must contain both ct and region information
        intersect_mask = self.r_axis_mask & self.ct_axis_mask
        if remove_correlated in ['ct', 'both']:
            self.ct_axis_mask[intersect_mask] = False
        if remove_correlated in ['region', 'both']:
            self.r_axis_mask[intersect_mask] = False
        # Get raw expression data for leftover relevant ct and region genes
        r_genes_raw = species_data.X[:, self.r_axis_mask]
        ct_genes_raw = species_data.X[:, self.ct_axis_mask]
        if isinstance(species_data.X, sparse.csc_matrix):
            r_genes_raw = r_genes_raw.toarray()
            ct_genes_raw = ct_genes_raw.toarray()
        # Compute correlation coefficient between all genes. Unfortunately can't just do all ct to all region
        # and will have to only select those later
        # Should result in a (len(r_genes_raw) + len(ct_genes_raw)) side square matrix
        corrcoefs = stats.spearmanr(r_genes_raw, ct_genes_raw).correlation
        # Threshold the correlations by magnitude, since a negative correlation is still information
        corrcoefs_significant = np.abs(corrcoefs) > GENE_CORR_THRESH
        # Find any ct genes that are correlated to a region gene or vice-versa
        # ct genes that are correlated to some region gene
        num_r_genes = r_genes_raw.shape[1]
        ct_corr_genes = corrcoefs_significant[num_r_genes:, :num_r_genes].any(axis=1)
        # region genes that are correlated to some cell type gene
        r_corr_genes = corrcoefs_significant[:num_r_genes, num_r_genes:].any(axis=1)
        # Convert the masks to indices to correctly remove correlated regions from them
        r_axis_mask_indices = np.where(self.r_axis_mask)[0]
        ct_axis_mask_indices = np.where(self.ct_axis_mask)[0]
        # Remove correlated genes
        if remove_correlated in ['ct', 'both']:
            self.ct_axis_mask[ct_axis_mask_indices[ct_corr_genes]] = False
        if remove_correlated in ['region', 'both']:
            self.r_axis_mask[r_axis_mask_indices[r_corr_genes]] = False

    def get_names(self) -> Sequence[str]:
        return self.data.index.values

    def get_corresponding_region_names(self) -> Sequence[str]:
        def get_region(ct_name: str):
            return ct_name.split('.')[0]

        return np.vectorize(get_region)(self.get_names())

    def __getitem__(self, item):
        return self.data.to_numpy()[item]

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    ct_data_loader = CTDataLoader('mouse', reprocess=True, remove_correlated='ct', normalize=True,
                                  gene_selection_method='lasso', lasso_cache_dir='models/lasso/M_lasso',
                                  dim_reduction=None, n_components=50)

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_mnn',
        max_region_diff=1,
        region_dist_scale=.7,
        verbose=False,
        pbar=True,
        integrity_check=True
    )

    start = time.process_time()
    agglomerate.agglomerate(ct_data_loader)
    end = time.perf_counter()
    pd.options.display.width = 0
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    print(agglomerate.linkage_mat_readable)
    print(agglomerate.compute_tree_score('BME'))
    agglomerate.view_tree3d()
    print(f'Total time elapsed: {(end - start) / 10}s')
