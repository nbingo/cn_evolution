from agglomerate.agglomerate_3d import Agglomerate3D
from data.data_loader import DataLoader
from metrics.metric_utils import spearmanr_connectivity
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import time
import scanpy as sc
from scipy import stats
import os
import pickle

sc.settings.verbosity = 0  # Please tell me everything all the time

P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2
GENE_CORR_THRESH = 0.5


class CTDataLoader(DataLoader):

    def __init__(self, reprocess: Optional[bool] = False):
        super().__init__()
        # We're going to find the right mask for the mouse and then use it on the chicken
        # Used saved data if possible
        if not reprocess and os.path.exists(f'withcolors_preprocessed/chicken_from_mouse_ex_colors.pickle'):
            with open(f'withcolors_preprocessed/chicken_from_mouse_ex_colors.pickle', mode='rb') as file:
                data_dict = pickle.load(file)
                self.data = data_dict['data']
                self.ct_axis_mask = data_dict['ct_axis_mask']
                self.r_axis_mask = data_dict['r_axis_mask']
                # No need to do anything else
                return

        mouse_data = sc.read(f'withcolors/mouse_ex_colors.h5ad')

        # Label each observation with its region and species
        mouse_data.obs['clusters'] = mouse_data.obs['clusters'].apply(lambda s: 'M_' + s)
        mouse_data.obs['subregion'] = mouse_data.obs['clusters'].apply(lambda s: s.split('.')[0])

        # Split different regions into separate AnnData-s
        species_data_region_split = [mouse_data[mouse_data.obs['subregion'] == sr] for sr in
                                     np.unique(mouse_data.obs['subregion'])]

        # Compute DEGs between different subregions to get region axis mask
        sc.tl.rank_genes_groups(mouse_data, groupby='subregion', method='wilcoxon')
        # Filter by adjusted p value and log fold change
        r_axis_name_mask = ((pd.DataFrame(mouse_data.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH) &
                            (pd.DataFrame(mouse_data.uns['rank_genes_groups']['logfoldchanges']) > AVG_LOG_FC_THRESH))
        # Our current mask is actually for names sorted by their z-scores, so have to get back to the original ordering
        r_axis_filtered_names = pd.DataFrame(mouse_data.uns['rank_genes_groups']['names'])[r_axis_name_mask]
        # Essentially take union between DEGs of different regions
        r_axis_filtered_names = r_axis_filtered_names.to_numpy().flatten()
        # Now go through genes in their original order and check if they are in our list of genes
        self.r_axis_mask = mouse_data.var.index.isin(r_axis_filtered_names)

        # Compute DEGs between cell types within subregions to get cell type axis mask
        ct_degs_by_subregion = []
        # Iterate over regions in this species
        for sr in species_data_region_split:
            # Need to have at least one cell type in the region
            if len(np.unique(sr.obs['clusters'])) > 1:
                # Compute DEGs for cell types in this region
                sc.tl.rank_genes_groups(sr, groupby='clusters', method='wilcoxon')
                # Filter by adjusted p value and log fold change
                deg_names_mask = ((pd.DataFrame(sr.uns['rank_genes_groups']['pvals_adj']) < P_VAL_ADJ_THRESH) &
                                  (pd.DataFrame(sr.uns['rank_genes_groups']['logfoldchanges']) > AVG_LOG_FC_THRESH))
                # Get the names
                ct_degs_by_subregion.append(pd.DataFrame(sr.uns['rank_genes_groups']['names'])[deg_names_mask])

        # Construct mask of genes in original ordering
        # Essentially take union between DEGs of different regions
        ct_axis_filtered_names = np.concatenate([degs.to_numpy().flatten() for degs in ct_degs_by_subregion])
        # Get rid of nans
        ct_axis_filtered_names = ct_axis_filtered_names[~pd.isnull(ct_axis_filtered_names)]
        # Now go through genes in their original order and check if they are in our list of genes
        self.ct_axis_mask = mouse_data.var.index.isin(ct_axis_filtered_names)

        # Find correlated genes between ct_axis_mask and r_axis_mask and remove them from both
        # First remove genes that appear in both masks since they must contain both ct and region information
        intersect_mask = self.r_axis_mask & self.ct_axis_mask
        self.r_axis_mask[intersect_mask] = False
        self.ct_axis_mask[intersect_mask] = False
        # Get raw expression data for leftover relevant ct and region genes
        r_genes_raw = mouse_data.X[:, self.r_axis_mask].toarray()
        ct_genes_raw = mouse_data.X[:, self.ct_axis_mask].toarray()
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
        self.r_axis_mask[r_axis_mask_indices[r_corr_genes]] = False
        self.ct_axis_mask[ct_axis_mask_indices[ct_corr_genes]] = False

        ################################################################################################################
        # At this point we've found the correct mask for the mouse, so we have to transfer it to the chicken
        ################################################################################################################
        chicken_data = sc.read(f'withcolors/chicken_ex_colors.h5ad')
        # Label each observation with its region and species
        chicken_data.obs['clusters'] = chicken_data.obs['clusters'].apply(lambda s: 'C_' + s)
        chicken_data.obs['subregion'] = chicken_data.obs['clusters'].apply(lambda s: s.split('.')[0])
        m_to_c = pd.read_csv('gene_translations.csv')
        # We only care about one to one orthologs
        m_to_c = m_to_c[m_to_c['Mouse homology type'] == 'ortholog_one2one']
        # Get the relevant names of the mouse genes
        mouse_r_gene_names = mouse_data.var.index[self.r_axis_mask]
        mouse_ct_gene_names = mouse_data.var.index[self.ct_axis_mask]
        # Get only the homologous mouse genes
        m_homologs = set(m_to_c[['Mouse gene name', 'Mouse gene stable ID']].to_numpy().flatten())
        h_mouse_r_gene_names = set(mouse_r_gene_names).intersection(m_homologs)
        h_mouse_ct_gene_names = set(mouse_ct_gene_names).intersection(m_homologs)
        # Get the corresponding chicken gene names and IDs
        h_chick_r_gene_names = m_to_c[m_to_c['Mouse gene name'].isin(h_mouse_r_gene_names)]     # Select the rows
        # Get the actual names and IDs
        h_chick_r_gene_names = h_chick_r_gene_names.loc[:, ['Gene stable ID', 'Gene name']].to_numpy().flatten()
        # Remove nans
        h_chick_r_gene_names = h_chick_r_gene_names[~pd.isnull(h_chick_r_gene_names)]
        # Repeat for cell type genes
        h_chick_ct_gene_names = m_to_c[m_to_c['Mouse gene name'].isin(h_mouse_ct_gene_names)]
        h_chick_ct_gene_names = h_chick_ct_gene_names.loc[:, ['Gene stable ID', 'Gene name']].to_numpy().flatten()
        h_chick_ct_gene_names = h_chick_ct_gene_names[~pd.isnull(h_chick_ct_gene_names)]
        # Now, from the scRNA data we collected, find which genes we want
        # These are the actual masks that we want, so set them as well
        self.r_axis_mask = chicken_data.var.index.str.upper().isin(h_chick_r_gene_names)
        self.ct_axis_mask = chicken_data.var.index.str.upper().isin(h_chick_ct_gene_names)

        # Average transcriptomes within each cell type and put into data frame with cell types as rows and genes as cols
        ct_names = np.unique(chicken_data.obs['clusters'])
        ct_avg_data = [chicken_data[chicken_data.obs['clusters'] == ct].X.mean(axis=0) for ct in ct_names]
        self.data = pd.concat([pd.DataFrame(data, columns=chicken_data.var.index, index=[cluster_name])
                               for data, cluster_name in zip(ct_avg_data, np.unique(chicken_data.obs['clusters']))])
        # Divide each row by mean, as in Tosches et al, rename columns,
        # and transpose so that column labels are genes and rows are cell types
        # Divide each row by mean
        self.data = self.data.div(self.data.mean(axis=0).to_numpy(), axis=1)

        # Save data
        data_dict = {'data': self.data, 'ct_axis_mask': self.ct_axis_mask, 'r_axis_mask': self.r_axis_mask}
        with open(f'withcolors_preprocessed/chicken_from_mouse_ex_colors.pickle', mode='wb') as file:
            pickle.dump(data_dict, file)

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
    ct_data_loader = CTDataLoader(reprocess=True)

    agglomerate = Agglomerate3D(
        cell_type_affinity=spearmanr_connectivity,
        linkage_cell='complete',
        linkage_region='homolog_avg',
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
