from matplotlib import pyplot as plt
from matplotlib import rcParams

from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

rcParams.update({'figure.autolayout': True})

import pandas as pd
import numpy as np
from typing import List, Optional

P_VAL_ADJ_THRESH = 0.01
AVG_LOG_FC_THRESH = 2


def read_data(species: List[str], deg_species: Optional[List[str]] = None, preprocess: Optional[bool] = True):
    if deg_species is None:
        deg_species = species

    species_degs = [pd.read_csv(f'TranscriptomeData/Nomi_{s}DEGs.csv', header=0, index_col=0) for s in deg_species]
    species_data = [pd.read_csv(f'TranscriptomeData/Nomi_{s}allaverage.csv', header=0, index_col=0) for s in species]

    if preprocess:
        # Filter for adjusted p-value and logFC
        for i in range(len(species_degs)):
            species_degs[i] = species_degs[i].loc[(species_degs[i]['p_val_adj'] < P_VAL_ADJ_THRESH) &
                                                  (species_degs[i]['avg_logFC'] > AVG_LOG_FC_THRESH)]

        # Find the common DEGs
        common_genes = species_degs[0]['gene']
        for deg in species_degs:
            common_genes = np.intersect1d(common_genes, deg['gene'])

        # Filter for common DEGs
        # Divide each row by mean, as in Tosches et al, rename columns,
        # and transpose so that column labels are genes and rows are cell types
        for i in range(len(species_data)):
            # Filter for common DEGs
            species_data[i] = species_data[i].loc[species_data[i].index.isin(common_genes)]
            # Divide each row by mean
            species_data[i] = species_data[i].div(species_data[i].mean(axis=1).values, axis=0)
            # Rename columns with species prefix and transpose so cell types are rows and genes are columns
            species_data[i] = species_data[i].add_prefix(f'{species[i][0]}_'.upper()).transpose()

    # Concatenate all the data
    return pd.concat(species_data)


def get_region(ct_name: str):
    return ct_name[0] + '_' + ct_name.split('.')[1]


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=np.max(linkage_matrix), **kwargs)


if __name__ == '__main__':
    data = read_data(['chicken'])


    def spearmanr_connectivity(x):
        # data_ct is assumed to be (n_variables, n_examples)
        rho, _ = spearmanr(x, axis=1)
        return 1 - rho


    agglomerate = AgglomerativeClustering(
        affinity=spearmanr_connectivity,
        linkage='complete',
        n_clusters=None,
        distance_threshold=0
    )
    agglomerate.fit(data.to_numpy())

    plt.title('Hierarchical clustering')
    plot_dendrogram(agglomerate, truncate_mode='level', labels=data.index.to_numpy())
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('dendrogram.pdf')
    plt.show()
