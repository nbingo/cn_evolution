import sys
sys.path.insert(1, '/Users/noamringach/PycharmProjects/3dtrees')

from agglomerate.batch_agglomerate_3d import BatchAgglomerate3D
from BasicDendrogram3D import CTDataLoader
from metrics.metric_utils import spearmanr_connectivity
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = CTDataLoader(['chicken'], orthologs=False)
    agglomerate = BatchAgglomerate3D(
        linkage_cell=['complete'],
        linkage_region=['homolog_avg'],
        cell_type_affinity=[spearmanr_connectivity],
        max_region_diff=[1],
        region_dist_scale=np.arange(0.5, 1.5, 0.005),
        verbose=False
    )

    agglomerate.agglomerate(data)
    best_agglomerators = agglomerate.get_best_agglomerators()
    pd.options.display.width = 0
    bme_as = best_agglomerators['BME'][1]
    bme_as_scored = [bme_a.compute_tree_score('BME') for bme_a in bme_as]
    best_bme_a = bme_as[np.argmin(bme_as_scored)]
    print(best_bme_a.compute_tree_score('BME'))
    print(best_bme_a.region_dist_scale)
    print(best_bme_a.linkage_mat_readable)
    best_bme_a.view_tree3d()
    # for metric in TREE_SCORE_OPTIONS:
    # for metric in ['BME']:
    #     print(f'Best {metric} score: {best_agglomerators[metric][0]}\n')
    #     for agglomerator in best_agglomerators[metric][1]:
    #         print(f'Agglomerator: {agglomerator}\n'
    #               f'region_dist_scale: {agglomerator.region_dist_scale}\n'
    #               f'Tree:\n{agglomerator.linkage_mat_readable}\n\n')
    #         if input('View tree? ([y]/n): ') == 'y':
    #             agglomerator.view_tree3d()
