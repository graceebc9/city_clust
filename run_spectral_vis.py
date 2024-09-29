from src.spectral_fns import run_cluster    
from src.gmm import run_cluster_gmm 
import pandas as pd 
input_path = "/Users/gracecolverd/City_clustering/resv3_clustering_data.csv" 
output_path = '/Users/gracecolverd/City_clustering/vis'

list_clusters = [5,6,7,8,9,10,11,12]
nrs =  1

# # run spectral 
# for num_clusters in list_clusters:
#     for affinity in ['rbf', 'nearest_neighbors']:
#         run_cluster(output_path, input_path, num_clusters, nrs, affinity)

# run GMM
input_path='/Users/gracecolverd/City_clustering/notebooks/pc_cl.csv' 
output_path = '/Users/gracecolverd/City_clustering/postcode_results'
for num_clusters in list_clusters: 
    print(num_clusters)
    run_cluster_gmm(output_path, input_path, num_clusters, nrs, n_init=1)

