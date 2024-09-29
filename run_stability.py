from src.stability import main_cluster_randomseed, gen_city_stability
import pandas as pd 
input_path = "/Users/gracecolverd/City_clustering/resv3_clustering_data.csv" 
num_clusters = 12 

runs=400
numbers = list(range(1, runs+1))
output_path = '/Users/gracecolverd/City_clustering/clustering_results'

for num_clusters in [6,7,8,9,12]:
# for num_clusters in [10,11]:
    for nrs in numbers:
        main_cluster_randomseed(output_path, input_path, num_clusters, nrs)

    df = pd.read_csv(input_path)    
    city_names = df['TCITY15NM'].tolist() 
    gen_city_stability(city_names, output_path, num_clusters ,  numbers, runs)