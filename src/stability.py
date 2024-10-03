import numpy as np
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime
from .spectral_fns import save_labels , train_spectral_model, load_and_prepare_data
from .gmm import train_kmeans_model 

def generate_random_numbers(x, min_value=0, max_value=100):
    return [random.randint(min_value, max_value) for _ in range(x)]


def main_cluster_randomseed_spectral(output_path, input_path, num_clusters, nrs, affinity):
    np.random.seed(nrs)
    output_path = f"{output_path}/spectral/{num_clusters}/cluster_results/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        X_train, data_cols = load_and_prepare_data(input_path)
        spectral_model_rbf, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz= train_spectral_model(X_train, num_clusters, affinity=affinity, nrs=nrs, norm=True )
        save_labels(output_path, labels_rbf, silhouette_avg, davies_bouldin, calinski_harabasz )
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main_cluster_rs_kmeans(output_path, input_path, num_clusters, nrs):
    np.random.seed(nrs)
    n_init=100
    norm=False 
    output_path = f"{output_path}/kmeans/{num_clusters}/cluster_results/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        X_train, data_cols = load_and_prepare_data(input_path)
        model, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = train_kmeans_model(X_train, num_clusters, nrs, n_init, norm )
        save_labels(output_path, labels_rbf, silhouette_avg, davies_bouldin, calinski_harabasz )
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def load_labels(filepath):
    return pd.read_csv(filepath)['Cluster'].values

def match_clusters(labels1, labels2):
    n_clusters = max(labels1.max(), labels2.max()) + 1
    contingency = pd.crosstab(labels1, labels2)
    _, col_ind = linear_sum_assignment(-contingency.values)
    return dict(enumerate(col_ind))

def remap_labels(labels, mapping):
    return np.array([mapping[label] for label in labels])

def compute_metrics(labels1, labels2):
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    return ari, nmi

def compute_stability(all_labels):
    n_runs = len(all_labels)
    n_cities = len(all_labels[0])
    stability = np.zeros(n_cities)
    
    for i in range(n_cities):
        city_labels = [labels[i] for labels in all_labels]
        stability[i] = len(set(city_labels)) / n_runs
    
    return 1 - stability  # Convert instability to stability

def create_co_occurrence_matrix(all_labels):
    n_cities = len(all_labels[0])
    co_occurrence = np.zeros((n_cities, n_cities))
    
    for labels in all_labels:
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                if labels[i] == labels[j]:
                    co_occurrence[i, j] += 1
                    co_occurrence[j, i] += 1
    
    return co_occurrence / len(all_labels)

def plot_co_occurrence_heatmap(co_occurrence, city_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence, xticklabels=city_names, yticklabels=city_names)
    plt.title("Co-occurrence Matrix Heatmap")
    plt.tight_layout()
    plt.show()


def compute_city_stability(all_labels, ref ):
    n_runs = len(all_labels)
    n_cities = len(all_labels[0])
    reference_labels = all_labels[ref]
    
    # Initialize stability array
    stability = np.zeros(n_cities)
    
    for i in range(0, n_runs):
        if i !=ref:
            labels = all_labels[i]
            mapping = match_clusters(labels, reference_labels)
            remapped_labels = remap_labels(labels, mapping)
            
            # Identify cities that remained in the same cluster
            stable_cities = reference_labels == remapped_labels
            
            # Update stability
            stability[stable_cities] += 1
    
    # Calculate stability as a fraction of runs where the city remained stable
    stability = stability / (n_runs - 1)
    
    return stability

def plot_city_stability(output_path, stability, city_names):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(len(stability))), y=stability)
    plt.title("Individual City Stability Across Runs")
    plt.xlabel("City Index")
    plt.ylabel("Stability (fraction of stable runs)")
    plt.ylim(0, 1)
    plt.tight_layout()
    # save fig 
    plt.savefig(os.path.join(output_path, 'city_stability.png'))


def identify_unstable_cities(stability, city_names, threshold=0.5):
    unstable_cities = [(city, stab) for city, stab in zip(city_names, stability) if stab < threshold]
    return sorted(unstable_cities, key=lambda x: x[1])



def gen_city_stability(city_names, output_path, num_cluster, random_seeds, run_count, model_type, ref_id):
    base_path = os.path.join(output_path, model_type, str(num_cluster), "cluster_results", "random_seed_{}", "labels.csv")

    run_path = os.path.join(output_path, model_type, str(num_cluster),str(ref_id),  'num_runs_'+str(run_count))  
    os.makedirs(run_path, exist_ok=True)    
    all_labels = [load_labels(base_path.format(seed)) for seed in random_seeds]

    stability = compute_city_stability(all_labels, ref_id)
    
    plot_city_stability(run_path, stability, city_names)

    unstable_cities = identify_unstable_cities(stability, city_names, threshold=0.8)

    # mean_stability_list = run_stability_with_diff_ref(all_labels)
    # Mean stability across CV: {np.mean(mean_stability_list):.3f}
   
    summary = f"""
    Spectral Clustering Run Summary
    random seeds:{len(random_seeds)}
    ===============================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Overall average stability: {np.mean(stability):.3f}
    Number of cities with stability < 0.5: {sum(stability < 0.5)}
    Number of cities with stability < 0.8: {sum(stability < 0.8)}


    
    Unstable cities (stability < 0.8)
    {unstable_cities}
   
    """
    
    with open(os.path.join(run_path, 'run_summary.txt'), 'w') as f:
        f.write(summary)


# def run_stability_with_diff_ref(all_labels):

#     def cal_one_stab(j, n_runs, all_labels, reference_labels, n_cities):
#         # Initialize stability array
#         stability = np.zeros(n_cities)
#         for i in range(0, n_runs):
#             if i!=j:
#                 labels = all_labels[i]
#                 mapping = match_clusters(labels, reference_labels)
#                 remapped_labels = remap_labels(labels, mapping)
                
#                 # Identify cities that remained in the same cluster
#                 stable_cities = reference_labels == labels 
                
#                 # Update stability
#                 stability[stable_cities] += 1

#         # Calculate stability as a fraction of runs where the city remained stable
#         stability = stability / (n_runs - 1)
#         return stability

#     n_runs = len(all_labels)
#     n_cities = len(all_labels[0])

#     mean_stability_list=[] 
#     # stab_list = []

#     for i in range(0, n_runs):
#         reference_labels = all_labels[i]
#         stability = cal_one_stab(i, n_runs, all_labels, reference_labels, n_cities)
#         mean_stability_list.append(np.mean(stability)) 
#         # stab_list.append(stability)
    
#     return mean_stability_list   

import os
import numpy as np
from datetime import datetime
import pandas as pd

def gen_city_stability_full_cv(city_names, output_path, num_cluster, random_seeds, run_count, model_type):
    base_path = os.path.join(output_path, model_type, str(num_cluster), "cluster_results", "random_seed_{}", "labels.csv")
    
    run_path = os.path.join(output_path, model_type, str(num_cluster), 'full_cv_analysis')
    os.makedirs(run_path, exist_ok=True)
    
    all_labels = [load_labels(base_path.format(seed)) for seed in random_seeds]
    
    stability_results = []
    
    for ref_id in range(run_count):
        stability = compute_city_stability(all_labels, ref_id)
        stability_results.append(stability)
    
    stability_df = pd.DataFrame(stability_results, columns=city_names)
    
    mean_stability = stability_df.mean()
    std_stability = stability_df.std()
    
    
    
    
    consistently_unstable_cities = identify_consistently_unstable_cities(stability_df, city_names, threshold=0.8)
    
    summary = f"""
    Full Cross-Validation Spectral Clustering Run Summary
    ====================================================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Number of runs: {run_count}
    Random seeds: {len(random_seeds)}
    
    Overall Statistics:
    - Mean stability across all cities and runs: {np.mean(mean_stability):.3f}
    - Standard deviation of stability: {np.mean(std_stability):.3f}
    - Number of cities with mean stability < 0.5: {sum(mean_stability < 0.5)}
    - Number of cities with mean stability < 0.8: {sum(mean_stability < 0.8)}
    
    Consistently Unstable Cities (mean stability < 0.8 in >20% of runs):
    {consistently_unstable_cities}
    """
    
    with open(os.path.join(run_path, 'full_cv_summary.txt'), 'w') as f:
        f.write(summary)


    
    stability_df.to_csv(os.path.join(run_path, 'detailed_stability_results.csv'), index=False)
    mean_stability.to_csv(os.path.join(run_path, 'mean_stability_by_city.csv'))
    # plot_city_stability(run_path, mean_stability.values, city_names, title="Mean Stability Across All Runs")

def identify_consistently_unstable_cities(stability_df, city_names, threshold=0.8, frequency_threshold=0.2):
    unstable_frequency = (stability_df < threshold).mean()
    consistently_unstable = unstable_frequency[unstable_frequency > frequency_threshold]
    return consistently_unstable.sort_values(ascending=False).to_string()

# Assuming these functions are defined elsewhere:
# load_labels, compute_city_stability, plot_city_stability