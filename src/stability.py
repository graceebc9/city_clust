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

def generate_random_numbers(x, min_value=0, max_value=100):
    return [random.randint(min_value, max_value) for _ in range(x)]


def main_cluster_randomseed(output_path, input_path, num_clusters, nrs):
    np.random.seed(nrs)
    output_path = f"{output_path}/{num_clusters}/cluster_results/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        X_train, data_cols = load_and_prepare_data(input_path)
        spectral_model_rbf, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = train_spectral_model(X_train, num_clusters)
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


def compute_city_stability(all_labels):
    n_runs = len(all_labels)
    n_cities = len(all_labels[0])
    reference_labels = all_labels[0]
    
    # Initialize stability array
    stability = np.zeros(n_cities)
    
    for i in range(1, n_runs):
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



def gen_city_stability(city_names, output_path, num_cluster, random_seeds, run_count):
    base_path = os.path.join(output_path, str(num_cluster), "cluster_results", "random_seed_{}", "labels.csv")

    run_path = os.path.join(output_path, str(num_cluster), 'num_runs_'+str(run_count))  
    os.makedirs(run_path, exist_ok=True)    
    all_labels = [load_labels(base_path.format(seed)) for seed in random_seeds]

    stability = compute_city_stability(all_labels)
    
    plot_city_stability(run_path, stability, city_names)

    unstable_cities = identify_unstable_cities(stability, city_names, threshold=0.8)



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

