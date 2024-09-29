import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture

from src.stability import load_labels, match_clusters, remap_labels, compute_metrics, compute_stability, create_co_occurrence_matrix    

# Abstract base class for clustering models
class ClusteringModel(ABC):
    @abstractmethod
    def train(self, X_train, num_clusters):
        pass

    @abstractmethod
    def get_labels(self):
        pass

class SpectralClusteringModel(ClusteringModel):
    def train(self, X_train, num_clusters, rs, affinity='rbf'):
        # set random seed 
        np.random.seed(rs)
        # Implement spectral clustering logic here
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_normalized = normalize(X_scaled)
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(X_normalized)
    
         if affinity == 'nearest_neighbors':
            # For nearest_neighbors, we need to specify the number of neighbors
            spectral = SpectralClustering(n_clusters=num_clusters, affinity=affinity, n_neighbors=10)
        else:
            spectral = SpectralClustering(n_clusters=num_clusters, affinity=affinity)
        
        labels_rbf = spectral.fit_predict(X_principal)
        
        silhouette_avg = silhouette_score(X_principal, labels_rbf)
        davies_bouldin = davies_bouldin_score(X_principal, labels_rbf)
        calinski_harabasz = calinski_harabasz_score(X_principal, labels_rbf)
        
        return spectral, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz


    def get_labels(self):
        # Return the labels
        pass

class KMeansModel(ClusteringModel):
    def train(self, X_train, num_clusters, rs ):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_normalized = normalize(X_scaled)
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(X_normalized)
        
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=rs)
        labels = kmeans_model.fit_predict(X_principal)
        
        silhouette_avg = silhouette_score(X_principal, labels)
        davies_bouldin = davies_bouldin_score(X_principal, labels)
        calinski_harabasz = calinski_harabasz_score(X_principal, labels)
        
        return kmeans_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz

    def get_labels(self):
        # Return the labels
        pass

class GMMModel(ClusteringModel):
    def train(self, X_train, num_clusters, rs ) :
        # Implement GMM clustering logic here
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(X_scaled)
        
        gmm_model = GaussianMixture(n_components=num_components, random_state=rs)
        labels = gmm_model.fit_predict(X_principal)
        self.labels = labels
        
        silhouette_avg = silhouette_score(X_principal, labels)
        davies_bouldin = davies_bouldin_score(X_principal, labels)
        calinski_harabasz = calinski_harabasz_score(X_principal, labels)
        
        return gmm_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz

    def get_labels(self):
        # Return the labels
        return self.labels
        


def load_and_prepare_data(input_path):
    data = pd.read_csv(input_path)
    data = data[data['TCITY15NM'] != 'London'].copy()
    X = data.drop(columns=['TCITY15NM'])
    data_cols = X.columns.tolist()
    return X, data_cols

def save_labels(output_path, labels , silhouette_avg, davies_bouldin, calinski_harabasz):
    pd.DataFrame(labels, columns=['Cluster']).to_csv(os.path.join(output_path, 'labels.csv'), index=False)
    # Create and save summary report
    summary = f"""
    Spectral Clustering Run Summary
    ===============================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Clustering Metrics:
    - Silhouette Score: {silhouette_avg:.4f}
    - Davies-Bouldin Index: {davies_bouldin:.4f}
    - Calinski-Harabasz Index: {calinski_harabasz:.4f}
    

    
    Files saved:
    - spectral_model_rbf.pkl: Trained Spectral Clustering model
    - scaler.pkl: StandardScaler object
    - pca.pkl: PCA object
    - labels.csv: Cluster labels for each data point
    - X_principal.csv: PCA-transformed data
    - ecdf_all_distributions.png: ECDF plots for all variables
    - kde_all_distributions.png: KDE plots for all variables

    """
    
    with open(os.path.join(output_path, 'run_summary_metrics.txt'), 'w') as f:
        f.write(summary)

def main_cluster_randomseed(output_path, input_path, num_clusters, nrs, model_class):
    np.random.seed(nrs)
    output_path = f"{output_path}/{model_class}/{num_clusters}/cluster_results/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        X_train, data_cols = load_and_prepare_data(input_path)
        model = model_class()
        if model_class == 'SpectralClusteringModel':
            model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = model.train(X_train, num_clusters, rs ,affinity )  
        else:
            model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = model.train(X_train, num_clusters, rs )
    
        
        save_labels(output_path, labels, silhouette_avg, davies_bouldin, calinski_harabasz)
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def gen_city_stability(city_names, output_path, num_cluster, random_seeds, run_count, model_class):
    base_path = os.path.join(output_path, m_class, str(num_cluster), "cluster_results", "random_seed_{}", "labels.csv")

    run_path = os.path.join(output_path, str(num_cluster), 'num_runs_'+str(run_count))  
    os.makedirs(run_path, exist_ok=True)    
    all_labels = [load_labels(base_path.format(seed)) for seed in random_seeds]

    stability = compute_city_stability(all_labels)
    
    plot_city_stability(run_path, stability, city_names)

    unstable_cities = identify_unstable_cities(stability, city_names, threshold=0.8)

    summary = f"""
    Clustering Run Summary
    Algorithm: {model_class.__name__}
    Random seeds: {len(random_seeds)}
    ===============================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Overall average stability: {np.mean(stability):.3f}
    Number of cities with stability < 0.5: {sum(stability < 0.5)}
    Number of cities with stability < 0.8: {sum(stability < 0.8)}
    
    Unstable cities (stability < 0.8):
    {unstable_cities}
    """
    
    with open(os.path.join(run_path, 'run_summary.txt'), 'w') as f:
        f.write(summary)

# Example usage
if __name__ == "__main__":
    # For Spectral Clustering
    main_cluster_randomseed("output_path", "input_path", 5, 42, SpectralClusteringModel)
    
    # For K-means
    # main_cluster_randomseed("output_path", "input_path", 5, 42, KMeansModel)
    
    # For GMM
    # main_cluster_randomseed("output_path", "input_path", 5, 42, GMMModel)