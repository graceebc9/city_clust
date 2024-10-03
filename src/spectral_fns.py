import os 
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



def load_and_prepare_data(input_path):
    data = pd.read_csv(input_path)
    data = data[data['TCITY15NM'] != 'London'].copy()
    X = data.drop(columns=['TCITY15NM'])
    data_cols = X.columns.tolist()
    return X, data_cols


def train_clustering_model(X_train, num_clusters, algorithm='kmeans', affinity='rbf', linkage='ward', 
                           metric='euclidean', norm=True,  random_state=42, n_init=10):
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    if norm:
        X_scaled = normalize(X_scaled)
    
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_scaled)
    
    # Clustering
    if algorithm.lower() == 'kmeans':
        model = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=n_init)
    elif algorithm.lower() == 'spectral':
        model = SpectralClustering(n_clusters=num_clusters, affinity=affinity, 
                                   random_state=random_state, n_init=n_init)
    elif algorithm.lower() == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage, metric=metric)
    elif algorithm.lower() == 'gmm':
        model = GaussianMixture(n_components=num_components, random_state=rs, n_init=n_init)
    else:
        raise ValueError("Unsupported algorithm. Choose 'kmeans', 'spectral', or 'hierarchical'.")
    
    labels = model.fit_predict(X_principal)
    
    # Evaluation Metrics
    silhouette_avg = silhouette_score(X_principal, labels)
    davies_bouldin = davies_bouldin_score(X_principal, labels)
    calinski_harabasz = calinski_harabasz_score(X_principal, labels)
    
    return model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz


# def train_spectral_model(X_train, num_clusters, affinity, nrs, norm=True, n_init =10, pca_comp =2 ):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     if norm ==True:
#         X_scaled = normalize(X_scaled)
#     pca = PCA(n_components=pca_comp)
#     X_principal = pca.fit_transform(X_scaled)
    
#     spectral_model_rbf = SpectralClustering(n_clusters=num_clusters, affinity=affinity, random_state=nrs, n_init=n_init)
#     labels_rbf = spectral_model_rbf.fit_predict(X_principal)
    
#     silhouette_avg = silhouette_score(X_principal, labels_rbf)
#     davies_bouldin = davies_bouldin_score(X_principal, labels_rbf)
#     calinski_harabasz = calinski_harabasz_score(X_principal, labels_rbf)
    

#     return spectral_model_rbf, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz

# def train_hierarchical_model(X_train, num_clusters, linkage='ward', metric='euclidean', norm=True, pca_comp=2):

#     # Preprocessing
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
    
#     if norm:
#         X_scaled = normalize(X_scaled)
    
#     pca = PCA(n_components=pca_comp)
#     X_principal = pca.fit_transform(X_scaled)
    
#     # Hierarchical Clustering
#     hierarchical_model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage, metric=metric)
#     labels = hierarchical_model.fit_predict(X_principal)
    
#     # Evaluation Metrics
#     silhouette_avg = silhouette_score(X_principal, labels)
#     davies_bouldin = davies_bouldin_score(X_principal, labels)
#     calinski_harabasz = calinski_harabasz_score(X_principal, labels)
    
#     return hierarchical_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz





# def train_gmm_model(X_train, num_components, rs, n_init, norm ):
#     # scaler = StandardScaler()s
#     scaler=None
#     pca=None
#     # X_scaled = scaler.fit_transform(X_train)
#     if norm ==True:
#         X_scaled = normalize(X_scaled)
#     # pca = PCA(n_components=2)
#     # X_principal = pca.fit_transform(X_train)
#     X_principal= X_train
#     gmm_model = GaussianMixture(n_components=num_components, random_state=rs, n_init=n_init)
#     labels = gmm_model.fit_predict(X_principal)
    
#     silhouette_avg = silhouette_score(X_principal, labels)
#     davies_bouldin = davies_bouldin_score(X_principal, labels)
#     calinski_harabasz = calinski_harabasz_score(X_principal, labels)
    
#     return gmm_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz



# def train_kmeans_model(X_train, num_components, rs, n_init, norm ):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     if norm ==True:
#         X_scaled = normalize(X_scaled)
#     pca = PCA(n_components=2)
#     X_principal = pca.fit_transform(X_scaled)
    
#     model = KMeans(n_clusters=num_components, random_state=rs, n_init=n_init)
#     labels = model.fit_predict(X_principal)
    
#     silhouette_avg = silhouette_score(X_principal, labels)
#     davies_bouldin = davies_bouldin_score(X_principal, labels)
#     calinski_harabasz = calinski_harabasz_score(X_principal, labels)
    
#     return model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz





def plot_variable_distributions(X_train, labels, run_path, cols):
    df = pd.DataFrame(X_train, columns=cols)
    df['Cluster'] = labels
    
    plots_folder = os.path.join(run_path, 'variable_distributions')
    os.makedirs(plots_folder, exist_ok=True)
    
    num_cols = 2
    num_rows = (len(cols) + num_cols - 1) // num_cols
    
    # ECDF Plot
    plt.figure(figsize=(11, 5 * num_rows))
    for i, col in enumerate(cols):
        plt.subplot(num_rows, num_cols, i + 1)
        for cluster in np.unique(labels):
            cluster_data = df[df['Cluster'] == cluster][col].dropna()
            sns.ecdfplot(cluster_data, label=f'Cluster {cluster}')
        plt.title(str(col), fontsize = 18)
        plt.xlabel(col, fontsize=16)
        plt.ylabel('ECDF',  fontsize=16)
        plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'ecdf_all_distributions.png'))
    plt.close()
    
    # KDE Plot
    plt.figure(figsize=(11, 5 * num_rows))
    for i, col in enumerate(cols):
        plt.subplot(num_rows, num_cols, i + 1)
        for cluster in np.unique(labels):
            cluster_data = df[df['Cluster'] == cluster][col].dropna()
            sns.kdeplot(cluster_data, fill=True, label=f'Cluster {cluster}')
        plt.title(f'KDE of {col} by Cluster')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'kde_all_distributions.png'))
    plt.close()



# def save_results_full(output_path, run_name, spectral_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz):
#     run_path = output_path
#     os.makedirs(run_path, exist_ok=True)
    
#     joblib.dump(spectral_model, os.path.join(run_path, 'spectral_model_rbf.pkl'))
#     joblib.dump(scaler, os.path.join(run_path, 'scaler.pkl'))
#     joblib.dump(pca, os.path.join(run_path, 'pca.pkl'))
    
#     # Save labels
#     pd.DataFrame(labels, columns=['Cluster']).to_csv(os.path.join(run_path, 'labels.csv'), index=False)
    
#     # Save PCA results
#     pd.DataFrame(X_principal, columns=['PC1', 'PC2']).to_csv(os.path.join(run_path, 'X_principal.csv'), index=False)
    
#     # Create and save summary report
#     summary = f"""
#     Spectral Clustering Run Summary
#     ===============================
#     Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
#     Clustering Metrics:
#     - Silhouette Score: {silhouette_avg:.4f}
#     - Davies-Bouldin Index: {davies_bouldin:.4f}
#     - Calinski-Harabasz Index: {calinski_harabasz:.4f}
    

    
#     Files saved:
#     - spectral_model_rbf.pkl: Trained Spectral Clustering model
#     - scaler.pkl: StandardScaler object
#     - pca.pkl: PCA object
#     - labels.csv: Cluster labels for each data point
#     - X_principal.csv: PCA-transformed data
#     - ecdf_all_distributions.png: ECDF plots for all variables
#     - kde_all_distributions.png: KDE plots for all variables

#     """
    
#     with open(os.path.join(run_path, 'run_summary.txt'), 'w') as f:
#         f.write(summary)
 


def save_results_full(output_path, run_name, spectral_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz):
    run_path = output_path
    os.makedirs(run_path, exist_ok=True)
    
    joblib.dump(spectral_model, os.path.join(run_path, 'spectral_model_rbf.pkl'))
    joblib.dump(scaler, os.path.join(run_path, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(run_path, 'pca.pkl'))
    
    # Save labels
    pd.DataFrame(labels, columns=['Cluster']).to_csv(os.path.join(run_path, 'labels.csv'), index=False)
    
    # Save PCA results
    pca_columns = [f'PC{i+1}' for i in range(X_principal.shape[1])]
    pd.DataFrame(X_principal, columns=pca_columns).to_csv(os.path.join(run_path, 'X_principal.csv'), index=False)
    
    # Create and save summary report
    summary = f"""
    Spectral Clustering Run Summary
    ===============================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Clustering Metrics:
    - Silhouette Score: {silhouette_avg:.4f}
    - Davies-Bouldin Index: {davies_bouldin:.4f}
    - Calinski-Harabasz Index: {calinski_harabasz:.4f}
    
    PCA Components: {X_principal.shape[1]}
    
    Files saved:
    - spectral_model_rbf.pkl: Trained Spectral Clustering model
    - scaler.pkl: StandardScaler object
    - pca.pkl: PCA object
    - labels.csv: Cluster labels for each data point
    - X_principal.csv: PCA-transformed data
    - ecdf_all_distributions.png: ECDF plots for all variables
    - kde_all_distributions.png: KDE plots for all variables
    """
    
    with open(os.path.join(run_path, 'run_summary.txt'), 'w') as f:
        f.write(summary)

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
    
    
# def run_cluster(output_path, input_path, num_clusters, nrs, affinity, norm, n_init, pca_comp=2 ):
#     # set random seed
#     np.random.seed(nrs)

#     output_path = f"{output_path}/spectral/{num_clusters}/{affinity}/{n_init}/{pca_comp}/random_seed_{nrs}"    
#     os.makedirs(output_path, exist_ok=True) 
#     dataset_name = os.path.basename(input_path).split('_')[0]
#     run_name = dataset_name
    
#     try:
#         X_train, data_cols = load_and_prepare_data(input_path)
#         spectral_model_rbf, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz =train_spectral_model(X_train, num_clusters, affinity, nrs, norm, n_init, pca_comp)
#         save_results_full(output_path, run_name, spectral_model_rbf, scaler, pca, labels_rbf, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz,)
#         plot_variable_distributions(X_train, labels_rbf, os.path.join(output_path, run_name), data_cols)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

 

def run_cluster(output_path, input_path, num_clusters, nrs, algorithm='kmeans', affinity='rbf', 
                linkage='ward', metric='euclidean', norm=True, n_init=10):
    # Set random seed
    np.random.seed(nrs)

    # Create output path
    output_path = f"{output_path}/{algorithm}/{num_clusters}/{affinity if algorithm == 'spectral' else linkage}/{n_init}/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 

    # Get dataset name
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        # Load and prepare data
        X_train, data_cols = load_and_prepare_data(input_path)

        # Run clustering
        model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz = train_clustering_model(
            X_train, num_clusters, algorithm=algorithm, affinity=affinity, linkage=linkage, 
            metric=metric, norm=norm,   random_state=nrs, n_init=n_init
        )

        # Save results
        save_results_full(output_path, run_name, model, scaler, pca, labels, X_principal, 
                          silhouette_avg, davies_bouldin, calinski_harabasz)

        # Plot variable distributions
        plot_variable_distributions(X_train, labels, os.path.join(output_path, run_name), data_cols)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Assuming these functions are defined elsewhere in your code:
# load_and_prepare_data()
# save_results_full()
# plot_variable_distributions()