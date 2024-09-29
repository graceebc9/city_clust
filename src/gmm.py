import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.spectral_fns import load_and_prepare_data

def train_gmm_model(X_train, num_components, rs, n_init ):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_scaled)
    
    gmm_model = GaussianMixture(n_components=num_components, random_state=rs, n_init=n_init)
    labels = gmm_model.fit_predict(X_principal)
    
    silhouette_avg = silhouette_score(X_principal, labels)
    davies_bouldin = davies_bouldin_score(X_principal, labels)
    calinski_harabasz = calinski_harabasz_score(X_principal, labels)
    
    return gmm_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz

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
        plt.title(f'ECDF of {col} by Cluster')
        plt.xlabel(col)
        plt.ylabel('ECDF')
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

def save_results_full(output_path, run_name, gmm_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin, calinski_harabasz):
    run_path = output_path
    os.makedirs(run_path, exist_ok=True)
    
    joblib.dump(gmm_model, os.path.join(run_path, 'gmm_model.pkl'))
    joblib.dump(scaler, os.path.join(run_path, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(run_path, 'pca.pkl'))
    
    # Save labels
    pd.DataFrame(labels, columns=['Cluster']).to_csv(os.path.join(run_path, 'labels.csv'), index=False)
    
    # Save PCA results
    pd.DataFrame(X_principal, columns=['PC1', 'PC2']).to_csv(os.path.join(run_path, 'X_principal.csv'), index=False)
    
    # Create and save summary report
    summary = f"""
    GMM Clustering Run Summary
    ==========================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Clustering Metrics:
    - Silhouette Score: {silhouette_avg:.4f}
    - Davies-Bouldin Index: {davies_bouldin:.4f}
    - Calinski-Harabasz Index: {calinski_harabasz:.4f}
    
    Files saved:
    - gmm_model.pkl: Trained GMM model
    - scaler.pkl: StandardScaler object
    - pca.pkl: PCA object
    - labels.csv: Cluster labels for each data point
    - X_principal.csv: PCA-transformed data
    - ecdf_all_distributions.png: ECDF plots for all variables
    - kde_all_distributions.png: KDE plots for all variables
    """
    
    with open(os.path.join(run_path, 'run_summary.txt'), 'w') as f:
        f.write(summary)

def save_labels(output_path, labels, silhouette_avg, davies_bouldin, calinski_harabasz):
    pd.DataFrame(labels, columns=['Cluster']).to_csv(os.path.join(output_path, 'labels.csv'), index=False)
    summary = f"""
    GMM Clustering Run Summary
    ==========================
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Clustering Metrics:
    - Silhouette Score: {silhouette_avg:.4f}
    - Davies-Bouldin Index: {davies_bouldin:.4f}
    - Calinski-Harabasz Index: {calinski_harabasz:.4f}
    """
    
    with open(os.path.join(output_path, 'run_summary_metrics.txt'), 'w') as f:
        f.write(summary)

def run_cluster_gmm(output_path, input_path, num_components, nrs, n_init):
    output_path = f"{output_path}/GMM/{num_components}/random_seed_{nrs}"    
    os.makedirs(output_path, exist_ok=True) 
    dataset_name = os.path.basename(input_path).split('_')[0]
    run_name = dataset_name
    
    try:
        X_train, data_cols = load_and_prepare_data(input_path)
        gmm_model, scaler, pca, labels, X_principal, silhouette_avg, davies_bouldin,calinski_harabasz = train_gmm_model(X_train, num_components, nrs, n_init)
        save_results_full(output_path, run_name, gmm_model, scaler, pca, labels, X_principal, 
                     silhouette_avg, davies_bouldin, calinski_harabasz)
        plot_variable_distributions(X_train, labels, os.path.join(output_path, run_name), data_cols)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
# run_cluster('/path/to/output', '/path/to/input.csv', num_components=5, nrs=42)