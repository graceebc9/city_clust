from src.spectral_fns import load_and_prepare_data

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import os 

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def main():
    output_path = '/Users/gracecolverd/City_clustering/gmm_bic_results'
    
    # input_path = "/Users/gracecolverd/City_clustering/resv3_clustering_data.csv"
    # dataset_name = 'resv3_clustering_data.csv'

    input_path = "/Users/gracecolverd/City_clustering/notebooks/pc_cl.csv"
    dataset_name = 'pl_cl.csv'


    output_path = os.path.join(output_path, dataset_name) 
    os.makedirs(output_path, exist_ok=True) 

    X, cols = load_and_prepare_data(input_path )



    param_grid = {
        "n_components": range(6, 15),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = normalize(X_scaled)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_normalized)

    grid_search.fit(X_principal)



    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    print(df.sort_values(by="BIC score").head())
    df.to_csv(os.path.join(output_path, 'bic_score.csv'))


    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    # save fig 
    plt.savefig(os.path.join(output_path, 'bic_score.png'))
    
if __name__ == "__main__":
    main()