
import pandas as pd 
import numpy as np 



    
typ_cols = [ '2 storeys terraces with t rear extension_pct',
 '3-4 storey and smaller flats_pct',
 'Domestic outbuilding_pct',
 'Large detached_pct',
 'Large semi detached_pct',
 'Linked and step linked premises_pct',
 'Medium height flats 5-6 storeys_pct',
 
 'Planned balanced mixed estates_pct',
 'Semi type house in multiples_pct',
 'Small low terraces_pct',
 'Standard size detached_pct',
 'Standard size semi detached_pct',
 'Tall flats 6-15 storeys_pct',
 'Tall terraces 3-4 storeys_pct',
 'Very large detached_pct',
 'Very tall point block flats_pct']

def compute_entropy(probs):
    """Compute the Shannon entropy of a probability distribution."""
    return -np.sum(probs * np.log2(probs + 1e-9))

def calculate_typology_entropy(df, typology_columns):
    # Group by city
    city_groups = df.groupby('TCITY15CD')

    # Initialize a list to store entropy values
    entropy_values = []

    # Calculate entropy for each city
    for city, group in city_groups:
        # Calculate the mean typology mix for the city
        typology_means = group[typology_columns].mean()
        
        # Normalize the typology means to get a probability distribution
        typology_probs = typology_means / typology_means.sum()
        
        # Compute the entropy for the city
        entropy = compute_entropy(typology_probs)
        
        # Append the entropy value with the city name
        entropy_values.append({'TCITY15CD': city, 'entropy': entropy})
    
    # Convert the entropy values list to a DataFrame
    entropy_df = pd.DataFrame(entropy_values)

    # Merge the entropy values back into the original DataFrame
    df = df.merge(entropy_df, on='TCITY15CD',  how='left')

    return df



def load_clusters(n, city , folder, typ):
    if typ == 'old':
        path = f'/Volumes/T9/Data_downloads/new-data-outputs/ml_results/{folder}/{city}_spectral_clustering_{n}/labels.csv'
    else:
        path = f'/Volumes/T9/Data_downloads/new-data-outputs/ml_results/citycl/{city}_clust_clusters_{n}/{city}_spectral_clustering_{n}/labels.csv'
    res = pd.read_csv(f'/Users/gracecolverd/New_dataset/ml_scripts/{city}_clust.csv' )
    labels = pd.read_csv(path)
    if len(res)!= len(labels):
        print('err')
        # raise Exception(mismatch in data)
    city_cluster = pd.concat([res, labels], axis=1)
    return city_cluster 



# def load_clusters(n, city , folder, typ):
#     if typ == 'old':
#         path = f'/Volumes/T9/Data_downloads/new-data-outputs/ml_results/{folder}/{city}_spectral_clustering_{n}/labels.csv'
#     else:
#         path = f'/Volumes/T9/Data_downloads/new-data-outputs/ml_results/{folder}/{city}_city_clust_clusters_{n}/{city}_spectral_clustering_{n}/labels.csv'
#     res = pd.read_csv(f'/Users/gracecolverd/New_dataset/ml_scripts/{folder}/{city}_city_clust.csv' )
#     labels = pd.read_csv(path)
#     if len(res)!= len(labels):
#         print('err')
#         # raise Exception(mismatch in data)
#     city_cluster = pd.concat([res, labels], axis=1)
#     return city_cluster 


def load_bsdata(data):
    lk = pd.read_csv('/Volumes/T9/2024_Data_downloads/lookups/pcs_to_oa_mapping_census2021/PCD_OA21_LSOA21_MSOA21_LAD_AUG23_UK_LU.csv', encoding='latin1')
    oa_lsoa = pd.read_csv('/Volumes/T9/2024_Data_downloads/lookups/OAs_to_LSOAs_to_MSOAs_to_LEP_to_LAD_(May_2022)_Lookup_in_England.csv')

    lsoa2011_lsoa_2012 = pd.read_csv('/Volumes/T9/2024_Data_downloads/lookups/lsoa2011-2021/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Best_Fit_Lookup_for_EW_(V2).csv')
    lsoa_city = pd.read_csv('/Volumes/T9/2024_Data_downloads/lookups/lsoa_city_2011/Lower_Layer_Super_Output_Area_(2011)_to_Major_Towns_and_Cities_(December_2015)_Lookup_in_England_and_Wales.csv')
    ls21_city = lsoa_city.merge(lsoa2011_lsoa_2012, on=['LSOA11CD'])[['LSOA11CD', 'LSOA21CD', 'TCITY15CD' , 'TCITY15NM']].copy() 

    data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True )

    city_data = data.merge(ls21_city, left_on='lsoa11cd', right_on ='LSOA11CD')
    outp= calculate_typology_entropy(city_data, typ_cols)

    wrk = outp[~outp['total_gas'].isna()].copy() 
    wrk['av_eui_h'] = wrk['total_gas'] / wrk['all_res_heated_vol_h_total']
    wrk['perc_diff_meters'] = wrk.diff_gas_meters_uprns_res / wrk.num_meters_gas

    return city_data,  wrk 


import os
import pandas as pd
# from tabulate import tabulate

