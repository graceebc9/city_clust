import pandas as pd
import glob 
import os  
import geopandas as gpd 



def load_pc_shp(pcs_to_load):

    ll = []
    for pc in pcs_to_load:
        if len(pc)==1:
            path = f'/Volumes/T9/Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291/one_letter_pc_code/{pc}/{pc}.shp'
        else:
            path = f'/Volumes/T9/Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291/two_letter_pc_code/{pc}.shp'
        sd = gpd.read_file(path)    
        ll.append(sd) 
    pc_shp = pd.concat(ll)
    return pc_shp 


def join_pc_map_three_pc(df, df_col,  pc_map  ):
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in ['pcd7', 'pcd8', 'pcds']:
        d = df.merge(pc_map , right_on = col, left_on = df_col  )
        final_d.append(d)
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 


def join_pc_map_three_pc_two(df, df_col1, dfcol2,  pc_map  ):
    # merge on any one of three columns in pc_map 
    final_d = [] 
    for col in ['pcd7', 'pcd8', 'pcds']:
        for dcol in [df_col1, dfcol2]:
            d = pc_map.merge(df, left_on = col, right_on = dcol  )
            final_d.append(d)

    print('starting merge') 
    # Concatenate the results
    merged_final = pd.concat(final_d ).drop_duplicates()
    
    if len(df) != len(merged_final):
        print('Warning: some postcodes not matched')
    return merged_final 


def process_uprn_df(uprn_df):
    # remove trailing space pcds 
    uprn_df['pcds_2'] = uprn_df['PCDS'].str.strip()
    # Check for non-numeric values in the 'UPRN' column
    non_numeric = pd.to_numeric(uprn_df['UPRN'], errors='coerce').isna()
    if non_numeric.any():
        print("Non-numeric values found in 'UPRN' column. These rows will be dropped.")
        # checek how many rows to be dropped 
        if len(uprn_df[non_numeric]) > 1000:
            print('Warning: more than 1000 rows will be dropped')
            raise ValueError('Too many rows to drop')
        # Optionally, handle these rows: drop, fill, etc.
        uprn_df = uprn_df[~non_numeric]
    # Now convert the 'UPRN' column to integers
    uprn_df['UPRN'] = uprn_df['UPRN'].astype(int)
    return uprn_df 


def create_vstreet_lookup(postcode_shapefile_path='/Volumes/T9/Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998/codepoint-poly_5267291', base_dir='/Users/gracecolverd/New_dataset'):
    def run_lk(vstreet_lookup, file):
                # open txt file 
                df= pd.read_csv(file, header=None )
                vstreet_lookup = pd.concat([vstreet_lookup, df])
                return vstreet_lookup 
    
    if os.path.isfile('src/mappings/vstreet_lookup.csv'):
        vstreet_lookup = pd.read_csv('data/mappings/vstreet_lookup.csv')
    else:
        vstreet_lookup = pd.DataFrame() 
        for file in glob.glob(os.path.join(postcode_shapefile_path, 'one_letter_pc_code/*/*lookup.txt') ) :
            vstreet_lookup = run_lk(vstreet_lookup, file)
        for file in glob.glob(os.path.join(postcode_shapefile_path, 'two_letter_pc_code/*lookup.txt') ) :
            vstreet_lookup = run_lk(vstreet_lookup, file)
        vstreet_lookup.to_csv(os.path.join(base_dir, 'src/mapping/vstreet_lookup.csv')) 
        
    return vstreet_lookup



def merge_files_together(folder_glob):

    final = [] 
    for f in folder_glob:
        df = pd.read_csv(f)
        final.append(df)
    final_df = pd.concat(final)
    return final_df


def check_merge_files(df1, df2, col1, col2):
    # Check if the files are empty
    if df1.empty or df2.empty:
        print("Error: One or both files are empty.")
        return False
    
    # Check if the columns to be merged on exist
    if col1 not in df1.columns or col2 not in df2.columns:
        print("Error: One or both columns to be merged on do not exist.")
        return False
    # Check columns are same type 
    if df1[col1].dtype != df2[col2].dtype:
        print('Warning: columns not same type')
    # If one column int, convert toher to int
    
    return True 