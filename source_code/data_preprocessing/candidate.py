# --- Core Libraries ---
import pandas as pd
import numpy as np
import os
from functools import reduce
import time
import warnings
import random # For cluster visualization colors

# --- Geospatial & Clustering ---
import geopandas as gpd
from shapely.geometry import Point
try:
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist # For tie-breaker distance calculation
    sklearn_present = True
except ImportError:
    print("Warning: scikit-learn or scipy not found. Clustering and final selection will be skipped.")
    print("Install with: pip install scikit-learn scipy")
    sklearn_present = False

# --- Visualization ---
try:
    import folium
    from folium.plugins import MarkerCluster # Optional for cleaner maps
    folium_present = True
except ImportError:
    print("Warning: folium not found. Visualization steps will be skipped.")
    print("Install with: pip install folium")
    folium_present = False

# --- Optional for Progress Bar ---
try:
    from tqdm import tqdm
    tqdm_present = True
except ImportError:
    tqdm_present = False
    # Dummy tqdm function if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        print("Processing...")
        return iterable

# --- Suppress specific warnings ---
warnings.filterwarnings('ignore', category=FutureWarning, module='geopandas*')
# Suppress SettingWithCopyWarning (use .copy() where appropriate to avoid it)
pd.options.mode.chained_assignment = None # default='warn'

# --- 1. Configuration & File Paths ---
# Base path (adjust if your project structure differs)
base_data_path = r"D:\project\HDT_EVCS_Opt\Data"

# Input file paths
od_file_path = os.path.join(base_data_path, r"Processed_Data\candidate\criteria\OD_criteria.csv")
rest_area_file_path = os.path.join(base_data_path, r"Processed_Data\candidate\criteria\rest_area_criteria.csv")
traffic_file_path = os.path.join(base_data_path, r"Processed_Data\candidate\criteria\traffic_critera.csv")
infra_file_path = os.path.join(base_data_path, r"Processed_Data\candidate\criteria\infra_criteria.xlsx") # Requires openpyxl
interval_file_path = os.path.join(base_data_path, r"Processed_Data\candidate\criteria\interval_criteria.csv")
link_to_emd_file = os.path.join(base_data_path, r"Raw_Data\Metropolitan area\LINK_ID_TO_EMD_CODE.csv") # For Sigungu mapping
# area_id_file = os.path.join(base_data_path, r"Raw_Data\Metropolitan area\AREA_ID_DATASET.csv") # No longer needed
SHAPEFILE_PATH = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\main_road_network_level_5.5\level5_5_link_probe_32_2020.shp" # For coordinates

# Output directory paths
output_base_path = os.path.join(base_data_path, r"Processed_Data\candidate") # Base directory for output
preliminary_output_dir_name = "Preliminary_Candidates" # Folder for intermediate/general outputs
final_output_dir_name = "Final_Candidates" # Folder for final selected outputs (English name)

preliminary_output_dir = os.path.join(output_base_path, preliminary_output_dir_name)
final_output_dir = os.path.join(output_base_path, final_output_dir_name)


# Clustering configuration
candidate_num = 500 # Final number of candidates (K-Means clusters)
TARGET_COUNT_FILTER = 3 * candidate_num # Target for initial filtering steps (OD, Traffic, Infra)
KMEANS_RANDOM_STATE = 42 # Seed for K-Means reproducibility

# Visualization Colors
VIS_COLORS_OD_TRAFFIC = {1: 'blue', 2: 'green', 3: 'yellow', 4: 'red'} # For OD/Traffic score
VIS_COLORS_OTHERS = 'red' # For Infra/Interval/RestArea individual viz
VIS_COLORS_FINAL = {1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'red'} # For final candidates by 'point'
# Define a list of distinct colors for cluster visualization (will repeat for k=500)
CLUSTER_COLORS = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']


# --- 2. Helper Functions ---

def get_midpoint(geom):
    """Calculates midpoint (latitude, longitude) of a LineString geometry."""
    if geom and geom.geom_type == 'LineString' and not geom.is_empty:
        try:
            if not geom.is_valid: geom = geom.buffer(0)
            if not geom.is_valid or geom.is_empty or geom.geom_type != 'LineString': return None, None
            mid_point = geom.interpolate(0.5, normalized=True)
            if mid_point.geom_type == 'Point': return mid_point.y, mid_point.x
            else: return None, None
        except Exception: return None, None
    return None, None

def load_link_coordinates(shapefile_path, target_crs="EPSG:4326"):
    """Loads coordinates from shapefile, calculates midpoints, returns mapping."""
    print(f"\n--- Loading Link Coordinates from Shapefile: {shapefile_path} ---")
    try:
        gdf = gpd.read_file(shapefile_path); gdf_link_id_type = object
        link_id_col_shp = next((col for col in gdf.columns if col.lower() == 'link_id'), 'LINK_ID')
        if link_id_col_shp != 'LINK_ID': gdf.rename(columns={link_id_col_shp: 'LINK_ID'}, inplace=True)
        if 'LINK_ID' not in gdf.columns: raise ValueError("Shapefile needs 'link_id' or 'LINK_ID'.")
        try: gdf['LINK_ID'] = gdf['LINK_ID'].astype(int); gdf_link_id_type = np.int64
        except: gdf['LINK_ID'] = gdf['LINK_ID'].astype(str); gdf_link_id_type = object
        if gdf.crs != target_crs: gdf = gdf.to_crs(target_crs)
        print("Calculating midpoints...")
        coords = gdf['geometry'].apply(get_midpoint); gdf[['Latitude', 'Longitude']] = pd.DataFrame(coords.tolist(), index=gdf.index)
        rows_before = len(gdf); gdf.dropna(subset=['Latitude', 'Longitude'], inplace=True); rows_after = len(gdf)
        if rows_before > rows_after: print(f"Dropped {rows_before - rows_after} rows missing midpoints.")
        if gdf.empty: print("Error: No valid coords."); return None, None
        link_coords_map = gdf[['LINK_ID', 'Latitude', 'Longitude']].drop_duplicates(subset=['LINK_ID']).reset_index(drop=True)
        print(f"Coordinates loaded for {len(link_coords_map)} unique LINK_IDs (Type: {gdf_link_id_type}).")
        return link_coords_map, gdf_link_id_type
    except Exception as e: print(f"Error loading coordinates: {e}"); return None, None

def select_best_candidate_in_cluster(group, centroid):
    """Selects the best candidate within a cluster group."""
    if group.empty: return None
    max_point = group['point'].max()
    top_points = group[group['point'] == max_point]
    if len(top_points) == 1: return top_points.iloc[0]
    else:
        top_points_coords = top_points[['Latitude', 'Longitude']].apply(pd.to_numeric, errors='coerce').dropna()
        if top_points_coords.empty: return top_points.iloc[0]
        distances = cdist(top_points_coords.values, [centroid])
        closest_original_index = top_points_coords.index[np.argmin(distances)]
        return top_points.loc[closest_original_index]


# --- 3. Data Processing Pipeline ---
dataframes_initial = {}
dataframes_sigungu_filtered = {}
final_individual_dfs = {}
merged_df = None
final_candidates_df = None

# --- Stage 1: Load and Process Initial Files ---
print("--- Stage 1: Loading and Initial Processing ---")
try: od_df_raw=pd.read_csv(od_file_path, encoding='utf-8-sig'); OD=od_df_raw[['LINK_ID', 'Freq']].rename(columns={'Freq': 'count'}); dataframes_initial['OD']=OD; print(f"Processed {od_file_path}")
except Exception as e: print(f"Error {od_file_path}: {e}")
try: rest_area_df_raw=pd.read_csv(rest_area_file_path, encoding='utf-8-sig'); rest_area_df_raw.rename(columns={'link_id':'LINK_ID'}, inplace=True, errors='ignore'); rest_area=rest_area_df_raw[['LINK_ID', 'count']]; dataframes_initial['rest_area']=rest_area; print(f"Processed {rest_area_file_path}")
except Exception as e: print(f"Error {rest_area_file_path}: {e}")
try: traffic_df_raw=pd.read_csv(traffic_file_path, encoding='utf-8-sig'); traffic=traffic_df_raw[['LINK_ID', 'Volume']].rename(columns={'Volume': 'count'}); dataframes_initial['traffic']=traffic; print(f"Processed {traffic_file_path}")
except Exception as e: print(f"Error {traffic_file_path}: {e}")
try: infra_df_raw=pd.read_excel(infra_file_path); infra_mapping={'물류창고': 1, '물류기지': 2, '공영차고지': 3, '물류단지': 4, '화물차휴게소': 5}; infra_selected=infra_df_raw[['LINK_ID', '종류', 'size']].copy(); infra_selected['infra']=infra_selected['종류'].map(infra_mapping); infra=infra_selected[['LINK_ID', 'infra', 'size']].copy(); infra.dropna(subset=['infra'], inplace=True); infra['infra']=infra['infra'].astype(int); dataframes_initial['infra']=infra; print(f"Processed {infra_file_path}")
except ImportError: print(f"Error {infra_file_path}: Missing 'openpyxl'. Install. Skipping.")
except Exception as e: print(f"Error {infra_file_path}: {e}")
try: interval_df_raw=pd.read_csv(interval_file_path, encoding='utf-8-sig'); interval=interval_df_raw[['LINK_ID']]; dataframes_initial['interval']=interval; print(f"Processed {interval_file_path}")
except Exception as e: print(f"Error {interval_file_path}: {e}")
print(f"\nInitial DFs available: {list(dataframes_initial.keys())}")


# --- Stage 1b: Load Coordinates ---
link_coords_map, gdf_link_id_type = load_link_coordinates(SHAPEFILE_PATH)
if link_coords_map is not None and gdf_link_id_type is not None:
     print(f"\nEnsuring initial DataFrame LINK_IDs match coordinate map type ({gdf_link_id_type})...")
     for name, df in dataframes_initial.items():
          if df is not None and 'LINK_ID' in df.columns and df['LINK_ID'].dtype != gdf_link_id_type:
               try: df['LINK_ID'] = df['LINK_ID'].astype(gdf_link_id_type)
               except Exception as e: print(f"  Warn: Type conversion failed for {name}: {e}")


# --- Stage 2 & 3: Sigungu Filtering ---
print("\n--- Stage 2 & 3: Sigungu Filtering ---")
link_to_sigungu_map = None; can_proceed_with_filter = False
# **** MODIFICATION: Updated excluded Sigungu IDs ****
excluded_sigungu_ids = [39010, 39020, 37430] # Exclude these three specific IDs
print(f"Sigungu IDs to exclude: {excluded_sigungu_ids}")
try:
    link_to_sigungu_df = pd.read_csv(link_to_emd_file, encoding='utf-8-sig')
    if 'k_link_id' not in link_to_sigungu_df.columns or 'sigungu_id' not in link_to_sigungu_df.columns: raise KeyError("Missing map cols.")
    link_to_sigungu_map = link_to_sigungu_df[['k_link_id', 'sigungu_id']].drop_duplicates(subset=['k_link_id']).reset_index(drop=True)
    sigungu_id_type = link_to_sigungu_map['sigungu_id'].dtype; print(f"Map 'sigungu_id' type: {sigungu_id_type}")
    try: excluded_sigungu_ids = [np.dtype(sigungu_id_type).type(x) for x in excluded_sigungu_ids]; print(f"Converted exclude list type: {excluded_sigungu_ids}")
    except Exception as e: print(f"Warn: Could not convert excluded_sigungu_ids type: {e}")
    if gdf_link_id_type and link_to_sigungu_map['k_link_id'].dtype != gdf_link_id_type:
         try: link_to_sigungu_map['k_link_id'] = link_to_sigungu_map['k_link_id'].astype(gdf_link_id_type); print("Map k_link_id type matched.")
         except Exception as e: print(f"Warn: k_link_id conversion failed: {e}")
    can_proceed_with_filter = True; print("Sigungu map loaded.")
except Exception as e: print(f"Error loading Sigungu map: {e}")

if can_proceed_with_filter and link_to_sigungu_map is not None:
    print("Executing Sigungu filtering...")
    for name, df in dataframes_initial.items():
        if df is None or df.empty or 'LINK_ID' not in df.columns: dataframes_sigungu_filtered[name] = df; continue
        print(f"Filtering {name}...")
        try:
             if df['LINK_ID'].dtype != link_to_sigungu_map['k_link_id'].dtype:
                 try: df['LINK_ID'] = df['LINK_ID'].astype(link_to_sigungu_map['k_link_id'].dtype)
                 except Exception as e: print(f" Warn: Cannot match LINK_ID types for {name}: {e}"); dataframes_sigungu_filtered[name] = df; continue
             df_merged = pd.merge(df, link_to_sigungu_map, left_on='LINK_ID', right_on='k_link_id', how='left')
             if 'sigungu_id' in df_merged.columns:
                 df_merged['sigungu_id_numeric'] = pd.to_numeric(df_merged['sigungu_id'], errors='coerce')
                 rows_to_keep = ~df_merged['sigungu_id_numeric'].isin(excluded_sigungu_ids) | df_merged['sigungu_id_numeric'].isna()
                 df_filtered = df_merged[rows_to_keep].copy()
                 df_final = df_filtered.drop(columns=['sigungu_id', 'k_link_id', 'sigungu_id_numeric'], errors='ignore')
                 removed_count = len(df) - len(df_final);
                 if removed_count > 0 : print(f"  {name}: {removed_count} rows removed.")
             else: df_final = df.copy()
             dataframes_sigungu_filtered[name] = df_final
        except Exception as e: print(f"Error filtering {name}: {e}"); dataframes_sigungu_filtered[name] = df
else: print("Skipping Sigungu filtering."); dataframes_sigungu_filtered = dataframes_initial


# --- Stage 4: Final Filtering, Scoring, and Adjustments ---
print("\n--- Stage 4: Applying Final Filtering, Scoring, and Adjustments ---")
print(f"Base candidate_num: {candidate_num}, TARGET_COUNT_FILTER: {TARGET_COUNT_FILTER}")
for name, df in dataframes_sigungu_filtered.items():
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: print(f"\nSkipping {name}."); final_individual_dfs[name] = None; continue
    print(f"\nProcessing final filter/score/adjust for: {name}")
    current_df = df.copy(); final_df = None
    try:
        # --- OD / Traffic ---
        if name in ['OD', 'traffic']:
            if 'count' not in current_df.columns or 'LINK_ID' not in current_df.columns: raise ValueError("Missing cols")
            current_df_sorted = current_df.sort_values(by='count', ascending=False); current_df_top = current_df_sorted.head(TARGET_COUNT_FILTER).copy()
            if not current_df_top.empty:
                q1, q2, q3 = current_df_top['count'].quantile([0.25, 0.5, 0.75]); bins = [-np.inf, q1, q2, q3, np.inf]; labels = [1, 2, 3, 4]
                current_df_top['point'] = pd.cut(current_df_top['count'], bins=bins, labels=labels, include_lowest=True, right=True); current_df_top['point'].fillna(1, inplace=True); current_df_top['point'] = current_df_top['point'].astype(int)
                temp_df = current_df_top.drop(columns=['count']); final_df = temp_df.rename(columns={'point': 'count'})
            else: final_df = pd.DataFrame(columns=['LINK_ID', 'count'])
        # --- INFRA ---
        elif name == 'infra':
            if 'LINK_ID' not in current_df.columns: raise ValueError("Missing LINK_ID")
            print(f"  DEBUG (infra): Input shape = {current_df.shape}")
            unique_link_ids = current_df['LINK_ID'].nunique(); print(f"  DEBUG (infra): Found {unique_link_ids} unique LINK_IDs.")
            if unique_link_ids < TARGET_COUNT_FILTER:
                print(f"  Unique LINK_IDs ({unique_link_ids}) < Target ({TARGET_COUNT_FILTER}). Selecting all unique LINK_IDs.")
                final_df = pd.DataFrame({'LINK_ID': current_df['LINK_ID'].unique()}); final_df['count'] = 1
            else:
                print(f"  Unique LINK_IDs ({unique_link_ids}) >= Target ({TARGET_COUNT_FILTER}). Applying priority/aggregation logic.")
                if 'infra' not in current_df.columns or 'size' not in current_df.columns: raise ValueError("Missing infra/size")
                infra_priority = current_df[current_df['infra'] != 1].copy(); infra_secondary = current_df[current_df['infra'] == 1].copy()
                infra_priority_unique = infra_priority.drop_duplicates(subset=['LINK_ID'], keep='first'); num_priority_unique = len(infra_priority_unique)
                print(f"  DEBUG (infra): Unique priority count = {num_priority_unique}")
                priority_link_ids = set(infra_priority_unique['LINK_ID'])
                infra_secondary_filtered = infra_secondary[~infra_secondary['LINK_ID'].isin(priority_link_ids)].copy()
                print(f"  DEBUG (infra): Secondary count after filtering overlaps = {len(infra_secondary_filtered)}")
                if not infra_secondary_filtered.empty:
                    infra_secondary_filtered['size'] = pd.to_numeric(infra_secondary_filtered['size'], errors='coerce'); infra_secondary_filtered.dropna(subset=['size'], inplace=True)
                    warehouse_aggregated_size = infra_secondary_filtered.groupby('LINK_ID', as_index=False)['size'].sum()
                    print(f"  DEBUG (infra): Aggregated unique warehouse count = {len(warehouse_aggregated_size)}")
                else: warehouse_aggregated_size = pd.DataFrame(columns=['LINK_ID', 'size'])
                if num_priority_unique >= TARGET_COUNT_FILTER:
                    print(f"  Selecting top {TARGET_COUNT_FILTER} from unique priority group.")
                    selected_priority_links_df = infra_priority_unique.head(TARGET_COUNT_FILTER)[['LINK_ID']]
                    final_df = selected_priority_links_df.copy(); final_df['count'] = 1
                else:
                    num_needed = TARGET_COUNT_FILTER - num_priority_unique; print(f"  Need {num_needed} more from aggregated warehouses.")
                    warehouse_aggregated_size_sorted = warehouse_aggregated_size.sort_values(by='size', ascending=False)
                    selected_warehouse_links_df = warehouse_aggregated_size_sorted.head(num_needed)[['LINK_ID']]
                    print(f"  Selected top {len(selected_warehouse_links_df)} aggregated warehouse LINK_IDs.")
                    final_links_df = pd.concat([infra_priority_unique[['LINK_ID']], selected_warehouse_links_df], ignore_index=True)
                    final_links_df = final_links_df.drop_duplicates(subset=['LINK_ID']).head(TARGET_COUNT_FILTER) # Safety check
                    final_df = final_links_df.copy(); final_df['count'] = 1
            if 'final_df' not in locals() or final_df is None: final_df = pd.DataFrame(columns=['LINK_ID', 'count'])
        # --- Interval ---
        elif name == 'interval':
            if 'LINK_ID' not in current_df.columns: raise ValueError("Missing LINK_ID")
            current_df['count'] = 1; final_df = current_df[['LINK_ID', 'count']].copy(); final_df.drop_duplicates(subset=['LINK_ID'], keep='first', inplace=True)
        # --- Rest Area ---
        elif name == 'rest_area':
            if 'LINK_ID' not in current_df.columns or 'count' not in current_df.columns: raise ValueError("Missing cols")
            final_df = current_df[['LINK_ID', 'count']].copy(); final_df.drop_duplicates(subset=['LINK_ID'], keep='first', inplace=True)
        else: final_df = None
        final_individual_dfs[name] = final_df
        if final_df is not None: print(f"  Finished final processing for {name}. Final Shape: {final_df.shape}")
    except Exception as e: print(f"  Error final processing {name}: {e}"); final_individual_dfs[name] = None


# --- Stage 5: Save Individual Final DataFrames ---
print(f"\n--- Stage 5: Saving Individual DataFrames to {preliminary_output_dir} ---")
# (Saving logic unchanged)
try:
    os.makedirs(preliminary_output_dir, exist_ok=True); print(f"Directory '{preliminary_output_dir}' checked.")
    for name, df in final_individual_dfs.items():
        if df is not None and not df.empty:
            output_file = os.path.join(preliminary_output_dir, f"{name}.csv")
            try: df.to_csv(output_file, index=False, encoding='utf-8-sig'); print(f"Saved {output_file}")
            except Exception as e: print(f"Error saving {name}: {e}")
        else: print(f"Skipping save for {name}.")
except Exception as e: print(f"Error in Stage 5: {e}")


# --- Stage 5b: Visualize Individual DataFrames ---
print(f"\n--- Stage 5b: Visualizing Individual DataFrames ---")
# (Visualization logic unchanged)
if not folium_present or link_coords_map is None: print("Skipping individual viz.")
else:
    individual_vis_dir = os.path.join(preliminary_output_dir, "Individual_Visualizations"); os.makedirs(individual_vis_dir, exist_ok=True)
    print(f"Saving individual viz to: {individual_vis_dir}")
    for name, df in final_individual_dfs.items():
        if df is None or df.empty: continue; print(f"Visualizing {name}...")
        try:
            df_with_coords = pd.merge(df, link_coords_map, on='LINK_ID', how='inner')
            if df_with_coords.empty: print(f"  No coords for {name}."); continue
            map_center = [df_with_coords['Latitude'].mean(), df_with_coords['Longitude'].mean()]; m = folium.Map(location=map_center, zoom_start=10, control_scale=True)
            color_map = {}; legend_title = f"{name}"; use_count_for_color = False
            if name in ['OD', 'traffic']: color_map = VIS_COLORS_OD_TRAFFIC; legend_title += " Score"; use_count_for_color = True
            elif name in ['infra', 'interval']: color_map = {1: VIS_COLORS_OTHERS}; legend_title += " (Presence)" ; use_count_for_color = True
            else: color_map = {0: VIS_COLORS_OTHERS}; legend_title += " Link";
            points_added = 0
            for _, row in tqdm(df_with_coords.iterrows(), total=len(df_with_coords), desc=f'Viz {name}'):
                count_val = row.get('count', 0); color = 'grey'
                if use_count_for_color: color = color_map.get(int(count_val), 'grey')
                elif name=='rest_area': color = color_map.get(0, 'grey')
                popup=f"LINK_ID: {row['LINK_ID']}<br>{name} Count: {count_val}"
                folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.7, popup=popup).add_to(m)
                points_added += 1
            if color_map: # Add simple legend
                legend_html = f'<div style="position:fixed; bottom:10px; right:10px; border:1px solid grey; z-index:9999; font-size:10px; background-color:white; padding: 3px;"><b>{legend_title}</b><br>'
                if name in ['OD', 'traffic']:
                    for score, clr in sorted(color_map.items()): legend_html += f'<i style="background:{clr};width:10px;height:10px;display:inline-block;"></i> {score}<br>'
                elif name in ['infra', 'interval'] and 1 in color_map : legend_html += f'<i style="background:{color_map[1]};width:10px;height:10px;display:inline-block;"></i> Present<br>'
                elif name == 'rest_area': legend_html += f'<i style="background:{color_map[0]};width:10px;height:10px;display:inline-block;"></i> Link<br>'
                legend_html += '</div>'; m.get_root().html.add_child(folium.Element(legend_html))
            map_filename = os.path.join(individual_vis_dir, f"{name}_visualization.html")
            m.save(map_filename); print(f"  Saved map for {name} ({points_added} pts)")
        except Exception as e: print(f"  Error visualizing {name}: {e}")


# --- Stage 6: Prepare for Merge & Merge DataFrames ---
print(f"\n--- Stage 6: Prepare for Merge & Merge DataFrames ---")
# (Merge logic unchanged)
dfs_to_merge = []; print("Preparing DFs for merge...")
for name, df in final_individual_dfs.items():
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        if 'count' in df.columns and 'LINK_ID' in df.columns:
            df_renamed = df.rename(columns={'count': name}); dfs_to_merge.append(df_renamed) # Rename count -> name
            print(f"  Prepared '{name}' (Shape: {df_renamed.shape})")
        else: print(f"  Skipping {name}: Missing cols.")
    else: print(f"  Skipping {name}: None or empty.")
merged_df = None
if len(dfs_to_merge) > 1:
    print(f"\nMerging {len(dfs_to_merge)} DFs..."); start_time = time.time()
    try:
        ref_dtype = dfs_to_merge[0]['LINK_ID'].dtype
        for i in range(1, len(dfs_to_merge)):
            if dfs_to_merge[i]['LINK_ID'].dtype != ref_dtype:
                 try: dfs_to_merge[i]['LINK_ID'] = dfs_to_merge[i]['LINK_ID'].astype(ref_dtype)
                 except Exception as e: print(f"Warn: LINK_ID type mismatch {i}: {e}")
        merge_func = lambda left, right: pd.merge(left, right, on='LINK_ID', how='outer'); merged_df = reduce(merge_func, dfs_to_merge)
        print(f"Merged. Shape: {merged_df.shape}. Adding points/coords...")
        value_cols = [col for col in merged_df.columns if col != 'LINK_ID']; merged_df[value_cols] = merged_df[value_cols].fillna(0)
        merged_df['point'] = merged_df[value_cols].apply(lambda row: (row != 0).sum(), axis=1)
        for col in value_cols: merged_df[col] = merged_df[col].astype(int)
        if link_coords_map is not None:
            if merged_df['LINK_ID'].dtype != link_coords_map['LINK_ID'].dtype:
                 try: merged_df['LINK_ID'] = merged_df['LINK_ID'].astype(link_coords_map['LINK_ID'].dtype)
                 except Exception as e: print(f"Warn: LINK_ID type mismatch merge coords: {e}")
            merged_df = pd.merge(merged_df, link_coords_map, on='LINK_ID', how='left')
            missing_coords = merged_df['Latitude'].isnull().sum();
            if missing_coords > 0: print(f"Warn: {missing_coords} rows miss coords post-merge.")
        print(f"Merge/Point/Coord complete ({time.time() - start_time:.2f}s). Final shape: {merged_df.shape}")
    except Exception as e: print(f"Error merge/point/coord: {e}"); merged_df = None
elif len(dfs_to_merge) == 1:
     print("Only 1 DF. Calc point/coord."); merged_df = dfs_to_merge[0].copy()
     orig_name = next((name for name, df_orig in final_individual_dfs.items() if df_orig is dfs_to_merge[0]), 'value') # Get original name
     value_cols = [orig_name] # The column is already named correctly
     merged_df[value_cols] = merged_df[value_cols].fillna(0); merged_df['point'] = merged_df[value_cols].apply(lambda row: (row != 0).sum(), axis=1)
     for col in value_cols: merged_df[col] = merged_df[col].astype(int)
     if link_coords_map is not None: merged_df = pd.merge(merged_df, link_coords_map, on='LINK_ID', how='left')
else: print("No DFs to merge.")


# --- Stage 7: K-Means Clustering ---
print(f"\n--- Stage 7: K-Means Clustering (k={candidate_num}) ---")
# (K-Means logic unchanged)
cluster_centroids = None
if not sklearn_present: print("Skipping K-Means.")
elif merged_df is None or 'Latitude' not in merged_df.columns or merged_df[['Latitude', 'Longitude']].isnull().any().any(): print("Skipping K-Means (Data/Coords missing).")
else:
    X_cluster_df = merged_df.dropna(subset=['Latitude', 'Longitude']); X_cluster = X_cluster_df[['Latitude', 'Longitude']].values
    if len(X_cluster) >= candidate_num:
        try:
            print(f"Running K-Means on {len(X_cluster)} points..."); start_time = time.time()
            kmeans = KMeans(n_clusters=candidate_num, random_state=KMEANS_RANDOM_STATE, n_init='auto')
            labels = kmeans.fit_predict(X_cluster); merged_df.loc[X_cluster_df.index, 'cluster'] = labels
            cluster_centroids = kmeans.cluster_centers_
            print(f"K-Means complete ({time.time() - start_time:.2f}s).")
        except Exception as e: print(f"Error K-Means: {e}")
    else: print(f"Skipping K-Means (Valid points {len(X_cluster_df)} < k {candidate_num}).")


# --- Stage 7b: Visualize Clusters ---
print(f"\n--- Stage 7b: Visualizing Clusters ---")
# (Cluster viz logic unchanged)
if not folium_present: print("Skipping cluster viz.")
elif merged_df is None or 'cluster' not in merged_df.columns or merged_df['cluster'].isnull().all(): print("Skipping cluster viz (No cluster data).")
elif 'Latitude' not in merged_df.columns or merged_df[['Latitude', 'Longitude']].isnull().any().any(): print("Skipping cluster viz (No coords).")
else:
    print("Creating cluster viz map...")
    df_viz_cluster = merged_df.dropna(subset=['Latitude', 'Longitude', 'cluster']).copy(); df_viz_cluster['cluster'] = df_viz_cluster['cluster'].astype(int)
    if not df_viz_cluster.empty:
        map_center = [df_viz_cluster['Latitude'].mean(), df_viz_cluster['Longitude'].mean()]
        m_clusters = folium.Map(location=map_center, zoom_start=10, control_scale=True)
        num_colors = len(CLUSTER_COLORS); points_added = 0
        for _, row in tqdm(df_viz_cluster.iterrows(), total=len(df_viz_cluster), desc='Viz Clusters'):
             cluster_id = int(row['cluster']); color = CLUSTER_COLORS[cluster_id % num_colors]
             popup=f"LINK_ID: {row['LINK_ID']}<br>Cluster: {cluster_id}<br>Point: {row['point']}"
             folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.6, popup=popup).add_to(m_clusters)
             points_added += 1
        cluster_map_filename = os.path.join(preliminary_output_dir, "Cluster_Visualization.html")
        try: m_clusters.save(cluster_map_filename); print(f"Saved cluster viz map ({points_added} pts)")
        except Exception as e: print(f"Error saving cluster map: {e}")
    else: print("No valid points for cluster viz.")


# --- Stage 8: Select Final Candidates ---
print(f"\n--- Stage 8: Selecting Final Candidates from Clusters ---")
# (Selection logic unchanged)
if cluster_centroids is None or 'cluster' not in merged_df.columns or merged_df['cluster'].isnull().all(): print("Skipping final selection.")
elif not sklearn_present: print("Skipping final selection (scipy missing).")
elif merged_df.dropna(subset=['Latitude', 'Longitude', 'point']).empty : print("Skipping final selection (No valid coords/points).")
else:
    print("Selecting best candidate per cluster..."); start_time = time.time()
    final_candidates_list = []
    selection_data = merged_df.dropna(subset=['cluster', 'Latitude', 'Longitude', 'point']).copy(); selection_data['cluster'] = selection_data['cluster'].astype(int)
    cluster_groups = selection_data.groupby('cluster')
    iterable_groups = tqdm(cluster_groups, total=len(cluster_groups), desc="Selecting Candidates") if tqdm_present else cluster_groups
    for cluster_id, group in iterable_groups:
        if 0 <= cluster_id < len(cluster_centroids):
             centroid = cluster_centroids[cluster_id]; best_candidate = select_best_candidate_in_cluster(group, centroid)
             if best_candidate is not None: final_candidates_list.append(best_candidate)
        else: print(f"Warn: Centroid missing for cluster {cluster_id}.")
    if final_candidates_list:
        final_candidates_df = pd.DataFrame(final_candidates_list).reset_index(drop=True); print(f"Selected {len(final_candidates_df)} candidates ({time.time() - start_time:.2f}s).")
        if len(final_candidates_df) != candidate_num and len(cluster_groups)>0 : print(f"Warn: Expected {candidate_num}, got {len(final_candidates_df)}.")
    else: print("No final candidates selected."); final_candidates_df = None


# --- Stage 9: Save Merged and Final Candidate DataFrames ---
print(f"\n--- Stage 9: Saving Merged and Final Candidate DataFrames ---")
# (Saving logic unchanged)
try: os.makedirs(final_output_dir, exist_ok=True); print(f"Directory '{final_output_dir}' checked.")
except Exception as e: print(f"Error creating final dir: {e}")
if merged_df is not None and not merged_df.empty:
    merged_output_file = os.path.join(preliminary_output_dir, "Merged_Preliminary_Candidates_With_Points.csv")
    try: merged_df.to_csv(merged_output_file, index=False, encoding='utf-8-sig'); print(f"Saved merged DF to {merged_output_file}")
    except Exception as e: print(f"Error saving merged DF: {e}")
else: print("Merged DF not saved.")
if final_candidates_df is not None and not final_candidates_df.empty:
    final_cand_output_file = os.path.join(final_output_dir, "Final_Candidates_Selected.csv")
    try: final_candidates_df.to_csv(final_cand_output_file, index=False, encoding='utf-8-sig'); print(f"Saved final candidates DF to {final_cand_output_file}")
    except Exception as e: print(f"Error saving final candidates DF: {e}")
else: print("Final candidates DF not saved.")


# --- Stage 10: Visualize Final Candidates ---
print(f"\n--- Stage 10: Visualizing Final Selected Candidates ---")
# (Visualization logic unchanged)
if not folium_present: print("Skipping final viz.")
elif final_candidates_df is None or final_candidates_df.empty: print("Skipping final viz (No data).")
elif 'Latitude' not in final_candidates_df.columns: print("Skipping final viz (No coords).")
else:
    print(f"Creating map for {len(final_candidates_df)} final candidates..."); map_center = [final_candidates_df['Latitude'].mean(), final_candidates_df['Longitude'].mean()]
    m_final = folium.Map(location=map_center, zoom_start=10, control_scale=True)
    fcg = folium.FeatureGroup(name=f'Final {candidate_num} Candidates')
    points_added = 0
    for _, row in tqdm(final_candidates_df.iterrows(), total=len(final_candidates_df), desc='Viz Final Candidates'):
        point_val = row.get('point', 0); color = VIS_COLORS_FINAL.get(int(point_val), 'gray')
        popup=f"LINK_ID: {row['LINK_ID']}<br>Point: {point_val}<br>Cluster: {row.get('cluster', 'N/A')}"
        folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.8, popup=popup).add_to(fcg)
        points_added +=1
    fcg.add_to(m_final)
    legend_html_final = f'<div style="position: fixed; bottom: 10px; right: 10px; border:1px solid grey; z-index:9999; font-size:10px; background-color: white; padding: 3px;"><b>Final Points</b><br>'
    present_points = sorted([p for p in final_candidates_df['point'].unique() if p in VIS_COLORS_FINAL])
    for score in present_points: legend_html_final += f'<i style="background:{VIS_COLORS_FINAL[score]};width:10px;height:10px;display:inline-block;"></i> {score}<br>'
    legend_html_final += '</div>'; m_final.get_root().html.add_child(folium.Element(legend_html_final))
    final_map_filename = os.path.join(final_output_dir, "Final_Candidates_Point_Map.html")
    try: m_final.save(final_map_filename); print(f"Saved final candidates map ({points_added} pts)")
    except Exception as e: print(f"Error saving final candidates map: {e}")


# --- Stage 11: Final Summary ---
print("\n--- Stage 11: All Processing Complete ---")
# (Summary - unchanged)
print("\nSummary of Outputs:")
print(f"* Preliminary Candidates Directory: {preliminary_output_dir}")
print(f"  - Individual Processed CSVs (OD.csv, etc.)")
print(f"  - Individual Visualizations (in Individual_Visualizations subfolder)")
print(f"  - Merged Data CSV (Merged_Preliminary_Candidates_With_Points.csv)")
print(f"  - Cluster Visualization Map (Cluster_Visualization.html)")
print(f"\n* Final Candidates Directory: {final_output_dir}")
print(f"  - Final Selected Candidates CSV (Final_Candidates_Selected.csv)")
print(f"  - Final Candidates Visualization Map (Final_Candidates_Point_Map.html)")
print("\n(Note: Files are created if steps succeed and data is available.)")