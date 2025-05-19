# Required libraries
import geopandas as gpd
import pandas as pd
import folium
import os
import time
from tqdm import tqdm # For progress bar
import numpy as np # For quantile calculation if needed

# --- Configuration ---
SHAPEFILE_PATH = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\main_road_network_level_5.5\level5_5_link_probe_32_2020.shp"
# *** UPDATED: CSV file path changed to link_yearly_volume ***
CSV_FILE_PATH = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\Data_Analysis\Traffic_OD\Link_Volume_2020\link_yearly_volume_2020.csv"
OUTPUT_DIR = r"D:\project\HDT_EVCS_Opt\Analysis\Candidate" # Directory to save maps
# *** UPDATED: New base name for output files reflecting the data source ***
OUTPUT_BASE_NAME = 'Link_Yearly_Volume_Visualize' # Base name for output files
TARGET_CRS = "EPSG:4326" # WGS 84 (Latitude/Longitude)

# --- Helper Function ---
def get_midpoint(geom):
    """
    Calculates the midpoint (latitude, longitude) of a LineString geometry.

    Args:
        geom (shapely.geometry.LineString): The geometry object.

    Returns:
        tuple: A tuple containing (latitude, longitude) or (None, None) if input is invalid.
    """
    if geom and geom.geom_type == 'LineString':
        mid_point = geom.interpolate(0.5, normalized=True)
        return mid_point.y, mid_point.x
    return None, None

# --- Data Loading and Preparation Function ---
def load_and_prepare_data(shapefile_path, csv_file_path):
    """
    Loads shapefile and CSV, processes columns (LINK_ID, Yearly_Volume, Year),
    matches links, calculates coordinates, and merges data.

    Args:
        shapefile_path (str): Path to the input shapefile.
        csv_file_path (str): Path to the input CSV file.

    Returns:
        geopandas.GeoDataFrame or None: Merged GeoDataFrame with coordinates and counts,
                                        or None if an error occurs.
    """
    print("--- Starting Data Loading and Preparation ---")

    # Step 1: Load Shapefile (Error handling included)
    print(f"Step 1: Loading shapefile from {shapefile_path}...")
    start_time = time.time()
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"Step 1 Complete: Shapefile loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None
    if 'link_id' not in gdf.columns:
        # Attempt to find case-insensitive match if 'link_id' is missing
        link_id_col = next((col for col in gdf.columns if col.lower() == 'link_id'), None)
        if link_id_col:
            print(f" - Found link identifier column as '{link_id_col}'. Renaming to 'link_id'.")
            gdf.rename(columns={link_id_col: 'link_id'}, inplace=True)
        else:
            print("Error: The column 'link_id' (or similar) does not exist in the shapefile.")
            return None
    print("Step 1 Validation: 'link_id' column found in shapefile.")


    # Step 2: Load CSV File and Process Columns
    print(f"Step 2: Loading CSV file from {csv_file_path}...")
    start_time = time.time()
    try:
        df_csv = pd.read_csv(csv_file_path)
        print(f"Step 2 Complete: CSV file loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # *** MODIFIED: Process LINK_ID, Yearly_Volume, Year columns ***
    print("Step 2b: Processing CSV columns (LINK_ID, Yearly_Volume, Year)...")

    # Check for required input columns
    # *** UPDATED: Changed required columns ***
    required_input_cols = ['LINK_ID', 'Yearly_Volume', 'Year']
    if not all(col in df_csv.columns for col in required_input_cols):
        missing_cols = [col for col in required_input_cols if col not in df_csv.columns]
        print(f"Error: Required columns missing in CSV: {missing_cols}")
        return None

    # 1. Rename LINK_ID to link_id
    df_csv.rename(columns={'LINK_ID': 'link_id'}, inplace=True)
    print(" - Renamed 'LINK_ID' to 'link_id'.")

    # 2. Rename Yearly_Volume to count and ensure numeric
    # *** UPDATED: Renaming Yearly_Volume to count ***
    df_csv.rename(columns={'Yearly_Volume': 'count'}, inplace=True)
    print(" - Renamed 'Yearly_Volume' to 'count'.")
    try:
        df_csv['count'] = pd.to_numeric(df_csv['count'], errors='coerce')
        rows_before_drop = len(df_csv)
        df_csv.dropna(subset=['count'], inplace=True) # Remove rows where count is not numeric
        rows_after_drop = len(df_csv)
        if rows_before_drop > rows_after_drop:
            # *** UPDATED: Changed warning message column name ***
            print(f" - Warning: Dropped {rows_before_drop - rows_after_drop} rows due to non-numeric values in 'Yearly_Volume' column.")
        print(f" - Ensured 'count' column is numeric.")
    except Exception as e:
        # *** UPDATED: Changed error message column name ***
        print(f"Error converting 'Yearly_Volume' column to numeric 'count': {e}")
        return None

    # 3. Drop the 'Year' column
    # *** UPDATED: Dropping 'Year' instead of 'year' ***
    try:
        df_csv.drop(columns=['Year'], inplace=True)
        print(" - Dropped 'Year' column.")
    except KeyError:
        print(" - 'Year' column not found, skipping drop.")
    except Exception as e:
        print(f"Error dropping 'Year' column: {e}")
        return None

    # Validate required final columns
    required_final_cols = ['link_id', 'count']
    if not all(col in df_csv.columns for col in required_final_cols):
        print(f"Error: Columns {required_final_cols} not available after processing CSV.")
        return None
    print("Step 2 Validation: 'link_id' and 'count' columns processed successfully.")

    # Step 3: Prepare and Match link_id (Error handling included)
    print("Step 3: Preparing and matching 'link_id' values...")
    start_time = time.time()
    try:
        # Ensure link_id columns are string type before cleaning
        gdf['link_id'] = gdf['link_id'].astype(str)
        df_csv['link_id'] = df_csv['link_id'].astype(str)

        gdf['link_id_clean'] = gdf['link_id'].str.strip().str.lower()
        df_csv['link_id_clean'] = df_csv['link_id'].str.strip().str.lower()
    except Exception as e:
        print(f"Error standardizing link_id columns: {e}")
        return None
    csv_link_ids = df_csv['link_id_clean'].unique()
    print(f" - Found {len(csv_link_ids)} unique link IDs in CSV file.")
    filtered_gdf = gdf[gdf['link_id_clean'].isin(csv_link_ids)].copy()
    print(f" - Filtered shapefile contains {len(filtered_gdf)} matching links.")
    if filtered_gdf.empty:
        print("Warning: No matching 'link_id' values found. Cannot proceed.")
        return None
    print(f"Step 3 Complete: Matching done in {time.time() - start_time:.2f} seconds.")

    # Step 4: Coordinate Calculation and CRS Conversion (Error handling included)
    print(f"Step 4: Converting CRS to {TARGET_CRS} and extracting midpoint coordinates...")
    start_time = time.time()
    if filtered_gdf.crs != TARGET_CRS:
        print(f" - Original CRS is {filtered_gdf.crs}. Converting to {TARGET_CRS}...")
        try:
            filtered_gdf = filtered_gdf.to_crs(TARGET_CRS)
            print(f" - CRS conversion successful.")
        except Exception as e:
            print(f"Error during CRS conversion: {e}")
            return None
    else:
        print(f" - GeoDataFrame is already in {TARGET_CRS}.")
    try:
        filtered_gdf[['Latitude', 'Longitude']] = pd.DataFrame(
            filtered_gdf['geometry'].apply(get_midpoint).tolist(),
            index=filtered_gdf.index
        )
    except Exception as e:
        print(f"Error calculating midpoints: {e}")
        return None
    initial_rows = len(filtered_gdf)
    filtered_gdf.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    if len(filtered_gdf) < initial_rows:
        print(f" - Warning: Dropped {initial_rows - len(filtered_gdf)} rows due to invalid geometry or calculation errors.")
    print(f"Step 4 Complete: Coordinates extracted in {time.time() - start_time:.2f} seconds.")

    # Step 5: Merge Data (Error handling included)
    print("Step 5: Merging coordinate data with count data...")
    start_time = time.time()
    coords_df = filtered_gdf[['link_id_clean', 'Latitude', 'Longitude']]
    # Select relevant columns from processed CSV data
    csv_data_df = df_csv[['link_id_clean', 'link_id', 'count']] # Use the processed columns
    merged_gdf = pd.merge(coords_df, csv_data_df, on='link_id_clean', how='inner')
    print(f"Step 5 Complete: Merged data contains {len(merged_gdf)} rows. Time: {time.time() - start_time:.2f} seconds.")
    print(f" - Sample merged data:\n{merged_gdf.head()}")
    print("--- Data Loading and Preparation Finished ---")
    return merged_gdf

# --- Map Creation Function ---
def create_map_for_top_n(merged_data, top_n, output_dir, output_base_name):
    """
    Creates and saves a Folium map visualizing the top N links based on count,
    using quantile classification for styling and adding a legend and scale bar.

    Args:
        merged_data (geopandas.GeoDataFrame): DataFrame with coordinates and counts.
        top_n (int): Number of top links to visualize.
        output_dir (str): Directory to save the HTML map.
        output_base_name (str): Base name for the output HTML file.
    """
    print(f"\n--- Creating Map for Top {top_n} Links ({output_base_name}) ---") # Added base name to print

    # Step 6a: Filter Top N
    print(f"Step 6a: Filtering top {top_n} links by 'count'...")
    start_time = time.time()
    top_n_data = merged_data.sort_values(by='count', ascending=False).head(top_n).copy()
    print(f" - Filtered data contains {len(top_n_data)} rows.")
    if top_n_data.empty:
        print(f"Warning: No data available after filtering for top {top_n}. Skipping map creation.")
        return
    print(f"Step 6a Complete: Filtering done in {time.time() - start_time:.2f} seconds.")

    # Step 6b: Quantile Classification
    print("Step 6b: Calculating quantiles and assigning bins...")
    start_time = time.time()
    bin_summary_data = {} # To store data for the legend
    try:
        bins, bin_edges = pd.qcut(top_n_data['count'], q=5, labels=False, retbins=True, duplicates='drop')
        top_n_data['quantile_bin'] = bins # Internal column name remains quantile_bin
        print(f" - Quantile bin edges: {bin_edges}")
        quantile_colors = ['blue', 'green', 'yellow', 'orange', 'red'] # 0(low) to 4(high)

        print(" - Count range per quantile rank:") # Changed print statement
        bin_summary = top_n_data.groupby('quantile_bin')['count'].agg(['min', 'max', 'count']).reset_index()
        for _, stats in bin_summary.iterrows():
            rank_label = int(stats['quantile_bin']) # Keep internal name, but use 'Rank' for display
            color = quantile_colors[rank_label]
            count_val = int(stats['count'])
            min_val = stats['min']
            max_val = stats['max']
            range_str = f"[{min_val:,.1f} - {max_val:,.1f}]"
            print(f"   Rank {rank_label} ({color}): Count ~{count_val:,}, Range {range_str}")
            # Store info for legend
            bin_summary_data[rank_label] = {'color': color, 'range': range_str, 'count': count_val}

    except ValueError as ve:
         # Handle cases where quantiles cannot be computed (e.g., too few unique values)
         print(f"Warning: Could not compute 5 quantiles due to data distribution ({ve}). Assigning all to rank 0.")
         top_n_data['quantile_bin'] = 0 # Assign all to the lowest rank
         quantile_colors = ['blue'] # Use only the first color
         # Update bin_summary_data for the single bin
         if not top_n_data.empty:
             min_val = top_n_data['count'].min()
             max_val = top_n_data['count'].max()
             count_val = len(top_n_data)
             range_str = f"[{min_val:,.1f} - {max_val:,.1f}]"
             bin_summary_data[0] = {'color': quantile_colors[0], 'range': range_str, 'count': count_val}
             print(f"   Rank 0 ({quantile_colors[0]}): Count ~{count_val:,}, Range {range_str}")
         else:
             bin_summary_data = {} # Clear legend data if no data

    except Exception as e:
        print(f"Error during quantile calculation: {e}. Proceeding without quantile styling.")
        top_n_data['quantile_bin'] = -1
        quantile_colors = ['gray']
        bin_summary_data = {} # Clear legend data on error

    print(f"Step 6b Complete: Quantile calculation done in {time.time() - start_time:.2f} seconds.")

    # Step 6c: Create Folium Map
    print("Step 6c: Creating Folium map and adding markers...")
    start_time = time.time()
    map_center = [36.5, 127.8]
    m = folium.Map(
        location=map_center,
        zoom_start=7,
        tiles='OpenStreetMap',
        control_scale=True # Add scale bar using parameter
    )
    print(" - Scale control added via map parameter.")

    print(f" - Adding markers for top {top_n} links...")
    fixed_radius = 4
    fixed_opacity = 1.0

    for _, row in tqdm(top_n_data.iterrows(), total=top_n_data.shape[0], desc=f'Adding Top {top_n} Markers'):
        lat = row.get('Latitude')
        lon = row.get('Longitude')
        count = row.get('count')
        original_link_id = row.get('link_id')
        quantile_bin = row.get('quantile_bin', -1) # Keep internal name

        if pd.isna(lat) or pd.isna(lon) or pd.isna(count):
            continue

        # Determine color based on quantile bin
        if quantile_bin != -1 and 0 <= quantile_bin < len(quantile_colors):
            color = quantile_colors[quantile_bin]
        else:
            # Fallback if quantile calculation failed or if only one bin was possible
            color = quantile_colors[0] if quantile_colors else 'gray'


        count_val = float(count)
        popup_html = f"Link ID: {original_link_id}<br>Count: {count_val:.0f}<br>Rank: {quantile_bin if quantile_bin != -1 else 'N/A'}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=fixed_radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fixed_opacity,
            popup=folium.Popup(popup_html, parse_html=True)
        ).add_to(m)
    print(f"Step 6c Complete: Map created and markers added in {time.time() - start_time:.2f} seconds.")

    # Step 6d: Add HTML Legend to Map
    print("Step 6d: Adding legend to the map...")
    if bin_summary_data: # Only add legend if quantile calculation was successful and meaningful
        legend_html = '''
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            width: 190px;
            height: auto;
            border:2px solid grey;
            z-index:9999;
            font-size:14px;
            background-color: white;
            padding: 10px;
            opacity: 0.9;
            ">
            <b>Legend (Count Quantiles)</b><br>
            <hr style="margin-top: 5px; margin-bottom: 5px;">
        '''
        # Add legend items dynamically, sorted by bin number (rank)
        for rank_label in sorted(bin_summary_data.keys()):
            item = bin_summary_data[rank_label]
            legend_html += f'&nbsp; <i style="background:{item["color"]}; width:18px; height:18px; display:inline-block; vertical-align:middle;"></i>&nbsp; Rank {rank_label} ({item["count"]:,} links)<br>'
            legend_html += f'&nbsp; <span style="font-size:11px; padding-left: 22px;">Range: {item["range"]}</span><br>'

        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        print(" - Legend added successfully.")
    else:
        print(" - Skipping legend addition (quantile calculation error or insufficient data).")
    print("Step 6d Complete.")


    # Step 7: Save Map
    # Use the updated OUTPUT_BASE_NAME for the filename
    output_filename = f"{output_base_name}_Top{top_n}_Quantile_Styled.html"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Step 7: Saving map to {output_path}...")
    start_time = time.time()
    try:
        os.makedirs(output_dir, exist_ok=True)
        m.save(output_path)
        print(f"Step 7 Complete: Map successfully saved in {time.time() - start_time:.2f} seconds.")
        print(f"--- Map Creation for Top {top_n} Finished ---")
    except Exception as e:
        print(f"Error saving map: {e}")

# --- Execute the Process ---
if __name__ == "__main__":
    # Pass the correct file paths from configuration
    prepared_data = load_and_prepare_data(SHAPEFILE_PATH, CSV_FILE_PATH)

    if prepared_data is not None and not prepared_data.empty:
        top_n_values = [2000, 1000]
        for n in top_n_values:
            # Pass the correct output base name from configuration
            create_map_for_top_n(
                merged_data=prepared_data,
                top_n=n,
                output_dir=OUTPUT_DIR,
                output_base_name=OUTPUT_BASE_NAME # Use the updated base name
            )
        print("\nAll requested maps have been generated.")
    else:
        print("\nData preparation failed or resulted in empty data. No maps were generated.")

