import geopandas as gpd
import pandas as pd
import os
import time

def convert_link_ids_to_coordinates(excel_file_path, output_csv_path):
    print("Step 1: Loading shapefile components...")
    time.sleep(1)  # 1초 대기
    shapefile_path = r"C:\Users\yemoy\OneDrive\바탕 화면\SEM_화물차충전소\level5_5_link_probe_32_2020.shp"
    gdf = gpd.read_file(shapefile_path)
    print("Step 1 Complete: Shapefile loaded.")
    
    if 'link_id' not in gdf.columns:
        raise KeyError("The column 'link_id' does not exist in the shapefile.")
    print("Step 1 Validation: link_id column found in shapefile.")
    
    print("Step 2: Loading Excel file...")
    time.sleep(1)  # 1초 대기
    df_excel = pd.read_excel(excel_file_path)
    print("Step 2 Complete: Excel file loaded.")
    
    if 'link_id' not in df_excel.columns or 'count' not in df_excel.columns:
        raise KeyError("The columns 'link_id' and 'count' must exist in the Excel file.")
    print("Step 2 Validation: link_id and count columns found in Excel file.")
    
    print("Step 3: Extracting LINK_IDs from Excel file...")
    time.sleep(1)  # 1초 대기
    link_ids = df_excel['link_id'].unique()
    print(f"Step 3 Complete: Extracted {len(link_ids)} LINK_IDs.")
    
    gdf['link_id'] = gdf['link_id'].astype(str).str.strip().str.lower()
    df_excel['link_id'] = df_excel['link_id'].astype(str).str.strip().str.lower()
    
    print("Shapefile link_id samples:")
    print(gdf['link_id'].head())
    print("Excel link_id samples:")
    print(df_excel['link_id'].head())
    
    print("Step 4: Filtering GeoDataFrame with LINK_IDs...")
    time.sleep(1)  # 1초 대기
    print("link_ids sample for debugging:")
    print(link_ids[:5])
    
    link_ids = [str(link_id).strip().lower() for link_id in link_ids]
    filtered_gdf = gdf[gdf['link_id'].isin(link_ids)]
    print(f"Step 4 Complete: Filtered GeoDataFrame contains {len(filtered_gdf)} rows.")
    
    if filtered_gdf.empty:
        print("Warning: No matching LINK_IDs found in the shapefile.")
    
    print("Step 5: Converting CRS and extracting coordinates...")
    time.sleep(1)  # 1초 대기
    
    if filtered_gdf.crs != "EPSG:4326":    # 좌표계 설정
        filtered_gdf = filtered_gdf.to_crs(epsg=4326)
    
    link_id_coords = filtered_gdf[['link_id', 'geometry']]
    
    def get_midpoint(geom):
        if geom and geom.geom_type == 'LineString':
            mid_point = geom.interpolate(0.5, normalized=True)
            return mid_point.y, mid_point.x
        return None, None
    
    link_id_coords['Latitude'], link_id_coords['Longitude'] = zip(*link_id_coords.geometry.apply(get_midpoint))
    
    link_id_coords = link_id_coords.drop(columns=['geometry'])
    print("Step 5 Complete: Coordinates extracted.")
    
    print(link_id_coords.head())
    
    print("Step 6: Merging with count data and saving results to CSV file...")
    time.sleep(1)  # 3초 대기
    # Merge with count data from Excel
    result_df = link_id_coords.merge(df_excel[['link_id', 'count']], on='link_id', how='left')
    
    output_csv_path = os.path.join(os.getcwd(), output_csv_path)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Step 6 Complete: Results saved to {output_csv_path}")
    
    return output_csv_path

# 예제 사용법
excel_file_path = r"C:\Users\yemoy\OneDrive\바탕 화면\SEM_화물차충전소\유전알고리즘1500iter.xlsx"
output_csv_path = 'ga_1500iter_5000truck.csv'
convert_link_ids_to_coordinates(excel_file_path, output_csv_path) 