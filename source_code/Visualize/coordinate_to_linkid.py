import pandas as pd
import geopandas as gpd
import time
import os
from shapely.geometry import Point
from tqdm import tqdm

def convert_coordinates_to_link_ids(excel_file_path, shapefile_path, output_csv_filename):
    # 현재 디렉토리에서 출력 파일 경로 설정
    output_csv_path = os.path.join(os.getcwd(), output_csv_filename)

    # Shapefile 데이터 읽기
    print("Step 1: Loading shapefile components...")
    time.sleep(1)
    global gdf
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=4326)
    print("Step 1 Complete: Shapefile loaded.")
    
    if 'link_id' not in gdf.columns:
        raise KeyError("The column 'link_id' does not exist in the shapefile.")
    print("Step 1 Validation: link_id column found in shapefile.")
    
    # Excel 파일에서 데이터 읽기
    print("Step 2: Loading Excel file...")
    time.sleep(1)
    df_excel = pd.read_excel(excel_file_path)
    print("Step 2 Complete: Excel file loaded.")
    
    if 'Latitude' not in df_excel.columns or 'Longitude' not in df_excel.columns:
        raise KeyError("The columns 'Latitude' and 'Longitude' must exist in the Excel file.")
    print("Step 2 Validation: Latitude and Longitude columns found in Excel file.")
    
    # 좌표를 GeoDataFrame으로 변환
    print("Step 3: Converting coordinates to GeoDataFrame...")
    df_excel['geometry'] = df_excel.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    points_gdf = gpd.GeoDataFrame(df_excel, geometry='geometry', crs="EPSG:4326")
    print("Step 3 Complete: GeoDataFrame created.")
    
    # 좌표를 링크 ID로 변환하는 함수
    def find_link_id(point):
        distances = gdf.geometry.distance(point)
        nearest_link = gdf.iloc[distances.idxmin()]
        return nearest_link['link_id']
    
    # 각 행에 대해 위도와 경도를 링크 ID로 변환
    print("Step 4: Matching coordinates to link IDs...")
    points_gdf['link_id'] = [find_link_id(point) for point in tqdm(points_gdf['geometry'], desc="Processing rows")]
    print("Step 4 Complete: Link IDs assigned.")
    
    # 변환된 데이터를 CSV 파일로 저장
    print("Step 5: Saving the results to CSV file...")
    try:
        points_gdf.to_csv(output_csv_path, index=False)
        print(f"Step 5 Complete: Results saved to {output_csv_path}")
    except Exception as e:
        print(f"파일을 저장하는 중 오류가 발생했습니다: {e}")
        exit()
    
    return output_csv_path

# 예제 사용법
excel_file_path = r"C:\Users\yemoy\OneDrive\바탕 화면\SEM_화물차충전소\변환해야하는휴게소.xlsx"
shapefile_path = r"C:\Users\yemoy\OneDrive\바탕 화면\SEM_화물차충전소\level5_5_link_probe_32_2020.shp"
output_csv_filename = "휴게소들들_좌표_링크id.csv"
convert_coordinates_to_link_ids(excel_file_path, shapefile_path, output_csv_filename)