import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
import folium
import zipfile
import requests
import io

# --- 1. 경로 및 설정 ---
# 입력 파일들이 있는 폴더 경로
input_dir = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Installed-infra"
# 시각화 결과물을 저장할 폴더 경로
output_dir = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Visualization\Infrastructure_Comparison"
# 격자 한 변의 크기 (미터 단위, 예: 25km)
cell_size_meters = 25000

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
except Exception as e:
    print(f"Error creating directory: {e}")
    output_dir = '.'

# --- 2. 대한민국 Shapefile 다운로드 및 로드 (안정적인 URL 사용) ---
shapefile_dir = os.path.join(output_dir, "shapefile")
shapefile_path = os.path.join(shapefile_dir, "ne_10m_admin_1_states_provinces.shp")

if not os.path.exists(shapefile_path):
    print("South Korea shapefile not found. Downloading...")
    try:
        os.makedirs(shapefile_dir, exist_ok=True)
        # Natural Earth에서 제공하는 더 안정적인 URL로 변경
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip"
        response = requests.get(url)
        response.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(shapefile_dir)
        print("Shapefile downloaded and extracted successfully.")
    except Exception as e:
        print(f"Failed to download or extract shapefile: {e}")
        exit()

try:
    # 전체 주/도 데이터에서 대한민국 데이터만 필터링
    world_states = gpd.read_file(shapefile_path)
    south_korea_map = world_states[world_states['iso_a2'] == 'KR']
    if south_korea_map.empty:
        raise ValueError("Could not find South Korea in the downloaded shapefile.")
except Exception as e:
    print(f"Failed to read or filter shapefile: {e}")
    exit()

# --- 3. 데이터 로딩 ---
seed42_files = {
    "5%_seed42": os.path.join(input_dir, "5%seed42.csv"),
    "10%_seed42": os.path.join(input_dir, "10%seed42.csv"),
    "15%_seed42": os.path.join(input_dir, "15%seed42.csv"),
    "20%_seed42": os.path.join(input_dir, "20%seed42.csv")
}
datasets = {}
try:
    for name, path in seed42_files.items():
        datasets[name] = pd.read_csv(path)
    print("All seed42 data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: A required data file was not found. {e}")
    exit()

# --- 4. 정사각형 격자 생성 (UTM 좌표계 사용) ---
print("Creating square grid...")
korea_utm = south_korea_map.to_crs(epsg=5186)
min_x, min_y, max_x, max_y = korea_utm.total_bounds

polygons = []
for x in range(int(min_x), int(max_x), cell_size_meters):
    for y in range(int(min_y), int(max_y), cell_size_meters):
        polygons.append(Polygon([
            (x, y), (x + cell_size_meters, y),
            (x + cell_size_meters, y + cell_size_meters), (x, y + cell_size_meters)
        ]))

grid_utm = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:5186")
# 대한민국 지도 영역과 겹치는 격자만 필터링
grid_utm_clipped = gpd.overlay(grid_utm, korea_utm, how='intersection')
grid_wgs84 = grid_utm_clipped.to_crs(epsg=4326)
grid_wgs84['grid_id'] = range(len(grid_wgs84))
print(f"Grid with {len(grid_wgs84)} cells created.")

# --- 5. 각 데이터셋 Folium 시각화 ---
for name, df in datasets.items():
    print(f"Processing and creating Folium map for: {name}")
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    if df.empty:
        print(f"Skipping empty dataframe: {name}")
        continue

    points_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )

    join_gdf = gpd.sjoin(grid_wgs84, points_gdf, how="inner", predicate="contains")
    count_by_grid = join_gdf.groupby('grid_id')['index_right'].count().reset_index(name='count')
    grid_with_counts = grid_wgs84.merge(count_by_grid, on='grid_id', how='left').fillna(0)

    # Folium 지도 생성
    m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles="CartoDB positron")

    # Choropleth 레이어로 격자 시각화
    folium.Choropleth(
        geo_data=grid_with_counts.to_json(),
        name='Station Count',
        data=grid_with_counts,
        columns=['grid_id', 'count'],
        key_on='feature.properties.grid_id',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name='Number of Charging Stations',
        highlight=True
    ).add_to(m)
    
    # 대한민국 행정구역 경계선 추가
    folium.GeoJson(
        south_korea_map,
        name='South Korea Boundary',
        style_function=lambda x: {'color': 'black', 'weight': 1.5, 'fillOpacity': 0}
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # 파일로 저장
    filename = os.path.join(output_dir, f'map_{name}.html')
    m.save(filename)
    print(f"Saved Folium map for '{name}' to {filename}")

print("\nAll visualizations have been generated successfully.")