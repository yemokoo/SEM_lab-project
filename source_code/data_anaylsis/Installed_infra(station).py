# ==============================================================================
# 1. 라이브러리 불러오기 (Import Libraries)
# ==============================================================================
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from branca import colormap
import numpy as np
import math
import os
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==============================================================================
# 2. 격자 생성 함수 (Grid Creation Function)
# ==============================================================================
def create_korea_mainland_grid(step_size=0.1, area_threshold=0.1):
    """대한민국 본토를 포함하는 지리적 격자(Grid)를 생성합니다."""
    print("기본 격자 지도를 생성합니다...")
    sk_geojson_url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json"
    sk_gdf = gpd.read_file(sk_geojson_url)
    mainland_gdf = sk_gdf[sk_gdf['name'] != 'Jeju-do'].copy()
    mainland_boundary = mainland_gdf.union_all()
    
    lat_min, lat_max, lon_min, lon_max = 33.0, 38.8, 125.5, 129.8
    center_lat = (lat_min + lat_max) / 2
    lon_step = step_size / math.cos(math.radians(center_lat))
    lat_bins = np.arange(lat_min, lat_max, step_size)
    lon_bins = np.arange(lon_min, lon_max, lon_step)
    
    features = []
    grid_index = 0
    for lat in lat_bins:
        for lon in lon_bins:
            cell_polygon = Polygon([(lon, lat), (lon + lon_step, lat), (lon + lon_step, lat + step_size), (lon, lat + step_size)])
            if mainland_boundary.intersects(cell_polygon):
                intersection_geom = mainland_boundary.intersection(cell_polygon)
                if (intersection_geom.area / cell_polygon.area) >= area_threshold:
                    features.append({
                        'type': 'Feature', 'geometry': cell_polygon.__geo_interface__,
                        'properties': {'grid_id': grid_index}
                    })
                    grid_index += 1
                    
    if not features: return None
    grid_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print(f"총 {grid_index}개의 격자 생성 완료.")
    return grid_gdf

# ==============================================================================
# 3. 데이터 처리 함수 (Data Processing Function)
# ==============================================================================
def process_station_data(full_filepath, base_grid_gdf):
    """좌표 데이터를 읽어 격자별 통계(개수, 백분율)를 계산하고 GeoDataFrame으로 반환합니다."""
    filename = os.path.basename(full_filepath)
    print(f"\n----- '{filename}' 파일 처리 시작 -----")
    try:
        df = pd.read_csv(full_filepath, encoding='utf-8')
        if '%' in filename and 'count' in df.columns:
            original_rows = len(df)
            df = df[df['count'] > 0].copy()
            print(f"-> 'count'가 0인 {original_rows - len(df)}개 행 필터링.")
            
        if '위도' in df.columns: df.rename(columns={'위도': 'Latitude', '경도': 'Longitude'}, inplace=True)
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print(f"-> 좌표 컬럼을 찾을 수 없습니다."); return None

        df = df[['Latitude', 'Longitude']].dropna()
        stations_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
        print(f"총 {len(stations_gdf)}개의 좌표 데이터를 처리합니다.")
        if stations_gdf.empty: return None

        joined_gdf = gpd.sjoin(base_grid_gdf, stations_gdf, how="left", predicate='contains')
        station_counts = joined_gdf.groupby('grid_id')['index_right'].count().reset_index(name='station_count')
        total_stations = len(stations_gdf)
        station_counts['percentage'] = (station_counts['station_count'] / total_stations) * 100 if total_stations > 0 else 0
        
        return base_grid_gdf.merge(station_counts, on='grid_id', how='left').fillna(0)
    except Exception as e:
        print(f"파일('{filename}') 처리 중 오류 발생: {e}"); return None

# ==============================================================================
# 4. 수치 분석 함수 (Quantitative Analysis Function)
# ==============================================================================
def analyze_distribution_similarity(processed_data, base_files, candidate_files, output_folder):
    """분포 유사성을 2D 상관계수와 MAE로 계산하고 결과를 txt 파일로 저장합니다."""
    report_path = os.path.join(output_folder, 'distribution_comparison_report.txt')
    print(f"\n----- 분포 유사성 분석 시작 (결과 파일: {report_path}) -----")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("화물차 충전소 후보군과 기존 인프라 분포 유사성 분석 보고서\n")
        f.write("="*60 + "\n")
        f.write("1. 2D 상관계수 (패턴 유사성): 두 분포의 공간적 패턴이 얼마나 유사한지 (1에 가까울수록 유사)\n")
        f.write("2. 평균 절대 오차 (MAE, 값 차이): 격자별 백분율 값의 차이가 평균적으로 얼마나 나는지 (0에 가까울수록 유사)\n\n")
        for candidate_name in candidate_files:
            if candidate_name not in processed_data: continue
            f.write(f"### '{candidate_name}' 후보군과 기존 인프라 비교 ###\n")
            candidate_df = processed_data[candidate_name]
            for base_name in base_files:
                if base_name not in processed_data: continue
                base_df = processed_data[base_name]
                merged_df = pd.merge(candidate_df[['grid_id', 'station_count', 'percentage']], 
                                     base_df[['grid_id', 'station_count', 'percentage']], 
                                     on='grid_id', suffixes=('_candidate', '_base'))
                correlation = merged_df['station_count_candidate'].corr(merged_df['station_count_base'])
                mae = mean_absolute_error(merged_df['percentage_base'], merged_df['percentage_candidate'])
                f.write(f"- '{base_name}'과의 비교:\n")
                f.write(f"  - 2D 상관계수: {correlation:.4f}\n")
                f.write(f"  - 평균 절대 오차 (MAE): {mae:.4f}%\n")
            f.write("\n")
    print("분석 보고서 저장 완료.")

# ==============================================================================
# 5. 밀도 지도 시각화 함수 (Density Map Visualization Function)
# ==============================================================================
def create_density_map(processed_gdf, filename, output_folder, global_vmax):
    """데이터의 공간 분포를 나타내는 밀도 지도를 생성합니다."""
    print(f"\n----- '{filename}' 밀도 지도 생성 시작 -----")
    
    processed_gdf['percentage_str'] = processed_gdf['percentage'].round(2).astype(str) + '%'
    cm = colormap.linear.YlOrRd_09.scale(vmin=0, vmax=global_vmax if global_vmax > 0 else 1)
    cm.caption = f'Percentage of Total Stations (Shared Scale, Max: {global_vmax:.2f}%)'
    
    style_function = lambda x: {
        'fillColor': cm(x['properties']['percentage']) if x['properties']['station_count'] > 0 else 'transparent',
        'color': 'black', 'weight': 0.5,
        'fillOpacity': 0.8 if x['properties']['station_count'] > 0 else 0.0,
    }
    tooltip = folium.features.GeoJsonTooltip(
        fields=['grid_id', 'station_count', 'percentage_str'],
        aliases=['격자 ID:', '개수:', '백분율:'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
    
    for theme, tile in {'light': 'CartoDB positron', 'dark': 'CartoDB dark_matter'}.items():
        m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles=tile)
        folium.GeoJson(processed_gdf, style_function=style_function, tooltip=tooltip).add_to(m)
        m.add_child(cm)
        
        base_name = os.path.splitext(filename)[0]
        suffix = '_dark' if theme == 'dark' else ''
        output_path = os.path.join(output_folder, f"map_{base_name}{suffix}.html")
        m.save(output_path)
        print(f"-> '{output_path}' ({theme} 테마) 저장 완료.")

# ==============================================================================
# 6. 분포 차이 시각화 함수 (Difference Map Visualization Function)
# ==============================================================================
def create_difference_map(processed_data, base_files, candidate_files, output_folder, global_v_max):
    """두 데이터셋 간의 공간 분포 차이를 시각화하는 지도를 생성합니다."""
    print("\n----- 분포 차이 지도 생성 시작 -----")
    
    for candidate_name in candidate_files:
        if candidate_name not in processed_data: continue
        for base_name in base_files:
            if base_name not in processed_data: continue
            
            candidate_df = processed_data[candidate_name]
            base_df = processed_data[base_name]
            
            merged_df = pd.merge(candidate_df[['grid_id', 'geometry', 'percentage']], 
                                 base_df[['grid_id', 'percentage']], 
                                 on='grid_id', suffixes=('_candidate', '_base'))
            
            merged_df['difference'] = merged_df['percentage_candidate'] - merged_df['percentage_base']
            merged_df['difference_str'] = merged_df['difference'].round(2).astype(str) + '%'

            reversed_colors = colormap.linear.RdBu_11.colors[::-1]
            cm = colormap.LinearColormap(colors=reversed_colors, vmin=-global_v_max, vmax=global_v_max)
            cm.caption = f'Distribution Difference (%p, Range: -{global_v_max:.2f}% ~ +{global_v_max:.2f}%)'

            style_function = lambda x: {
                'fillColor': cm(x['properties']['difference']),
                'color': 'black' if x['properties']['difference'] != 0 else 'transparent',
                'fillOpacity': 0.7 if x['properties']['difference'] != 0 else 0,
                'weight': 0.5
            }
            tooltip = folium.features.GeoJsonTooltip(
                fields=['grid_id', 'difference_str'], aliases=['격자 ID:', '차이:'],
                style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            )

            for theme, tile in {'light': 'CartoDB positron', 'dark': 'CartoDB dark_matter'}.items():
                m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles=tile)
                folium.GeoJson(merged_df, style_function=style_function, tooltip=tooltip).add_to(m)
                m.add_child(cm)
                
                c_name = os.path.splitext(candidate_name)[0]
                b_name = os.path.splitext(base_name)[0]
                suffix = '_dark' if theme == 'dark' else ''
                output_path = os.path.join(output_folder, f"map_diff_{c_name}_vs_{b_name}{suffix}.html")
                m.save(output_path)
                print(f"-> '{output_path}' ({theme} 테마) 저장 완료.")

# ==============================================================================
# 7. 컬러바 이미지 생성 함수 (Colorbar Image Generation Function)
# ==============================================================================
def create_colorbar_image(colormap, vmin, vmax, label, output_path):
    """주어진 스케일과 컬러맵으로 수평 컬러바 이미지를 생성하여 저장합니다."""
    print(f"\n----- 컬러바 이미지 생성: '{os.path.basename(output_path)}' -----")
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal', label=label)
    
    cb.ax.tick_params(labelsize=10)
    cb.set_label(label, size=12, weight='bold')

    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
        print(f"-> 컬러바 이미지 저장 완료: '{output_path}'")
    except Exception as e:
        print(f"-> 컬러바 이미지 저장 실패: {e}")
    finally:
        plt.close(fig)

# ==============================================================================
# 8. 메인 실행 코드 (Main Execution)
# ==============================================================================
# -------------------- 사용자 설정 --------------------
INPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Installed-infra"
OUTPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Installed-infra\Infrastructure_Comparison"
BASE_FILES = ["Hydrogen_station.csv", "Electric_station.csv"]
CANDIDATE_FILES = ["2%.csv", "5%.csv", "10%.csv", "15%.csv", "20%.csv"]
# ----------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"결과 저장 폴더: '{OUTPUT_DIR}'")

# Step 1: 기본 격자 생성
grid = create_korea_mainland_grid(step_size=0.1, area_threshold=0.1)
if grid is None:
    print("격자 생성에 실패하여 프로그램을 종료합니다.")
else:
    # Step 2: 모든 파일 데이터 처리
    all_files_to_process = BASE_FILES + CANDIDATE_FILES
    processed_results = {}
    for fname in all_files_to_process:
        result_gdf = process_station_data(os.path.join(INPUT_DIR, fname), grid.copy())
        if result_gdf is not None:
            processed_results[fname] = result_gdf
            
    # Step 3: 시각화를 위한 전역 스케일 계산
    print("\n----- 전역 시각화 스케일 계산 -----")
    global_density_vmax = max(gdf['percentage'].max() for gdf in processed_results.values()) if processed_results else 0
    
    all_diffs = []
    for c_name in CANDIDATE_FILES:
        for b_name in BASE_FILES:
            if c_name in processed_results and b_name in processed_results:
                merged = pd.merge(processed_results[c_name][['grid_id', 'percentage']],
                                  processed_results[b_name][['grid_id', 'percentage']],
                                  on='grid_id', suffixes=('_c', '_b'))
                all_diffs.extend((merged['percentage_c'] - merged['percentage_b']).tolist())
    
    if all_diffs:
        abs_diffs = pd.Series(all_diffs).abs()
        q95 = abs_diffs[abs_diffs > 0].quantile(0.95)
        global_diff_vmax = q95 if pd.notna(q95) and q95 > 0 else abs_diffs.max()
    else:
        global_diff_vmax = 0
        
    print(f"밀도 지도 스케일 (최대 %): {global_density_vmax:.4f}")
    print(f"차이 지도 스케일 (최대 절대값 차이 %): {global_diff_vmax:.4f}")
            
    # Step 4: 수치 분석 실행
    analyze_distribution_similarity(processed_results, BASE_FILES, CANDIDATE_FILES, OUTPUT_DIR)
    
    # Step 5: 지도 시각화 실행
    for filename, gdf in processed_results.items():
        create_density_map(gdf, filename, OUTPUT_DIR, global_density_vmax)
    create_difference_map(processed_results, BASE_FILES, CANDIDATE_FILES, OUTPUT_DIR, global_diff_vmax)
    
    # Step 6: 컬러바 이미지 생성
    create_colorbar_image('YlOrRd', 0, global_density_vmax, 'Distribution Density (%)', os.path.join(OUTPUT_DIR, 'colorbar_density.png'))
    create_colorbar_image('RdBu_r', -global_diff_vmax, global_diff_vmax, 'Distribution Difference (%p)', os.path.join(OUTPUT_DIR, 'colorbar_difference.png'))

print("\n모든 작업이 완료되었습니다.")