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

# [수정] Matplotlib 폰트를 'Times New Roman'으로 설정
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix' # Times New Roman과 유사한 수학 폰트 설정

# ==============================================================================
# 2. 격자 생성 함수 (Grid Creation Function)
# ==============================================================================
def create_korea_mainland_grid(step_size=0.1, area_threshold=0.1):
    """대한민국 본토를 포함하는 지리적 격자(Grid)를 생성합니다."""
    print("기본 격자 지도를 생성합니다...")
    try:
        sk_geojson_url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo_simple.json"
        sk_gdf = gpd.read_file(sk_geojson_url)
    except Exception as e:
        print(f"대한민국 지도 데이터를 불러오는 데 실패했습니다: {e}")
        return None
        
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
    """
    좌표 데이터를 읽어 격자별 통계(충전기 수 합계, 백분율)를 계산하고 GeoDataFrame으로 반환합니다.
    '충전기개수' 또는 'count' 열이 있으면 해당 값을 합산하고, 없으면 행 개수를 셉니다.
    """
    filename = os.path.basename(full_filepath)
    print(f"\n----- '{filename}' 파일 처리 시작 -----")
    try:
        df = pd.read_csv(full_filepath, encoding='utf-8')
        
        # [수정] 전기차 데이터의 'Latitude', 'Longitude' 컬럼을 우선적으로 처리
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
             pass # 컬럼 이름이 이미 올바름
        elif '위도' in df.columns and '경도' in df.columns:
            df.rename(columns={'위도': 'Latitude', '경도': 'Longitude'}, inplace=True)
        else:
             print(f"-> 좌표 컬럼(Latitude, Longitude 또는 위도, 경도)을 찾을 수 없습니다."); return None
         
        count_col = None
        if '충전기개수' in df.columns:
            count_col = '충전기개수'
        elif 'count' in df.columns:
            count_col = 'count'

        if count_col:
            print(f"-> '{count_col}' 열의 **값 합산**을 기준으로 집계합니다.")
            df.rename(columns={count_col: 'value_to_sum'}, inplace=True)
            df['value_to_sum'] = pd.to_numeric(df['value_to_sum'], errors='coerce').fillna(0)
            df = df[df['value_to_sum'] > 0].copy()
        else:
            print("-> 계수 열이 없어 **행 개수**를 기준으로 집계합니다.")
            df['value_to_sum'] = 1
             
        required_cols = ['Latitude', 'Longitude', 'value_to_sum']
        
        # [수정] 요청한 3개 열만 사용하도록 필터링 (다른 열이 없어도 진행 가능)
        df_clean = df[required_cols].dropna(subset=['Latitude', 'Longitude']).copy()
        
        stations_gdf = gpd.GeoDataFrame(
            df_clean, 
            geometry=gpd.points_from_xy(df_clean.Longitude, df_clean.Latitude), 
            crs="EPSG:4326"
        )
        
        print(f"총 {len(stations_gdf)}개 위치의 데이터를 처리합니다.")
        if stations_gdf.empty: return None

        joined_gdf = gpd.sjoin(stations_gdf, base_grid_gdf, how="inner", predicate='within')
        
        station_counts = joined_gdf.groupby('grid_id')['value_to_sum'].sum().reset_index()
        station_counts.rename(columns={'value_to_sum': 'station_count'}, inplace=True)

        total_stations_value = stations_gdf['value_to_sum'].sum()
        station_counts['percentage'] = (station_counts['station_count'] / total_stations_value) * 100 if total_stations_value > 0 else 0
        
        merged_gdf = base_grid_gdf.merge(station_counts, on='grid_id', how='left')
        merged_gdf[['station_count', 'percentage']] = merged_gdf[['station_count', 'percentage']].fillna(0)
        merged_gdf['station_count'] = merged_gdf['station_count'].astype(int)
        
        return merged_gdf
        
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
        f.write("충전소 후보군과 기존 인프라 분포 유사성 분석 보고서\n")
        f.write("="*60 + "\n")
        f.write("1. 2D 상관계수 (패턴 유사성): 두 분포의 공간적 패턴이 얼마나 유사한지 (1에 가까울수록 유사)\n")
        f.write("2. 평균 절대 오차 (MAE, 값 차이): 격자별 백분율 값의 차이가 평균적으로 얼마나 나는지 (0에 가까울수록 유사)\n\n")
        for candidate_name in candidate_files:
            if candidate_name not in processed_data: continue
            f.write(f"### '{candidate_name}' 후보군과 기존 인프라 비교 ###\n")
            candidate_df = processed_data[candidate_name]
            
            # [수정] base_files 목록에 EV 데이터가 포함되어 자동으로 함께 분석됨
            for base_name in base_files:
                if base_name not in processed_data: continue
                base_df = processed_data[base_name]
                merged_df = pd.merge(candidate_df[['grid_id', 'station_count', 'percentage']], 
                                     base_df[['grid_id', 'station_count', 'percentage']], 
                                     on='grid_id', suffixes=('_candidate', '_base'))
                
                if merged_df['station_count_candidate'].nunique() < 2 or merged_df['station_count_base'].nunique() < 2:
                    correlation = float('nan')
                else:
                    correlation = merged_df['station_count_candidate'].corr(merged_df['station_count_base'])
                    
                mae = mean_absolute_error(merged_df['percentage_base'], merged_df['percentage_candidate'])
                f.write(f"- '{base_name}'과의 비교:\n")
                f.write(f"   - 2D 상관계수: {correlation:.4f}\n")
                f.write(f"   - 평균 절대 오차 (MAE): {mae:.4f}%\n")
            f.write("\n")
    print("분석 보고서 저장 완료.")

# ==============================================================================
# 5. 밀도 지도 시각화 함수 (Density Map Visualization Function)
# ==============================================================================
def create_density_map(processed_gdf, filename, output_folder, global_vmax):
    """데이터의 공간 분포를 나타내는 밀도 지도를 생성합니다. (선형 스케일)"""
    print(f"\n----- '{filename}' 밀도 지도 생성 시작 -----")
    
    processed_gdf['percentage_str'] = processed_gdf['percentage'].round(2).astype(str) + '%'
    
    cm = colormap.linear.YlOrRd_09.scale(vmin=0, vmax=global_vmax if global_vmax > 0 else 1)
    cm.caption = f'Charger Distribution (Shared Scale based on 99th Percentile, Max: {global_vmax:.2f}%)'
    
    style_function = lambda x: {
        'fillColor': cm(x['properties']['percentage']) if x['properties']['station_count'] > 0 else 'transparent',
        'color': 'black' if x['properties']['station_count'] > 0 else 'transparent',
        'weight': 0.5,
        'fillOpacity': 0.8 if x['properties']['station_count'] > 0 else 0.0,
    }
    tooltip = folium.features.GeoJsonTooltip(
        fields=['grid_id', 'station_count', 'percentage_str'],
        aliases=['격자 ID:', '충전기 수:', '백분율:'],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
    
    for theme, tile in {'light': 'CartoDB positron', 'dark': 'CartoDB dark_matter'}.items():
        m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles=tile, control_scale=True)
        folium.GeoJson(processed_gdf.to_json(), style_function=style_function, tooltip=tooltip).add_to(m)
        
        # [수정] 사용자 요청에 따라 지도에 스케일바(범례) 추가
        # m.add_child(cm)
        
        base_name = os.path.splitext(filename)[0]
        suffix = '_dark' if theme == 'dark' else ''
        output_path = os.path.join(output_folder, f"map_density_{base_name}{suffix}.html")
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
        
        # [수정] base_files 목록에 EV 데이터가 포함되어 자동으로 함께 비교 맵 생성
        for base_name in base_files:
            if base_name not in processed_data: continue
            
            candidate_df = processed_data[candidate_name]
            base_df = processed_data[base_name]
            
            merged_df = pd.merge(candidate_df[['grid_id', 'geometry', 'percentage']], 
                                 base_df[['grid_id', 'percentage']], 
                                 on='grid_id', suffixes=('_candidate', '_base'))
            
            merged_df['difference'] = merged_df['percentage_candidate'] - merged_df['percentage_base']
            merged_df['difference_str'] = merged_df['difference'].round(2).astype(str) + '%'

            cm = colormap.linear.RdBu_11.scale(vmin=-global_v_max, vmax=global_v_max)
            cm.caption = f'Distribution Difference (%p, Candidate - Base)'

            style_function = lambda x: {
                'fillColor': cm(x['properties']['difference']),
                'color': 'black' if x['properties']['difference'] != 0 else 'transparent',
                'fillOpacity': 0.7 if x['properties']['difference'] != 0 else 0,
                'weight': 0.5
            }
            tooltip = folium.features.GeoJsonTooltip(
                fields=['grid_id', 'difference_str'], aliases=['격자 ID:', '차이 (후보-기존):'],
                style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            )

            for theme, tile in {'light': 'CartoDB positron', 'dark': 'CartoDB dark_matter'}.items():
                m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles=tile, control_scale=True)
                folium.GeoJson(merged_df.to_json(), style_function=style_function, tooltip=tooltip).add_to(m)
                
                # [수정] 사용자 요청에 따라 지도에 스케일바(범례) 추가
                # m.add_child(cm)
                
                c_name = os.path.splitext(candidate_name)[0].replace('.', '')
                b_name = os.path.splitext(base_name)[0]
                suffix = '_dark' if theme == 'dark' else ''
                output_path = os.path.join(output_folder, f"map_diff_{c_name}_vs_{b_name}{suffix}.html")
                m.save(output_path)
                print(f"-> '{output_path}' ({theme} 테마) 저장 완료.")

# ==============================================================================
# 7. 컬러바 이미지 생성 함수 (Colorbar Image Generation Function)
# ==============================================================================
def create_colorbar_image(colormap_name, vmin, vmax, label, output_path):
    """
    주어진 스케일과 컬러맵으로 수평 컬러바 이미지를 생성하여 저장합니다.
    [수정] 폰트는 'Times New Roman'으로 설정됩니다 (코드 1번 섹션에서 전역 설정됨).
    """
    print(f"\n----- 컬러바 이미지 생성: '{os.path.basename(output_path)}' -----")
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap_name)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal', label=label)
    
    # 전역 폰트 설정(Times New Roman)이 적용됨
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
# 8. [추가] 총 충전기 개수 보고서 생성 함수
# ==============================================================================
def save_total_counts_report(processed_data, all_files_list, candidate_files_list, output_folder):
    """각 데이터셋의 총 충전기 개수를 텍스트 파일로 저장합니다."""
    report_path = os.path.join(output_folder, 'total_charger_counts_report.txt')
    print(f"\n----- 총 충전기 개수 보고서 생성 (결과 파일: {report_path}) -----")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("데이터셋별 총 충전기 개수 (격자 내 합계 기준)\n")
        f.write("="*50 + "\n")
        
        # 파일을 기준 인프라 / 후보군으로 분리하여 정렬
        base_files_sorted = sorted([f for f in all_files_list if f not in candidate_files_list and f in processed_data])
        candidate_files_sorted = sorted([f for f in all_files_list if f in candidate_files_list and f in processed_data])
        
        f.write("\n### 기준 인프라 (Base Infrastructure) ###\n")
        for fname in base_files_sorted:
            total_count = processed_data[fname]['station_count'].sum()
            f.write(f"- {fname}: {total_count}\n")

        f.write("\n### 후보군 (Candidate Sets) ###\n")
        for fname in candidate_files_sorted:
            total_count = processed_data[fname]['station_count'].sum()
            f.write(f"- {fname}: {total_count}\n")
            
    print("총 충전기 개수 보고서 저장 완료.")


# ==============================================================================
# 9. 메인 실행 코드 (Main Execution) - [수정됨]
# ==============================================================================
# -------------------- 사용자 설정 --------------------
# 원본 파일들이 위치한 폴더 
INPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Installed-infra" 

# 새로운 수소 충전소 분석 결과를 저장할 폴더
OUTPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis\Installed-infra\Infrastructure_Comparison(charger)"

# 분석할 후보군 파일 목록
CANDIDATE_FILES = ["2%.csv", "5%.csv", "10%.csv", "15%.csv", "20%.csv"]

# [추가] 분석할 전기차 충전소 파일 목록 (기준 인프라에 포함됨)
EV_FILES = ["Electric_station_charger(ALL).csv", "Electric_station_charger(Logistic).csv"]
# ----------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"결과 저장 폴더: '{OUTPUT_DIR}'")

# Step 0: 수소/전기 충전소 데이터 불러오기 및 세분화
print("\n----- 기준 인프라 데이터 준비 시작 -----")
# [수정] new_base_files 리스트에 EV_FILES를 먼저 추가
new_base_files = EV_FILES.copy()
print(f"-> 전기차 충전소 파일 {len(new_base_files)}개를 기본 분석 파일로 추가합니다.")

main_hydrogen_file = 'Hydrogen_station_charger.csv'
main_hydrogen_path = os.path.join(INPUT_DIR, main_hydrogen_file)

# [추가] 생성된 수소차 파일 이름을 저장할 리스트
GENERATED_H2_FILES = [] 

try:
    h2_df = pd.read_csv(main_hydrogen_path, encoding='utf-8')
    new_base_files.append(main_hydrogen_file) # 원본 수소 파일 추가
    
    bus_truck_df = h2_df[h2_df['충전가능차량'].str.contains('버스|트럭', na=False)].copy()
    bus_truck_file = 'hydrogen_bus_truck.csv'
    bus_truck_path = os.path.join(OUTPUT_DIR, bus_truck_file)
    bus_truck_df.to_csv(bus_truck_path, index=False, encoding='utf-8-sig')
    print(f"-> '버스/트럭' 데이터 {len(bus_truck_df)}개 추출하여 '{bus_truck_path}'에 저장 완료.")
    new_base_files.append(bus_truck_file)
    GENERATED_H2_FILES.append(bus_truck_file) # 생성된 파일 목록에 추가

    liquid_df = h2_df[h2_df['구분'].str.contains('액화', na=False)].copy()
    liquid_file = 'hydrogen_liquefied.csv'
    liquid_path = os.path.join(OUTPUT_DIR, liquid_file)
    liquid_df.to_csv(liquid_path, index=False, encoding='utf-8-sig')
    print(f"-> '액화' 데이터 {len(liquid_df)}개 추출하여 '{liquid_path}'에 저장 완료.")
    new_base_files.append(liquid_file)
    GENERATED_H2_FILES.append(liquid_file) # 생성된 파일 목록에 추가

    combined_condition = (h2_df['충전가능차량'].str.contains('버스|트럭', na=False)) | (h2_df['구분'].str.contains('액화', na=False))
    combined_df = h2_df[combined_condition].copy()
    combined_file = 'hydrogen_combined.csv'
    combined_path = os.path.join(OUTPUT_DIR, combined_file)
    combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"-> '버스/트럭 또는 액화' 통합 데이터 {len(combined_df)}개 추출하여 '{combined_path}'에 저장 완료.")
    new_base_files.append(combined_file)
    GENERATED_H2_FILES.append(combined_file) # 생성된 파일 목록에 추가

except FileNotFoundError:
    print(f"경고: 기본 수소 충전소 파일 '{main_hydrogen_path}'을 찾을 수 없습니다. 전기차 데이터로만 계속 진행합니다.")
except Exception as e:
    print(f"수소 충전소 데이터 처리 중 오류 발생: {e}. 전기차 데이터로만 계속 진행합니다.")

# Step 1: 기본 격자 생성
grid = create_korea_mainland_grid(step_size=0.1, area_threshold=0.1)
if grid is None:
    print("격자 생성에 실패하여 프로그램을 종료합니다.")
else:
    # Step 2: 모든 파일 데이터 처리
    # [수정] new_base_files (H2+EV)와 CANDIDATE_FILES (전동화율)를 합쳐 전체 목록 생성
    all_files_to_process = new_base_files + CANDIDATE_FILES
    processed_results = {}
    
    for fname in all_files_to_process:
        # [수정] 파일 경로 로직 변경
        # Step 0에서 생성된 파일은 OUTPUT_DIR, 그 외(원본 H2, EV, 후보군)는 INPUT_DIR
        if fname in GENERATED_H2_FILES:
            fpath = os.path.join(OUTPUT_DIR, fname)
        else:
            fpath = os.path.join(INPUT_DIR, fname)
            
        if not os.path.exists(fpath):
            print(f"경고: '{fpath}' 파일을 찾을 수 없어 건너뜁니다.")
            continue
            
        result_gdf = process_station_data(fpath, grid.copy())
        if result_gdf is not None:
            processed_results[fname] = result_gdf
            
    # Step 3: 시각화를 위한 전역 스케일 계산
    print("\n----- 전역 시각화 스케일 계산 (이상치 영향 감소) -----")
    if processed_results:
        # [수정] EV 데이터를 포함한 모든 처리 결과(processed_results)로 스케일 계산
        all_percentages = pd.concat([gdf['percentage'] for gdf in processed_results.values()])
        
        q99 = all_percentages[all_percentages > 0].quantile(0.99)
        global_density_vmax = q99 if pd.notna(q99) and q99 > 0 else all_percentages.max()

        all_diffs = []
        # [수정] 후보군(CANDIDATE_FILES) vs 모든 기준 인프라(new_base_files) 간의 차이 계산
        for c_name in CANDIDATE_FILES:
            for b_name in new_base_files: # new_base_files는 H2와 EV를 모두 포함
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
    else:
        global_density_vmax = 0
        global_diff_vmax = 0
        print("처리된 데이터가 없어 스케일 계산을 건너뜁니다.")
        
    print(f"밀도 지도 스케일 (상위 99% 백분율 기준 최대값): {global_density_vmax:.4f}%")
    print(f"차이 지도 스케일 (상위 95% 절대값 차이 기준 최대값): {global_diff_vmax:.4f}%")
            
    # Step 4: 수치 분석 실행
    # [수정] new_base_files가 EV 데이터를 포함하므로 자동으로 확장 분석됨
    analyze_distribution_similarity(processed_results, new_base_files, CANDIDATE_FILES, OUTPUT_DIR)
    
    # Step 5: 지도 시각화 실행
    # [수정] EV 데이터가 포함된 모든 결과에 대해 밀도 지도 생성
    for filename, gdf in processed_results.items():
        create_density_map(gdf, filename, OUTPUT_DIR, global_density_vmax)
    # [수정] EV 데이터가 포함된 new_base_files 기준으로 차이 지도 생성
    create_difference_map(processed_results, new_base_files, CANDIDATE_FILES, OUTPUT_DIR, global_diff_vmax)
    
    # Step 6: 컬러바 이미지 생성
    create_colorbar_image('YlOrRd', 0, global_density_vmax, 'Charger Distribution Density (%)', os.path.join(OUTPUT_DIR, 'colorbar_density.png'))
    create_colorbar_image('RdBu_r', -global_diff_vmax, global_diff_vmax, 'Distribution Difference (%p)', os.path.join(OUTPUT_DIR, 'colorbar_difference.png'))

    # Step 7: [추가] 총 충전기 개수 보고서 생성
    save_total_counts_report(processed_results, all_files_to_process, CANDIDATE_FILES, OUTPUT_DIR)

print("\n모든 작업이 완료되었습니다.")