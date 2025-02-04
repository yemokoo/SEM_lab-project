import os
import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import platform

# 초기 값 설정
num_of_candidate = 500

# 파일 경로들
OD_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\OD"
interval_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\interval.csv"
infra_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\infra.xlsx"
traffic_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\traffic"
link_centroids_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\link_centroids.csv"
rest_area_link_ids_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate\rest_area.csv"

# 저장 경로 설정
output_dir = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra"

candidate_multiple = 2

def OD_candidate_selection(OD_file_path, candidate_multiple, num_of_candidate, link_centroids_file_path):
    # OD 데이터 불러오기 (폴더 내 모든 CSV 파일 읽기)
    all_files = os.listdir(OD_file_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]

    OD_df_list = []
    for file in csv_files:
        file_path = os.path.join(OD_file_path, file)
        OD_df_list.append(pd.read_csv(file_path))

    OD_df = pd.concat(OD_df_list, ignore_index=True)
    
    # LINK_ID별 COUNT 계산
    link_counts_df = pd.concat([OD_df['ORIGIN_LINK_ID'], OD_df['DESTINATION_LINK_ID']], axis=0).rename('LINK_ID').value_counts().reset_index()
    link_counts_df.columns = ['LINK_ID', 'COUNT']

    # link_centroids 데이터 불러오기
    link_centroids_df = pd.read_csv(link_centroids_file_path)

    # Inner Join 수행
    joined_df = link_counts_df.merge(link_centroids_df, on="LINK_ID", how="inner")

    # COUNT 기준으로 정렬 (내림차순)
    sorted_df = joined_df.sort_values(by='COUNT', ascending=False)

    # 상위 candidate_multiple * num_of_candidate 개수만큼 선택
    num_select = int(candidate_multiple * num_of_candidate)
    representative_links_df = sorted_df.head(num_select).copy()
    representative_links_df['LINK_ID'] = pd.to_numeric(representative_links_df['LINK_ID'])
    representative_links_df['OD'] = 1  # category 열 추가
    representative_links_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\OD.csv", index=False)  # 최종 결과 CSV 파일 저장 (LINK_ID만)
    return representative_links_df[['LINK_ID', 'OD']]  # 링크 아이디와 category 열만 반환

def infra_candidate_selection(infra_file_path, candidate_multiple, num_of_candidate, link_centroids_file_path):
    # 엑셀 파일에서 필요한 열만 읽어오기
    infra_df = pd.read_excel(infra_file_path, usecols=['종류', 'size', 'LINK_ID'])

    # size 열을 숫자형으로 변환 (오류 처리 포함)
    infra_df['size'] = pd.to_numeric(infra_df['size'], errors='coerce')

    # '물류창고'가 아닌 경우 size를 -1로 설정
    infra_df.loc[infra_df['종류'] != '물류창고', 'size'] = -1

    # 종류별로 열 생성 및 값 채우기 (필요한 종류만 고려)
    for category in ['물류기지', '공영차고지', '물류창고']:
        infra_df[category] = (infra_df['종류'] == category).astype(int)

    # 필요한 열만 선택 (필요한 종류만 고려)
    infra_df = infra_df[['LINK_ID', '물류기지', '공영차고지', '물류창고', 'size']]

    # '물류창고' 이외의 시설이 있는 링크 ID 추출 (필요한 종류만 고려)
    non_warehouse_link_ids = infra_df[infra_df[['물류기지', '공영차고지']].any(axis=1)]['LINK_ID']

    # '물류창고' 이외 시설 포함 링크 추가 (우선 포함)
    non_warehouse_result_df = infra_df[infra_df['LINK_ID'].isin(non_warehouse_link_ids)][['LINK_ID']].copy()

    # '물류창고'만 있는 링크 ID 추출
    warehouse_only_df = infra_df[~infra_df['LINK_ID'].isin(non_warehouse_link_ids)].copy()

    # '물류창고'만 있는 링크를 size 기준으로 내림차순 정렬
    warehouse_only_sorted_df = warehouse_only_df.sort_values(by='size', ascending=False)

    # 목표 후보 개수 계산
    target_candidates = int(candidate_multiple * num_of_candidate)

    # '물류창고'에서 추가로 선택해야 하는 개수 계산
    warehouse_candidates_needed = max(0, target_candidates - len(non_warehouse_result_df))

    # '물류창고' 중에서 size 상위 링크 선택 (필요한 개수만큼)
    warehouse_candidates_df = warehouse_only_sorted_df.head(warehouse_candidates_needed)[['LINK_ID']].copy()

    # 최종 후보 DataFrame 생성 (물류창고 외 시설 + 물류창고)
    final_result_df = pd.concat([non_warehouse_result_df, warehouse_candidates_df], ignore_index=True)

    final_result_df['infra'] = 1  # category 열 추가

    final_result_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\infra.csv", index=False)  # 최종 결과 CSV 파일 저장
    return final_result_df[['LINK_ID', 'infra']]  # 링크 아이디와 category 열만 반환

def traffic_candidate_selection(traffic_file_path, candidate_multiple, num_of_candidate, link_centroids_file_path):
    # traffic 데이터 불러오기 (폴더 내 모든 CSV 파일 읽기)
    all_files = os.listdir(traffic_file_path)
    csv_files = [file for file in all_files if file.endswith('.csv')]

    traffic_df_list = []
    for file in csv_files:
        file_path = os.path.join(traffic_file_path, file)
        traffic_df_list.append(pd.read_csv(file_path, usecols=['LINK_ID', 'AVG_DAILY_TOTAL_COUNT']))

    traffic_df = pd.concat(traffic_df_list, ignore_index=True)

    # LINK_ID, AVG_DAILY_TOTAL_COUNT 열을 숫자형으로 변환 (오류 처리 포함)
    traffic_df['LINK_ID'] = pd.to_numeric(traffic_df['LINK_ID'], errors='coerce')
    traffic_df['AVG_DAILY_TOTAL_COUNT'] = pd.to_numeric(traffic_df['AVG_DAILY_TOTAL_COUNT'], errors='coerce')

    # link_centroids 데이터 불러오기
    link_centroids_df = pd.read_csv(link_centroids_file_path)

    # Inner Join 수행
    joined_df = traffic_df.merge(link_centroids_df, on="LINK_ID", how="inner")

    # AVG_DAILY_TOTAL_COUNT 기준으로 정렬 (내림차순)
    sorted_df = joined_df.sort_values(by='AVG_DAILY_TOTAL_COUNT', ascending=False)

    # 상위 candidate_multiple * num_of_candidate 개수만큼 선택
    num_select = int(candidate_multiple * num_of_candidate)
    representative_links_df = sorted_df.head(num_select).copy()

    representative_links_df['traffic'] = 1  # category 열 추가
    representative_links_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\traffic.csv", index=False)  
    return representative_links_df[['LINK_ID', 'traffic']]  # 링크 아이디와 category 열만 반환

def interval_candidate_selection(interval_file_path):

    interval_df = pd.read_csv(interval_file_path)
    # LINK_ID 열을 정수형으로 변환 (오류 처리 포함)
    interval_df['LINK_ID'] = pd.to_numeric(interval_df['LINK_ID'])

    # 필요한 열만 선택하여 새로운 DataFrame 생성
    representative_links_df = interval_df[['LINK_ID']].copy()

    # LINK_ID 열을 정수형으로 변환 및 중복 제거
    representative_links_df['LINK_ID'] = representative_links_df['LINK_ID'].astype(int)
    representative_links_df = representative_links_df.drop_duplicates(subset=['LINK_ID'])  # 중복 제거

    representative_links_df['interval'] = 1  # interval 열 추가
    representative_links_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\interval.csv", index=False)
    return representative_links_df[['LINK_ID', 'interval']]  # 링크 아이디와 category 열만 반환

def rest_area_candidate_selection(rest_area_link_ids_path):

    rest_area_link_ids_df = pd.read_csv(rest_area_link_ids_path)

    # 열 이름을 대문자로 변경
    rest_area_link_ids_df.columns = rest_area_link_ids_df.columns.str.upper()
    rest_area_link_ids_df['rest_area'] = 1  # category 열 추가
    rest_area_link_ids_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\rest_area.csv", index=False)
    return rest_area_link_ids_df[['LINK_ID', 'rest_area']]  # 링크 아이디와 category 열만 반환

def merge_candidate_selection(OD_df, infra_df, traffic_df, interval_df, rest_area_df):
    """
    개별 후보 선정 결과를 하나의 DataFrame으로 병합합니다.

    Args:
        OD_df (pd.DataFrame): OD 기반 후보 선정 결과 DataFrame (LINK_ID, OD 컬럼 포함).
        infra_df (pd.DataFrame): Infra 기반 후보 선정 결과 DataFrame (LINK_ID, infra 컬럼 포함).
        traffic_df (pd.DataFrame): 교통량 기반 후보 선정 결과 DataFrame (LINK_ID, traffic 컬럼 포함).
        interval_df (pd.DataFrame): 간선도로 구간 기반 후보 선정 결과 DataFrame (LINK_ID, interval 컬럼 포함).
        rest_area_df (pd.DataFrame): 휴게소 기반 후보 선정 결과 DataFrame (LINK_ID, rest_area 컬럼 포함).

    Returns:
        pd.DataFrame: 모든 후보 선정 결과를 병합한 DataFrame (LINK_ID 및 각 category 컬럼 포함).
    """

    # 모든 DataFrame을 outer join으로 병합합니다.
    # outer join은 모든 LINK_ID를 포함하고, 각 DataFrame에 없는 컬럼은 NaN으로 채웁니다.
    merged_df = OD_df.merge(infra_df, on='LINK_ID', how='outer')
    merged_df = merged_df.merge(traffic_df, on='LINK_ID', how='outer')
    merged_df = merged_df.merge(interval_df, on='LINK_ID', how='outer')
    merged_df = merged_df.merge(rest_area_df, on='LINK_ID', how='outer')

    # NaN 값을 0으로 채웁니다. (해당 기준에 해당하지 않는 경우 0으로 표시)
    merged_df = merged_df.fillna(0)

    # Fill NaN values with 0 and then remove duplicate LINK_ID
    merged_df.drop_duplicates(subset=['LINK_ID'], keep='first', inplace=True)

    return merged_df

def combine_candidate_dataframes(merged_candidate_df, link_centroids_file_path, n_clusters, random_state=1):
    """
    병합된 후보 데이터프레임을 이용하여 클러스터링하고 최종 후보지를 선정하는 함수

    Args:
        merged_candidate_df (pd.DataFrame): 이미 병합된 후보 데이터프레임 (merge_candidate_selection 함수의 결과).
        link_centroids_file_path (str): 링크 중심 좌표 정보 파일 경로.
        n_clusters (int): 클러스터 개수.
        random_state (int): KMeans random state.

    Returns:
        pd.DataFrame: 최종 후보지 DataFrame.
    """

    all_candidate_df = merged_candidate_df.copy() # Use the pre-merged DataFrame

    # 링크별 카테고리 합계 'count' 열 생성
    all_categories = ['OD', 'infra', 'traffic', 'interval', 'rest_area'] # 'rest_area' 추가
    all_candidate_df['count'] = all_candidate_df[all_categories].sum(axis=1)

    # 링크 중심 좌표 정보 로드
    link_centroids_df = pd.read_csv(link_centroids_file_path)

    # 좌표 정보 병합
    merged_df = pd.merge(all_candidate_df, link_centroids_df, on='LINK_ID', how='inner')

    # 휴게소 df 와 비휴게소 df 분리
    rest_area_merged_df = merged_df[merged_df['rest_area'] == 1].copy()
    non_rest_area_merged_df = merged_df[merged_df['rest_area'] == 0].copy()

    # 숫자형 NaN 값 0으로 채우기
    num_cols = ['centroid_x', 'centroid_y', 'count']
    non_rest_area_merged_df[num_cols] = non_rest_area_merged_df[num_cols].fillna(0)
    rest_area_merged_df[num_cols] = rest_area_merged_df[num_cols].fillna(0)

    # 목표 클러스터 개수 조정
    num_rest_areas = len(rest_area_merged_df)
    target_non_rest_area_candidates = max(1, n_clusters - num_rest_areas)
    kmeans_n_clusters = target_non_rest_area_candidates

    # KMeans 클러스터링
    X = non_rest_area_merged_df[num_cols].values
    kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    non_rest_area_merged_df['cluster'] = kmeans.labels_

    # 대표 링크 선택
    representative_links = []
    for cluster_id in non_rest_area_merged_df['cluster'].unique():
        cluster_df = non_rest_area_merged_df[non_rest_area_merged_df['cluster'] == cluster_id].copy()
        if not cluster_df.empty:
            cluster_df.sort_values(by='count', ascending=False, inplace=True)
            max_count = cluster_df['count'].iloc[0]
            same_count_links = cluster_df[cluster_df['count'] == max_count]
            if len(same_count_links) == 1:
                representative_links.append(same_count_links.iloc[0]['LINK_ID'])
            else:
                centroid = kmeans.cluster_centers_[cluster_id]
                same_count_links.loc[:, num_cols] = same_count_links[num_cols].astype(float) # SettingWithCopyWarning 방지
                centroid = centroid.astype(float)
                distances = np.linalg.norm(same_count_links[num_cols].values - centroid, axis=1)
                closest_link_index = distances.argmin()
                representative_links.append(same_count_links.iloc[closest_link_index]['LINK_ID'])

    representative_links_df = pd.DataFrame({'LINK_ID': representative_links})
    representative_links_df = pd.merge(representative_links_df, non_rest_area_merged_df, on='LINK_ID', how='left')

    # 최종 결과 생성 (휴게소 + 대표 링크)
    final_result_df = pd.concat([rest_area_merged_df, representative_links_df], ignore_index=True)
    final_result_df = final_result_df.drop_duplicates(subset=['LINK_ID'], keep='first')

    # 필요한 열만 선택 및 정수형으로 변환
    final_cols = ['LINK_ID', 'OD', 'traffic', 'interval', 'infra', 'rest_area', 'count']
    final_result_df = final_result_df[final_cols].fillna(0).astype({'LINK_ID': 'int', 'OD': 'int', 'traffic': 'int', 'interval': 'int', 'infra': 'int', 'rest_area': 'int', 'count': 'int'})

    print(f"최종 선택된 링크 개수: {len(final_result_df)}")
    # 최종 결과 CSV 파일 저장 경로 수정
    final_result_df[['LINK_ID']].to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\candidate.csv", index=False)
    final_result_df.to_csv(r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\candidate(detail).csv", index=False)
    return final_result_df

def visualize_candidate_selection(candidate_df, title, link_centroids_file_path, filename, output_dir): # output_dir parameter 추가
    """
    후보지 선정 결과를 한국 지도 위에 시각화하는 함수 (EPSG:3857 좌표계 - Web Mercator)
    """
    link_centroids_df = pd.read_csv(link_centroids_file_path)
    merged_df = pd.merge(candidate_df, link_centroids_df, on='LINK_ID', how='inner')

    # matplotlib 설정
    plt.figure(figsize=(10, 12))
    ax = plt.axes(projection=ccrs.epsg(3857)) # Using EPSG:3857 (Web Mercator)

    # 한국 지도 배경 (extent for South Korea -  APPROXIMATE in Web Mercator)
    # Extent is roughly set to cover South Korea, you might need to fine-tune this for EPSG:3857
    ax.set_extent([14000000, 14800000, 4000000, 4600000], crs=ccrs.epsg(3857)) # Approximate extent in EPSG:3857 - ADJUST IF NEEDED
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor='black')

    # 후보지 위치 산점도 (transform from Lat/Lon (PlateCarree) to Web Mercator)
    ax.scatter(merged_df['centroid_x'], merged_df['centroid_y'], transform=ccrs.PlateCarree(), s=10, label='Candidates') # Assuming centroid_x, centroid_y are in Lat/Lon (WGS 84)

    # 제목 및 라벨
    plt.title(title)
    plt.xlabel('X (EPSG:3857 - Web Mercator)') # Label updated to EPSG:3857
    plt.ylabel('Y (EPSG:3857 - Web Mercator)')  # Label updated to EPSG:3857
    plt.legend()

    # 저장
    output_path = os.path.join(output_dir, filename) # output_dir와 filename을 결합하여 저장 경로 생성
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

if __name__ == '__main__':

    # 각 함수 실행
    OD_df = OD_candidate_selection(OD_file_path=OD_file_path, candidate_multiple=candidate_multiple, num_of_candidate=num_of_candidate, link_centroids_file_path=link_centroids_file_path)
    infra_df = infra_candidate_selection(infra_file_path=infra_file_path, candidate_multiple=candidate_multiple, num_of_candidate=num_of_candidate, link_centroids_file_path=link_centroids_file_path)
    traffic_df = traffic_candidate_selection(traffic_file_path=traffic_file_path, candidate_multiple=candidate_multiple, num_of_candidate=num_of_candidate, link_centroids_file_path=link_centroids_file_path)
    interval_df = interval_candidate_selection(interval_file_path=interval_file_path)
    rest_area_df = rest_area_candidate_selection(rest_area_link_ids_path=rest_area_link_ids_path)

    # 모든 후보 선정 결과 병합
    merged_candidate_df = merge_candidate_selection(OD_df, infra_df, traffic_df, interval_df, rest_area_df)
    final_result_df = combine_candidate_dataframes(merged_candidate_df=merged_candidate_df, link_centroids_file_path=link_centroids_file_path, n_clusters=num_of_candidate, random_state=1)

     # 시각화 함수 호출
