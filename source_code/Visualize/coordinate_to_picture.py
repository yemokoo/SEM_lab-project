import pandas as pd
import folium
from tqdm import tqdm

def get_marker_properties(count):
    """
    충전기 대수(count)에 따라 마커의 반지름과 색상을 결정하는 함수.
    - 작은 수량: 초록색, 큰 수량: 주황색에서 빨간색 계열로 변화.
    - 반지름은 기본값 5에 count에 비례하여 증가.
    """
    # 반지름 조절 (필요에 따라 계수를 조정 가능)
    radius = 1 + count * 0.1 
    
    # 색상 결정: count가 낮으면 green, 중간이면 orange, 높으면 red
    if 0<  count <= 10:
        color = 'green'
    elif count <= 20:
        color = 'orange'
    elif count <= 30:
        color = 'blue'    
    else:
        color = 'red'
    return radius, color

def plot_coordinates_on_map(csv_file_path):
    # Step 1: CSV 파일에서 데이터를 불러오기
    df = pd.read_csv(csv_file_path)
    
    # Step 2: 대한민국 중심에 고속도로 등이 두드러지는 'Stamen Toner' 타일을 사용하는 지도 생성
    map_center = [36.5, 127.8]  # 대한민국 중심 좌표 (위도, 경도)
    # OpenStreetMap 사용
    m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB dark_matter')

    
    # Step 3: 좌표 데이터를 지도에 추가
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing rows'):
        latitude = row.get('Latitude')
        longitude = row.get('Longitude')
        count = row.get('count')
        
        # 유효한 좌표인지 확인
        if pd.isna(latitude) or pd.isna(longitude):
            continue
        
        # 숫자형으로 변환 (에러 발생 시 해당 행은 건너뜀)
        try:
            count_val = float(count)
        except:
            continue
        
        # count 값에 따른 마커 속성 결정
        radius, color = get_marker_properties(count_val)
        
        # 마커 생성 (link_id와 count를 팝업으로 표시)
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(f"Link ID: {row['link_id']}<br>Count: {count}", parse_html=True)
        ).add_to(m)
    
    # Step 4: 지도를 HTML 파일로 저장
    m.save('1000개_linkid_Kmeans_Carto_dark_matter.html')
    
    return m

# 예제 사용법
csv_file_path = r"C:\Users\yemoy\SEM_화물차충전소\semlab\kmeans_clusters.csv"
map_object = plot_coordinates_on_map(csv_file_path)

# Jupyter Notebook에서 지도 표시
map_object
