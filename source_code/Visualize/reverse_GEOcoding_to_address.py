import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 예시 데이터프레임 생성 (실제 데이터셋으로 대체 가능)
df = pd.read_csv(r"C:\Users\yemoy\SEM_화물차충전소\semlab\ga_1500iter_5000truck.csv")
output_csv_path = "output_with_address.csv"
# Nominatim을 이용한 역지오코딩 객체 생성 (User Agent는 임의로 지정)
geolocator = Nominatim(user_agent="myGeocoder", timeout=10)

# RateLimiter를 사용해 API 호출 간 최소 1초의 간격을 두도록 설정 (과도한 요청 방지)
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
def reverse_geocode(row):
    """
    위도와 경도 값을 받아 역지오코딩 후 주소를 반환하는 함수.
    language 매개변수를 'ko'로 설정하여 한국어 주소를 요청함.
    """
    try:
        # 좌표를 튜플로 전달하고, 언어 설정을 통해 한국어 주소를 받음
        location = reverse((row['Latitude'], row['Longitude']), language='ko')
        print("변환 횟수 : ", row.name)  
        return location.address if location else None
    except Exception as e:
        print(f"오류 발생 (인덱스 {row.name}): {e}")
        return None

# 각 행에 대해 역지오코딩 수행하여 '주소' 컬럼 추가
df['주소'] = df.apply(reverse_geocode, axis=1)

df.to_csv(output_csv_path, index=False)
print(f"결과가 '{output_csv_path}'로 저장되었습니다.")


