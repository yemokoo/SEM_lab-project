import glob
import re
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os

# SparkSession 생성 (기존 코드 사용)
spark = (
    SparkSession.builder.appName("PathAnalysis")
    .config("spark.executor.memory", "30g")
    .config("spark.driver.memory", "15g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.default.parallelism", "400")
    .config("spark.memory.fraction", "0.8")
    .config("spark.executor.cores", "4")
    .config("spark.python.worker.reuse", "false")
    .config("spark.executor.heartbeatInterval", "300s")
    .config("spark.network.timeout", "1000s")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)

# 파일 경로 설정 (일 단위)
file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\analyzed_paths_for_simulator(DAY)\*" # 이 때 사용하는 경로 파일은 60km 필터링이 수행되지 않는 경로파일
output_dir = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis"  # 출력 디렉토리
output_filename = "vaild_path_ratio.csv"  # 출력 파일 이름


def extract_date(filename):
    """파일명에서 날짜 정보를 추출합니다."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)  # YYYY-MM-DD 형식으로 반환
    return ""  # 파일 형식이 맞지 않는 경우 빈 문자열 반환


def process_csv_files_daily(file_path):
    """
    파일 경로에서 파일 목록을 가져와 날짜별로 정렬하고, 각 파일을 처리하여 결과를 반환합니다.

    Args:
        file_path: 파일 경로 (와일드카드 포함).

    Returns:
        처리 결과 DataFrame.
    """
    file_list = glob.glob(file_path)
    sorted_file_list = sorted(file_list, key=extract_date)

    # CSV 파일 읽기 (Schema 정의)
    schema = StructType([
        StructField("OBU_ID", StringType(), True),
        StructField("TRIP_ID", StringType(), True),
        StructField("DATETIME", StringType(), True),
        StructField("LINK_ID", IntegerType(), True),
        StructField("LINK_LENGTH", DoubleType(), True),
        StructField("DRIVING_TIME_MINUTES", DoubleType(), True),
        StructField("CUMULATIVE_DRIVING_TIME_MINUTES", DoubleType(), True),
        StructField("CUMULATIVE_LINK_LENGTH", DoubleType(), True),
        StructField("sigungu_id", StringType(), True),
        StructField("AREA_ID", IntegerType(), True),
    ])

    results = []
    total_trips_sum = 0  # 총 trip 수 합계
    trips_over_60km_sum = 0  # 60km 초과 trip 수 합계


    for file in sorted_file_list:
        date_str = extract_date(file)
        if not date_str:  # 날짜 형식이 아닌 파일은 건너뛰기
            print(f"Skipping invalid file format: {file}")
            continue

        try:
            df = spark.read.csv(file, header=True, schema=schema)
            # 불필요한 컬럼 제거
            df = df.drop("DATETIME", "LINK_ID", "DRIVING_TIME_MINUTES", "sigungu_id", "AREA_ID")

            # TRIP_ID 별로 그룹화하여 마지막 행 추출 (CUMULATIVE_LINK_LENGTH 기준 정렬)
            window_spec = Window.partitionBy("TRIP_ID").orderBy(desc("CUMULATIVE_LINK_LENGTH"))
            last_rows_df = df.withColumn("row_number", row_number().over(window_spec)).where(col("row_number") == 1).drop("row_number")

            # 전체 행 개수
            total_rows = last_rows_df.count()

            # CUMULATIVE_LINK_LENGTH > 60 필터링 후 행 개수
            filtered_rows = last_rows_df.filter(col("CUMULATIVE_LINK_LENGTH") > 60).count()
            print(f"Date: {date_str} / Total_Trips: {total_rows} / Trips_Over_60km: {filtered_rows}")

            results.append((date_str, total_rows, filtered_rows))

            total_trips_sum += total_rows # 각 파일별 합계 누적
            trips_over_60km_sum += filtered_rows

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    result_df = pd.DataFrame(results, columns=["Date", "Total_Trips", "Trips_Over_60km"])  # 컬럼명 변경
    result_df.to_csv(os.path.join(output_dir, output_filename), index=False, mode='w')

    print(f"Results saved to {os.path.join(output_dir, output_filename)}")

    # 총합 및 비율 출력
    print(f"Total Trips Sum: {total_trips_sum}")
    print(f"Trips Over 60km Sum: {trips_over_60km_sum}")
    if total_trips_sum > 0:  # 0으로 나누는 것 방지
        ratio = (trips_over_60km_sum / total_trips_sum) * 100
        print(f"Ratio (Trips Over 60km / Total Trips): {ratio:.2f}%")
    else:
        print("Ratio (Trips Over 60km / Total Trips): N/A (Total Trips is 0)")

    spark.stop()

process_csv_files_daily(file_path)