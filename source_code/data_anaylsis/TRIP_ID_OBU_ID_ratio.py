import glob
import re
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import logging

# SparkSession 생성 (설정 조정)
spark = (
    SparkSession.builder.appName("PathAnalysis")
    .config("spark.executor.memory", "30g")  # 필요에 따라
    .config("spark.driver.memory", "15g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.default.parallelism", "400")
    .config("spark.memory.fraction", "0.8")
    .config("spark.executor.cores", "4")
    .config("spark.python.worker.reuse", "true")  # 이제 True로 해도 됨
    .config("spark.executor.heartbeatInterval", "300s")
    .config("spark.network.timeout", "1000s")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 파일 경로 설정 (동일)
base_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\analyzed_paths_for_simulator(DAY)"
output_dir = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis"
output_filename = "distinct_counts.csv"


def extract_date(filepath):
    """파일 경로에서 날짜 정보(YYYY-MM-DD)를 추출합니다."""
    dir_path = os.path.dirname(filepath)
    match = re.search(r'(\d{4}-\d{2}-\d{2})', dir_path)
    if match:
        return match.group(1)
    return ""


def process_csv_files_daily_distinct(base_path):
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

    # 결과를 저장할 *빈 리스트* 생성 (날짜별 집계)
    results_list = []

    # 1. 모든 파일 가져오기
    file_list = glob.glob(os.path.join(base_path, "**", "*"), recursive=True)

    # 2. CSV 파일 필터링, 날짜 형식 확인
    filtered_files = [file for file in file_list if file.lower().endswith(".csv") and extract_date(file)]

    # 3. 날짜별로 파일 그룹화
    files_by_date = {}
    for file in filtered_files:
        date_str = extract_date(file)
        if date_str not in files_by_date:
            files_by_date[date_str] = []
        files_by_date[date_str].append(file)

    for date_str, files in sorted(files_by_date.items()):
        try:
            # 같은 날짜의 파일들을 하나의 DataFrame으로 읽기
            df = spark.read.csv(files, header=True, schema=schema)

            # 불필요한 열 조기 제거
            df = df.drop("DATETIME", "LINK_ID", "LINK_LENGTH", "DRIVING_TIME_MINUTES",
                         "CUMULATIVE_DRIVING_TIME_MINUTES", "CUMULATIVE_LINK_LENGTH",
                         "sigungu_id", "AREA_ID")

            # 날짜별 OBU_ID와 TRIP_ID의 고유 개수 계산
            distinct_obu_count = df.select("OBU_ID").distinct().count()
            distinct_trip_count = df.select("TRIP_ID").distinct().count()

            logger.info(f"Date: {date_str} / Distinct OBU_ID Count: {distinct_obu_count} / Distinct TRIP_ID Count: {distinct_trip_count}")

            # 날짜별 결과를 *리스트에 추가*
            results_list.append({"Date": date_str, "Distinct_OBU_ID_Count": distinct_obu_count, "Distinct_TRIP_ID_Count": distinct_trip_count})


        except Exception as e:
            logger.exception(f"Error processing files for date {date_str}: {e}")
            continue


    # Spark DataFrame을 사용하지 않고, 바로 Pandas DataFrame 생성
    results_pd_df = pd.DataFrame(results_list)

    # Null 값 처리 (필요한 경우)
    results_pd_df = results_pd_df.fillna(0)

    # 전체 합계 계산 (Pandas 사용)
    total_distinct_obu_count = results_pd_df["Distinct_OBU_ID_Count"].sum()
    total_distinct_trip_count = results_pd_df["Distinct_TRIP_ID_Count"].sum()

    # 결과 출력
    print(f"Total Distinct OBU_ID Count: {total_distinct_obu_count}")
    print(f"Total Distinct TRIP_ID Count: {total_distinct_trip_count}")

    # 비율 계산
    if total_distinct_obu_count > 0:
        ratio = (total_distinct_trip_count / total_distinct_obu_count) * 100
        print(f"Ratio (Total Distinct TRIP_ID / Total Distinct OBU_ID): {ratio:.2f}%")
    else:
        print("Ratio (Total Distinct TRIP_ID / Total Distinct OBU_ID): N/A (Total Distinct OBU_ID is 0)")

    # 결과 저장
    results_pd_df.to_csv(os.path.join(output_dir, output_filename), index=False, mode='w')
    logger.info(f"Results saved to {os.path.join(output_dir, output_filename)}")

    spark.stop()

process_csv_files_daily_distinct(base_path)