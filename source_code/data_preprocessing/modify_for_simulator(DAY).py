# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os # os 모듈 임포트 확인
import time # 시간 측정

# 1. SparkSession 생성 (원본 코드 설정 유지)
spark = (
    SparkSession.builder.appName("PathAnalysisCSVPathUpdate") # App 이름 변경
    .config("spark.executor.memory", "24g")
    .config("spark.driver.memory", "12g")
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
print("SparkSession 생성 완료.")

# 2. CSV 파일 스키마 정의 (원본 Raw 데이터 스키마 - 원본 코드와 동일)
raw_schema = StructType([
    StructField("OBU_ID", IntegerType(), True),
    StructField("GROUP_ID", IntegerType(), True),
    StructField("VEH_TYPE", IntegerType(), True),
    StructField("SEQ", IntegerType(), True),
    StructField("DATETIME", TimestampType(), True),
    StructField("LINK_ID", StringType(), True),
    StructField("IN_TIME", StringType(), True),
    StructField("OUT_TIME", StringType(), True),
    StructField("LINK_LENGTH", FloatType(), True),
    StructField("LINK_SPEED", FloatType(), True),
    StructField("POINT_LENGTH", FloatType(), True),
    StructField("POINT_SPEED", FloatType(), True),
    StructField("DAS_FAKE_TYPE", IntegerType(), True),
    StructField("REAL_TYPE", IntegerType(), True),
    StructField("EMD_CODE", IntegerType(), True),
])

# 3. 파일 목록 가져오기, 정렬 및 처리 함수 정의 (원본 로직 유지)
def process_csv_files(file_path, schema, output_base_dir): # 출력 기본 디렉토리 인자 추가
    """
    CSV 파일 목록을 날짜순으로 정렬하고, 하루 단위로 Spark로 처리합니다.
    각 날짜별 데이터를 CSV로 순차적으로 저장합니다. (경로 수정됨)

    Args:
        file_path (str): CSV 파일이 있는 디렉토리 경로 (와일드카드 사용 가능).
        schema (StructType): CSV 파일의 스키마.
        output_base_dir (str): CSV 결과를 저장할 기본 디렉토리 경로.
    """
    start_time_total = time.time()
    print(f"Raw CSV 처리 시작 (원본 로직 유지): {file_path}")
    print(f"CSV 출력 기본 경로: {output_base_dir}")

    # 3-1. 파일 목록 가져오기 (원본 코드와 동일)
    file_list = glob.glob(file_path)
    if not file_list:
        print("오류: 지정된 경로에서 CSV 파일을 찾을 수 없습니다.")
        return

    # 3-2. 파일 이름에서 날짜 추출 및 정렬 (원본 코드와 동일)
    def get_date_from_filename(filename):
        match = re.search(r"AUTO_P1_TRUCK_SERO_(\d{8})", filename)
        if match:
            return int(match.group(1))
        else:
            match_date_only = re.search(r"(\d{8})\.csv", os.path.basename(filename))
            if match_date_only:
                return int(match_date_only.group(1))
            print(f"경고: 파일 이름에서 날짜를 추출할 수 없음 - {filename}")
            return 0

    # 3-3. 날짜 기준으로 파일 목록 정렬 (원본 코드와 동일)
    file_list.sort(key=get_date_from_filename)
    print(f"총 {len(file_list)}개의 Raw CSV 파일 발견 및 정렬 완료.")

    # --- 행정구역 및 지역 ID 데이터 로드 (경로 수정) ---
    try:
        # *** 경로 수정: D 드라이브 반영 ***
        admin_path = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\Metropolitan area\LINK_ID_DATASET.csv"
        admin_schema = StructType([
            StructField("sido_id", IntegerType(), True),
            StructField("sigungu_id", IntegerType(), True),
            StructField("emd_id", IntegerType(), True)
        ])
        admin_df_raw = spark.read.csv(admin_path, header=True, schema=admin_schema)
        admin_df = admin_df_raw.withColumn("SIGUNGU_ID", substring(col("sigungu_id").cast("string"), 1, 4))

        # *** 경로 수정: D 드라이브 반영 ***
        area_csv_path = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\Metropolitan area\AREA_ID_DATASET.csv"
        area_schema = StructType([
            StructField("AREA_ID", IntegerType(), True),
            StructField("SIGUNGU_ID", StringType(), True)
        ])
        area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)
        joined_df_admin_area = admin_df.join(area_df, admin_df["SIGUNGU_ID"] == area_df["SIGUNGU_ID"], "left")
        joined_df_admin_area = joined_df_admin_area.drop(area_df["SIGUNGU_ID"])
        print("행정구역 및 지역 ID 데이터 로드 및 조인 완료.")
    except Exception as e:
        print(f"오류: 행정구역 또는 지역 ID 데이터 로드/조인 실패 - {e}")
        return


    # 3-4. 파일 하나씩 처리 (일 단위 처리)
    processed_files = 0
    for file in file_list:
        start_time_file = time.time()
        date_int = get_date_from_filename(file)
        if date_int == 0:
            print(f"건너뛰기 (날짜 추출 실패): {file}")
            continue

        date_str = str(date_int)
        output_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}" # YYYY-MM-DD 형식 날짜 문자열 생성
        batch_files = [file]

        print(f"\n처리 시작: {file} (날짜: {output_date})")

        # 3-5. Spark로 파일 읽기, 데이터 처리 (원본 코드 로직 유지)
        try:
            # 원본 코드의 필터링 조건 추가: DRIVING_TIME >= 1
            df = (
                spark.read.csv(batch_files, header=True, schema=raw_schema, sep=",")
                .na.fill(0, subset=["OBU_ID"])
                .drop("VEH_TYPE", "SEQ", "LINK_SPEED", "POINT_LENGTH", "POINT_SPEED", "DAS_FAKE_TYPE", "REAL_TYPE")
                .withColumn("LINK_ID", col("LINK_ID").cast(IntegerType()))
                .withColumn("IN_TIME_TS", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("IN_TIME")), "yyyy-MM-dd HHmmss"))
                .withColumn("OUT_TIME_TS", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("OUT_TIME")), "yyyy-MM-dd HHmmss"))
                .na.drop(subset=["IN_TIME_TS", "OUT_TIME_TS"])
                .withColumn("DRIVING_TIME", (unix_timestamp("OUT_TIME_TS") - unix_timestamp("IN_TIME_TS")).cast("double"))
                .filter((col("LINK_ID") != -1) & (col("DRIVING_TIME") >= 1)) # *** 원본 필터 조건 적용 ***
                .drop("IN_TIME", "OUT_TIME", "IN_TIME_TS", "OUT_TIME_TS")
                .withColumn("DRIVING_TIME_MINUTES", round(col("DRIVING_TIME") / 60, 1))
                .drop("DRIVING_TIME")
                 .withColumn("LINK_LENGTH", round(col("LINK_LENGTH") / 1000, 1))
            )
            print(f"  - 파일 읽기 및 기본 처리 완료")

            # 3-6. 데이터 집계 및 변환 (원본 코드 로직 유지)
            df = df.withColumn("TRIP_ID", concat(col("OBU_ID"), lit("_"), col("GROUP_ID")))
            df = df.withColumn("DATE", date_format(col("DATETIME"), "yyyy-MM-dd"))

            min_date_in_file = df.select(min("DATE")).first()[0]
            if not min_date_in_file:
                 print("경고: 파일 내 최소 날짜를 찾을 수 없어 TOTAL_MINUTES 계산에 오류 가능성.")
                 min_date_in_file = output_date # 날짜 문자열 사용

            df = df.withColumn("DAY_DIFF", datediff(col("DATE"), lit(min_date_in_file)))
            df = df.withColumn("TOTAL_MINUTES",
                               (col("DAY_DIFF") * 1440 + hour(col("DATETIME")) * 60 + minute(col("DATETIME"))).cast(IntegerType()))
            df = df.withColumn("COMBINED_DATETIME", concat(floor(col("TOTAL_MINUTES") / 60), lit(":"), lpad(col("TOTAL_MINUTES") % 60, 2, "0")))

            window_spec_cumulative = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy("DATE", "DATETIME")
            df = df.withColumn("CUMULATIVE_DRIVING_TIME_MINUTES",
                               round(sum("DRIVING_TIME_MINUTES").over(window_spec_cumulative), 1)) \
                   .withColumn("CUMULATIVE_LINK_LENGTH",
                               round(sum("LINK_LENGTH").over(window_spec_cumulative), 1))

            # 원본 코드의 DATETIME 컬럼 변경 로직 (HH:mm)
            # 주의: 이 변경으로 인해 원본 Timestamp 정보가 손실됨
            df = df.withColumn("DATETIME", date_format(col("DATETIME"), "HH:mm"))
            df = df.drop("DAY_DIFF", "TOTAL_MINUTES") # 원본 코드처럼 TOTAL_MINUTES 제거

            print(f"  - 집계 및 변환 완료")

            # 3-7. 행정구역 정보 조인 (원본 코드 로직 유지)
            df = df.join(joined_df_admin_area, df["EMD_CODE"] == joined_df_admin_area["emd_id"], "left")
            print(f"  - 행정구역 정보 조인 완료")


            # 3-8. 데이터 필터링, 열 이름 변경 및 선택 (원본 코드 로직 유지)
            # 원본 코드의 필터링 로직
            window_spec_filter = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(desc("DATETIME")) # HH:mm 문자열로 정렬됨 (의도 확인 필요)
            # *** 원본 필터 조건 적용 ***
            df = (
                df.withColumn("last_cumulative_link_length", last("CUMULATIVE_LINK_LENGTH").over(window_spec_filter))
                .filter(col("last_cumulative_link_length") > 60) # 각 그룹의 마지막 CUMULATIVE_LINK_LENGTH 값이 60km 이상
                .drop("last_cumulative_link_length") # 임시 열 제거
                .groupBy("OBU_ID", "TRIP_ID")
                .agg(count("*").alias("count"))
                # .filter(col("count") >= 10) # 원본 코드에서 주석 처리된 부분 유지
                .drop("count") # 임시 열 제거
                .join(df.drop("DAY_DIFF", "TOTAL_MINUTES"), ["OBU_ID", "TRIP_ID"], "inner") # 조인 시 df에서 DAY_DIFF, TOTAL_MINUTES 제거된 상태여야 함
                .drop("DATE", "EMD_CODE", "emd_id", "sido_id") # 원본 코드의 drop 컬럼들
                .withColumnRenamed("COMBINED_DATETIME", "DATETIME") # 원본 코드의 rename
                # 원본 코드의 최종 select 컬럼들
                .select("OBU_ID", "TRIP_ID", "DATETIME", "LINK_ID", "LINK_LENGTH", "DRIVING_TIME_MINUTES", "CUMULATIVE_DRIVING_TIME_MINUTES", "CUMULATIVE_LINK_LENGTH", "sigungu_id", "AREA_ID") # admin_df["sigungu_id"] -> "sigungu_id" 로 변경 가정
            )
            print(f"  - 원본 필터링 로직 및 최종 선택 완료")

            # 최종 정렬 (원본 코드)
            df = df.orderBy("OBU_ID", "TRIP_ID", "DATETIME") # HH:mm 문자열로 정렬됨

            # 3-9. 파일 내용 처리 및 저장 (CSV 저장, 경로 수정)
            # *** 출력 경로 수정: 날짜별 폴더 생성 ***
            # output_path = f"D:\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY){output_date}" # 원본 방식 (파일)
            output_path = os.path.join(output_base_dir, output_date) # 수정 방식 (폴더)
            print(f"  - 저장 경로: {output_path}")

            start_time_write = time.time()
            df.coalesce(10).write.csv(output_path, header=True, mode="overwrite")
            end_time_write = time.time()
            print(f"  - Saved {output_date} data to {output_path}. 쓰기 시간: {end_time_write - start_time_write:.2f} 초.")

            processed_files += 1
            end_time_file = time.time()
            print(f"  - 파일 처리 총 소요 시간: {end_time_file - start_time_file:.2f} 초.")

        except Exception as e:
            print(f"오류 발생: 파일 처리 중 - {file}")
            print(e)
            continue

    end_time_total = time.time()
    print(f"\n총 {processed_files}개 파일 처리 완료.")
    print(f"전체 Raw CSV 처리 및 CSV 저장 작업 완료. 총 소요 시간: {end_time_total - start_time_total:.2f} 초.")


# 4. 함수 호출 (경로 수정)
# *** 경로 수정: D 드라이브 반영 ***
raw_file_path = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\Trajectory\*.csv"
# *** 경로 수정: D 드라이브 반영 및 출력 기본 경로 변수화 ***
csv_output_base_dir = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY_RAW)" # 날짜 폴더가 생성될 상위 경로

# 함수 실행 시 출력 경로 전달
process_csv_files(raw_file_path, raw_schema, csv_output_base_dir)

# 5. SparkSession 종료
spark.stop()
print("SparkSession 종료됨.")
