# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import time # 시간 측정

# 1. SparkSession 생성 (원본 코드 설정 유지)
spark = (
    SparkSession.builder.appName("PathAnalysisToParquetRestoredPathUpdate") # App 이름 변경
    .config("spark.executor.memory", "24g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.default.parallelism", "400")
    .config("spark.memory.fraction", "0.8")
    .config("spark.executor.cores", "4")
    .config("spark.python.worker.reuse", "false") # 원본 설정 유지
    .config("spark.executor.heartbeatInterval", "300s")
    .config("spark.network.timeout", "1000s")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    # 필요시 Adaptive Execution 추가 가능
    # .config("spark.sql.adaptive.enabled", "true")
    # .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)
print("SparkSession 생성 완료.")

# 2. CSV 파일 스키마 정의 (원본 Raw 데이터 스키마 - 원본 코드와 동일)
raw_schema = StructType([
    StructField("OBU_ID", IntegerType(), True),
    StructField("GROUP_ID", IntegerType(), True),
    StructField("VEH_TYPE", IntegerType(), True),
    StructField("SEQ", IntegerType(), True),
    StructField("DATETIME", TimestampType(), True), # 원본 DATETIME은 Timestamp
    StructField("LINK_ID", StringType(), True),
    StructField("IN_TIME", StringType(), True),
    StructField("OUT_TIME", StringType(), True),
    StructField("LINK_LENGTH", FloatType(), True), # *** 원본 컬럼명 사용 ***
    StructField("LINK_SPEED", FloatType(), True),
    StructField("POINT_LENGTH", FloatType(), True),
    StructField("POINT_SPEED", FloatType(), True),
    StructField("DAS_FAKE_TYPE", IntegerType(), True),
    StructField("REAL_TYPE", IntegerType(), True),
    StructField("EMD_CODE", IntegerType(), True),
])

# 3. 파일 목록 가져오기, 정렬 및 처리 함수 정의 (원본 로직 유지)
def process_raw_csv_to_parquet_original_logic(raw_file_path, raw_schema, parquet_output_base_path):
    """
    원본 CSV 처리 로직을 유지하면서, 최종 출력을 Parquet으로 저장합니다.
    TOTAL_MINUTES 컬럼을 Parquet에 포함시킵니다.

    Args:
        raw_file_path (str): Raw CSV 파일 경로 (와일드카드 사용 가능).
        raw_schema (StructType): Raw CSV 스키마.
        parquet_output_base_path (str): Parquet 저장 기본 경로.
    """
    start_time_total = time.time()
    print(f"Raw CSV 처리 시작 (원본 로직 유지): {raw_file_path}")
    print(f"Parquet 출력 경로: {parquet_output_base_path}")

    # 3-1. 파일 목록 가져오기 (원본 코드와 동일)
    file_list = glob.glob(raw_file_path)
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
        partition_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        batch_files = [file]

        print(f"\n처리 시작: {file} (날짜: {partition_date})")

        # 3-5. Spark로 파일 읽기, 데이터 처리 (원본 코드 로직 유지)
        try:
            df = (
                spark.read.csv(batch_files, header=True, schema=raw_schema, sep=",")
                .na.fill(0, subset=["OBU_ID"])
                .drop("VEH_TYPE", "SEQ", "LINK_SPEED", "POINT_LENGTH", "POINT_SPEED", "DAS_FAKE_TYPE", "REAL_TYPE")
                .withColumn("LINK_ID", col("LINK_ID").cast(IntegerType()))
                .withColumn("IN_TIME_TS", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("IN_TIME")), "yyyy-MM-dd HHmmss"))
                .withColumn("OUT_TIME_TS", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("OUT_TIME")), "yyyy-MM-dd HHmmss"))
                .na.drop(subset=["IN_TIME_TS", "OUT_TIME_TS"])
                .withColumn("DRIVING_TIME", (unix_timestamp("OUT_TIME_TS") - unix_timestamp("IN_TIME_TS")).cast("double"))
                .filter(col("LINK_ID") != -1)
                .drop("IN_TIME", "OUT_TIME", "IN_TIME_TS", "OUT_TIME_TS")
                .withColumn("DRIVING_TIME_MINUTES", round(col("DRIVING_TIME") / 60, 1))
                .drop("DRIVING_TIME")
                 .withColumn("LINK_LENGTH", round(col("LINK_LENGTH") / 1000, 1)) # 단위 변환은 하되 이름은 LINK_LENGTH 유지
            )
            print(f"  - 파일 읽기 및 기본 처리 완료")

            # 3-6. 데이터 집계 및 변환 (원본 코드 로직 유지)
            df = df.withColumn("TRIP_ID", concat(col("OBU_ID"), lit("_"), col("GROUP_ID")))
            df = df.withColumn("DATE", date_format(col("DATETIME"), "yyyy-MM-dd"))

            min_date_in_file = df.select(min("DATE")).first()[0]
            if not min_date_in_file:
                 print("경고: 파일 내 최소 날짜를 찾을 수 없어 TOTAL_MINUTES 계산에 오류 가능성.")
                 min_date_in_file = partition_date

            df = df.withColumn("DAY_DIFF", datediff(col("DATE"), lit(min_date_in_file)))
            df = df.withColumn("TOTAL_MINUTES",
                               (col("DAY_DIFF") * 1440 + hour(col("DATETIME")) * 60 + minute(col("DATETIME"))).cast(IntegerType()))
            df = df.withColumn("COMBINED_DATETIME", concat(floor(col("TOTAL_MINUTES") / 60), lit(":"), lpad(col("TOTAL_MINUTES") % 60, 2, "0")))

            window_spec_cumulative = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy("DATE", "DATETIME")
            df = df.withColumn("CUMULATIVE_DRIVING_TIME_MINUTES",
                               round(sum("DRIVING_TIME_MINUTES").over(window_spec_cumulative), 1)) \
                   .withColumn("CUMULATIVE_LINK_LENGTH",
                               round(sum("LINK_LENGTH").over(window_spec_cumulative), 1))

            df = df.withColumn("DATETIME_HHMM", date_format(col("DATETIME"), "HH:mm"))
            df = df.drop("DAY_DIFF")

            print(f"  - 집계 및 변환 완료 (TOTAL_MINUTES 계산됨)")

            # 3-7. 행정구역 정보 조인 (원본 코드 로직 유지)
            df = df.join(joined_df_admin_area, df["EMD_CODE"] == joined_df_admin_area["emd_id"], "left")
            print(f"  - 행정구역 정보 조인 완료")


            # 3-8. 데이터 필터링, 열 이름 변경 및 최종 선택 (Parquet 저장용으로 수정)
            # *** 필터링 로직 복구 ***
            window_spec_filter = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(desc("DATETIME"))
            df_filtered = (
                df.withColumn("last_cumulative_link_length", last("CUMULATIVE_LINK_LENGTH").over(window_spec_filter))
                .drop("last_cumulative_link_length")
                .groupBy("OBU_ID", "TRIP_ID")
                .agg(count("*").alias("count"))
                .drop("count")
            )
            df = df.join(df_filtered, ["OBU_ID", "TRIP_ID"], "inner")
            print(f"  - 원본 필터링 로직 적용 완료")

            # *** 수정: 최종 Parquet 저장용 컬럼 선택 ***
            final_df = df.select(
                col("OBU_ID"),
                col("TRIP_ID"),
                col("TOTAL_MINUTES"),
                col("LINK_ID"),
                col("LINK_LENGTH"),
                col("DRIVING_TIME_MINUTES"),
                col("CUMULATIVE_DRIVING_TIME_MINUTES"),
                col("CUMULATIVE_LINK_LENGTH"),
                col("sigungu_id"),
                col("AREA_ID")
            )
            print(f"  - 최종 컬럼 선택 완료")


            # 3-9. Parquet으로 저장 (날짜별 파티셔닝) - 수정된 부분
            final_df_to_save = final_df.withColumn("processing_date", lit(partition_date))
            start_time_write = time.time()
            final_df_to_save.coalesce(10) \
                .write \
                .partitionBy("processing_date") \
                .parquet(parquet_output_base_path, mode="append")
            end_time_write = time.time()

            processed_files += 1
            end_time_file = time.time()
            print(f"  - 처리 완료 및 Parquet 저장: {partition_date}. 쓰기 시간: {end_time_write - start_time_write:.2f} 초.")
            print(f"  - 파일 처리 총 소요 시간: {end_time_file - start_time_file:.2f} 초.")

        except Exception as e:
            print(f"오류 발생: 파일 처리 중 - {file}")
            print(e)
            continue

    end_time_total = time.time()
    print(f"\n총 {processed_files}개 파일 처리 완료.")
    print(f"전체 Raw CSV 처리 및 Parquet 저장 작업 완료. 총 소요 시간: {end_time_total - start_time_total:.2f} 초.")


# 4. 함수 호출 (경로 수정)
# *** 경로 수정: D 드라이브 반영 ***
raw_file_path = r"D:\project\HDT_EVCS_Opt\Data\Raw_Data\Trajectory\*.csv"
# *** 경로 수정: D 드라이브 반영 ***
parquet_output_base_path = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY_PARQUET_RAW)"

process_raw_csv_to_parquet_original_logic(raw_file_path, raw_schema, parquet_output_base_path)

# 5. SparkSession 종료
spark.stop()
print("SparkSession 종료됨.")

