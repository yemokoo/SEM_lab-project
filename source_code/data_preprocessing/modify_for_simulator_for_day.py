# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import time
import datetime
import holidays # 한국 공휴일 조회용
import traceback

# 0. 2020년 대한민국 공휴일 정보 로드
try:
    kr_holidays_2020 = holidays.KR(years=2020)
    print("2020년 대한민국 공휴일 정보 로드 완료.")
except Exception as e:
    print(f"오류: 공휴일 정보 로드 실패 - {e}")
    kr_holidays_2020 = {} # 조회 실패 시 빈 딕셔너리 사용

# 1. 최적화된 설정을 포함한 SparkSession 생성
spark = (
    SparkSession.builder.appName("PathAnalysisStoppingTimeByOBU_ParquetOnly_V6_Concise")
    .config("spark.executor.memory", "24g") # Executor 메모리
    .config("spark.driver.memory", "12g")  # Driver 메모리
    .config("spark.sql.shuffle.partitions", "800") # Shuffle 파티션 수 (데이터 규모 따라 조정 필요)
    .config("spark.default.parallelism", "400") # 기본 병렬 처리 수준
    .config("spark.memory.fraction", "0.8") # Spark 실행/저장 공간 비율
    .config("spark.executor.cores", "4") # Executor 당 CPU 코어 수
    .config("spark.python.worker.reuse", "false") # Python 워커 재사용 비활성화 (메모리 누수 방지)
    .config("spark.executor.heartbeatInterval", "300s") # 긴 작업 시간 고려
    .config("spark.network.timeout", "1000s") # 대규모 Shuffle/Task 고려
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") # 이전 버전 호환성용 시간 파싱
    .getOrCreate()
)
print("SparkSession 생성 완료.")

# 2. 원본 CSV 파일 스키마 정의
raw_schema = StructType([
    StructField("OBU_ID", IntegerType(), True), # 차량 고유 ID
    StructField("GROUP_ID", IntegerType(), True), # 트립 세그먼트 ID (정지 감지용)
    StructField("VEH_TYPE", IntegerType(), True), # 사용 안 함
    StructField("SEQ", IntegerType(), True),      # 사용 안 함
    StructField("DATETIME", TimestampType(), True), # 레코드 기록 시각 (정렬 기준)
    StructField("LINK_ID", StringType(), True),   # 도로 링크 ID
    StructField("IN_TIME", StringType(), True),   # 링크 진입 시각 (HHmmss)
    StructField("OUT_TIME", StringType(), True),  # 링크 진출 시각 (HHmmss)
    StructField("LINK_LENGTH", FloatType(), True), # 링크 길이 (미터)
    StructField("LINK_SPEED", FloatType(), True), # 사용 안 함
    StructField("POINT_LENGTH", FloatType(), True),# 사용 안 함
    StructField("POINT_SPEED", FloatType(), True), # 사용 안 함
    StructField("DAS_FAKE_TYPE", IntegerType(), True), # 사용 안 함
    StructField("REAL_TYPE", IntegerType(), True),     # 사용 안 함
    StructField("EMD_CODE", IntegerType(), True), # 행정동 코드 (지역 정보 조인용)
])

# 3. 메인 처리 함수 정의
def process_csv_files(file_path_pattern, schema, parquet_output_base_dir, holidays_kr):
    """
    OBU_ID별 일일 활동 기준으로 원본 궤적 CSV 파일을 처리합니다.
    주말/공휴일을 필터링하고, 주행 시간 및 정지 시간(GROUP_ID 변경 기준)을 계산합니다.
    OBU_ID별 일일 누적값을 계산하고, 총 일일 주행 거리가 60km 이하인 OBU는 필터링합니다.
    처리된 모든 날짜의 결과를 Parquet 파일로 저장합니다.
    """
    start_time_total = time.time()
    print(f"Raw CSV 처리 시작 (공휴일/주말 제외, OBU_ID 기준): {file_path_pattern}")
    print(f"Parquet 출력 기본 경로 (모든 날짜): {parquet_output_base_dir}")

    # 3-1. 파일 목록 가져오기 및 파일명 기준 날짜순 정렬
    file_list = glob.glob(file_path_pattern)
    if not file_list:
        print("오류: 지정된 경로에서 CSV 파일을 찾을 수 없습니다.")
        return

    # 파일명에서 YYYYMMDD 형식의 날짜 정수 추출
    def get_date_from_filename(filename):
        basename = os.path.basename(filename)
        match = re.search(r"(\d{8})\.csv$", basename)
        if match: return int(match.group(1))
        match_prefix = re.search(r"_(\d{8})\.csv$", basename)
        if match_prefix: return int(match_prefix.group(1))
        print(f"경고: 파일 이름에서 날짜(YYYYMMDD)를 추출할 수 없음 - {basename}")
        return 0

    file_list.sort(key=get_date_from_filename)
    print(f"총 {len(file_list)}개의 Raw CSV 파일 발견 및 정렬 완료.")

    # --- 3-2. 행정구역/지역 조회 데이터 로드 및 캐싱 (성능 향상 목적) ---
    try:
        admin_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Metropolitan area\LINK_ID_DATASET.csv"
        admin_schema = StructType([ StructField("sido_id", IntegerType(), True), StructField("sigungu_id", IntegerType(), True), StructField("emd_id", IntegerType(), True)])
        admin_df_raw = spark.read.csv(admin_path, header=True, schema=admin_schema)
        admin_df = admin_df_raw.withColumn("SIGUNGU_ID_ADMIN", substring(col("sigungu_id").cast("string"), 1, 4)) # 조인용 4자리 시군구 코드

        area_csv_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Metropolitan area\AREA_ID_DATASET.csv"
        area_schema = StructType([ StructField("AREA_ID", IntegerType(), True), StructField("SIGUNGU_ID", StringType(), True)])
        area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)

        # 시군구 코드 기준 조인
        joined_df_admin_area = admin_df.join(area_df, admin_df["SIGUNGU_ID_ADMIN"] == area_df["SIGUNGU_ID"], "left")
        joined_df_admin_area = joined_df_admin_area.select(col("emd_id"), col("sido_id"), col("SIGUNGU_ID_ADMIN").alias("SIGUNGU_ID"), col("AREA_ID"))
        joined_df_admin_area.cache() # 메모리 캐싱
        print("행정구역 및 지역 ID 데이터 로드, 조인 및 캐싱 완료.")
        del admin_df_raw, admin_df, area_df
    except Exception as e:
        print(f"오류: 행정구역 또는 지역 ID 데이터 로드/조인 실패 - {e}")
        traceback.print_exc()
        spark.stop()
        return

    # 3-3. 정렬된 파일 목록 순회 처리
    processed_files = 0
    skipped_files_holiday_weekend = 0

    for file in file_list:
        start_time_file = time.time()
        date_int = get_date_from_filename(file)
        if date_int == 0: continue
        date_str = str(date_int)
        if len(date_str) != 8: continue

        # --- 주말 및 2020년 공휴일 필터링 ---
        try:
            file_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            if file_date.year != 2020: continue # 2020년 데이터만 처리
            is_holiday = file_date in holidays_kr if isinstance(holidays_kr, holidays.HolidayBase) else False
            if file_date.weekday() >= 5 or is_holiday: # 주말(토=5, 일=6) 또는 공휴일 제외
                skipped_files_holiday_weekend += 1
                continue
        except ValueError:
            print(f"건너뛰기 (날짜 변환 오류): {os.path.basename(file)}")
            continue

        output_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        print(f"\n처리 시작 (주중): {os.path.basename(file)} (날짜: {output_date})")

        # --- 메인 데이터 처리 블록 ---
        try:
            # 3-4 & 3-5: CSV 읽기, 정제, 시간/거리 계산
            df = (
                spark.read.csv([file], header=True, schema=schema, sep=",", timestampFormat="yyyy-MM-dd HH:mm:ss")
                .na.fill(0, subset=["OBU_ID", "GROUP_ID"])
                .drop("VEH_TYPE", "SEQ", "LINK_SPEED", "POINT_LENGTH", "POINT_SPEED", "DAS_FAKE_TYPE", "REAL_TYPE") # 불필요 컬럼 제거
                .withColumn("LINK_ID", col("LINK_ID").cast(IntegerType()))
                .filter(col("DATETIME").isNotNull() & col("GROUP_ID").isNotNull())
            )

            # IN/OUT TIME -> Timestamp 변환 및 주행 시간(분), 링크 길이(km) 계산
            df = df.withColumn("IN_TIME_TS", when(col("IN_TIME").isNotNull() & (length(trim(col("IN_TIME"))) == 6), to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("IN_TIME")), "yyyy-MM-dd HHmmss")).otherwise(None)) \
                  .withColumn("OUT_TIME_TS", when(col("OUT_TIME").isNotNull() & (length(trim(col("OUT_TIME"))) == 6), to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("OUT_TIME")), "yyyy-MM-dd HHmmss")).otherwise(None)) \
                  .na.drop(subset=["IN_TIME_TS", "OUT_TIME_TS"]) \
                  .withColumn("DRIVING_TIME_SEC", (unix_timestamp("OUT_TIME_TS") - unix_timestamp("IN_TIME_TS")).cast("long")) \
                  .filter((col("DRIVING_TIME_SEC") >= 0) & (col("DRIVING_TIME_SEC") < 86400)) \
                  .withColumn("DRIVING_TIME_MINUTES", round(col("DRIVING_TIME_SEC") / 60.0, 1)) \
                  .withColumn("LINK_LENGTH_KM", round(col("LINK_LENGTH") / 1000.0, 1)) \
                  .filter(col("LINK_ID") != -1) \
                  .drop("IN_TIME", "OUT_TIME", "IN_TIME_TS", "OUT_TIME_TS", "DRIVING_TIME_SEC", "LINK_LENGTH") # 중간 컬럼 제거

            # OBU_ID별 시간순 정렬 윈도우 정의
            window_obu_time_ordered = Window.partitionBy("OBU_ID").orderBy("DATETIME")

            # 3-6. OBU_ID별 일일 누적 주행 시간/거리 계산
            df = df.withColumn("CUMULATIVE_DRIVING_TIME_MINUTES", round(sum("DRIVING_TIME_MINUTES").over(window_obu_time_ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)), 1)) \
                  .withColumn("CUMULATIVE_LINK_LENGTH_KM", round(sum("LINK_LENGTH_KM").over(window_obu_time_ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)), 1))

            # --- 3-7. 행정구역/지역 데이터 조인 ---
            df = df.join(broadcast(joined_df_admin_area), df["EMD_CODE"] == joined_df_admin_area["emd_id"], "left") \
                  .drop("EMD_CODE", "emd_id")

            # --- 3-8. GROUP_ID 변경 기준 정지 시간 계산 ---
            # lag 함수로 이전 행 정보 가져오기 (OBU_ID 파티션 내)
            df = df.withColumn("prev_GROUP_ID", lag("GROUP_ID", 1).over(window_obu_time_ordered)) \
                  .withColumn("prev_DATETIME", lag("DATETIME", 1).over(window_obu_time_ordered)) \
                  .withColumn("prev_DRIVING_TIME_MINUTES", lag("DRIVING_TIME_MINUTES", 1).over(window_obu_time_ordered))

            # 이전 행의 링크 통행 종료 시각 계산
            df = df.withColumn("prev_datetime_end_ts",
                               when(col("prev_DATETIME").isNotNull() & col("prev_DRIVING_TIME_MINUTES").isNotNull(),
                                    (unix_timestamp("prev_DATETIME") + col("prev_DRIVING_TIME_MINUTES") * 60).cast("timestamp"))
                               .otherwise(None))

            # GROUP_ID 변경 여부 확인
            df = df.withColumn("is_group_change",
                              (col("OBU_ID") == lag("OBU_ID", 1).over(window_obu_time_ordered)) & \
                              (col("GROUP_ID") != col("prev_GROUP_ID")))

            # GROUP_ID 변경 시 정지 시간 계산 (현재 행 시작 - 이전 행 종료)
            df = df.withColumn("stopping_time_at_change",
                               when(col("is_group_change") & col("prev_datetime_end_ts").isNotNull(),
                                    (unix_timestamp("DATETIME") - unix_timestamp("prev_datetime_end_ts")) / 60.0)
                               .otherwise(0.0))
            df = df.withColumn("stopping_time_at_change",
                               when(col("stopping_time_at_change") < 0, 0.0)
                               .otherwise(round(col("stopping_time_at_change"), 1))) # 음수 방지 및 반올림

            # --- 3-9. 정지 행(Stopping Rows) 생성 ---
            # lead 함수로 다음 행 정보 가져오기: 다음 행의 정지 시간(lag(-1))과 다음 행의 GROUP_ID 변경 여부(lead(1))
            df = df.withColumn("stopping_time_for_prev_group", lag("stopping_time_at_change", -1, 0.0).over(window_obu_time_ordered))
            df = df.withColumn("is_stop_point", lead("is_group_change", 1, False).over(window_obu_time_ordered)) # 다음 행이 group change면 현재 행이 stop point

            # 정지 행 생성 기준 데이터 필터링 (is_stop_point == True)
            stopping_rows_base = df.filter(col("is_stop_point")).alias("base")
            # 정지 행의 시작 시각 계산 (현재 주행 종료 시점)
            stopping_rows_base = stopping_rows_base.withColumn("stop_row_datetime",
                               (unix_timestamp(col("DATETIME")) + col("DRIVING_TIME_MINUTES") * 60).cast("timestamp"))

            # 정지 행 데이터 구성
            stopping_rows_new = stopping_rows_base.select(
                col("OBU_ID"), col("GROUP_ID"), # 임시 유지
                col("stop_row_datetime").alias("DATETIME"), col("LINK_ID"),
                lit(0.0).alias("LINK_LENGTH_KM"), lit(0.0).alias("DRIVING_TIME_MINUTES"), # 주행 관련 값 0
                col("CUMULATIVE_DRIVING_TIME_MINUTES"), col("CUMULATIVE_LINK_LENGTH_KM"), # 누적값 상속
                col("SIGUNGU_ID"), col("AREA_ID"), col("sido_id"),
                col("stopping_time_for_prev_group").alias("STOPPING_TIME") # 계산된 정지 시간
            )

            # 원본 주행 행 데이터 준비
            df_original_rows = df.select(
                col("OBU_ID"), col("GROUP_ID"), # 임시 유지
                col("DATETIME"), col("LINK_ID"), col("LINK_LENGTH_KM"), col("DRIVING_TIME_MINUTES"),
                col("CUMULATIVE_DRIVING_TIME_MINUTES"), col("CUMULATIVE_LINK_LENGTH_KM"),
                col("SIGUNGU_ID"), col("AREA_ID"), col("sido_id"),
                lit(0.0).alias("STOPPING_TIME") # 주행 행의 정지 시간은 0
            )

            # 주행 행과 정지 행 결합
            df_combined = df_original_rows.unionByName(stopping_rows_new)

            # 임시 컬럼 및 데이터프레임 정리
            temp_cols = ["prev_GROUP_ID", "prev_DATETIME", "prev_DRIVING_TIME_MINUTES",
                         "prev_datetime_end_ts", "is_group_change", "stopping_time_at_change",
                         "is_stop_point", "stopping_time_for_prev_group"]
            del stopping_rows_base, stopping_rows_new, df_original_rows, df

            # --- 3-10. OBU_ID별 일일 누적 정지 시간 계산 ---
            df_combined = df_combined.withColumn("CUMULATIVE_STOPPING_TIME_MINUTES",
                                                 round(sum("STOPPING_TIME").over(window_obu_time_ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)), 1))

            # --- 3-11. 하루 총 주행 거리 <= 90km 인 OBU 필터링 ---
            window_obu_max = Window.partitionBy("OBU_ID")
            df_with_max = df_combined.withColumn("max_cum_length_per_obu", max("CUMULATIVE_LINK_LENGTH_KM").over(window_obu_max))
            df_filtered = df_with_max.filter(col("max_cum_length_per_obu") > 90) \
                                     .drop("max_cum_length_per_obu")
            print(f"  - 최종 필터링 완료 (OBU_ID 별 하루 총 누적 거리 <= 90km 인 OBU 전체 제거)")
            del df_with_max, df_combined

            # --- 3-12. 최종 컬럼 선택, 이름 변경, 정렬, 포맷팅 ---
            df_final = df_filtered.withColumnRenamed("LINK_LENGTH_KM", "LINK_LENGTH") \
                                  .withColumnRenamed("CUMULATIVE_LINK_LENGTH_KM", "CUMULATIVE_LINK_LENGTH")
            df_final = df_final.orderBy("OBU_ID", "DATETIME") # 최종 정렬
            # 최종 컬럼 선택 (GROUP_ID 제외)
            df_final = df_final.select(
                    "OBU_ID", "DATETIME", "LINK_ID", "LINK_LENGTH", "DRIVING_TIME_MINUTES",
                    "STOPPING_TIME", "CUMULATIVE_DRIVING_TIME_MINUTES",
                    "CUMULATIVE_STOPPING_TIME_MINUTES", "CUMULATIVE_LINK_LENGTH",
                    "SIGUNGU_ID", "AREA_ID"
                   )
            # DATETIME -> HH:mm 형식 문자열 변환
            df_final = df_final.withColumn("DATETIME_HM", date_format(col("DATETIME"), "HH:mm")) \
                               .drop("DATETIME") \
                               .withColumnRenamed("DATETIME_HM", "DATETIME")
            print(f"  - 최종 컬럼 선택, 이름 변경, 정렬 및 포맷 완료")

            # --- 3-13. 처리된 데이터를 Parquet 형식으로 저장 ---
            parquet_output_path = os.path.join(parquet_output_base_dir, output_date)
            print(f"  - Parquet 저장 경로: {parquet_output_path}")
            start_time_write_parquet = time.time()
            # coalesce: 출력 파일 수 조절, mode="overwrite": 덮어쓰기
            df_final.coalesce(10).write.parquet(parquet_output_path, mode="overwrite")
            end_time_write_parquet = time.time()
            print(f"  - Saved Parquet data for {output_date}. 쓰기 시간: {end_time_write_parquet - start_time_write_parquet:.2f} 초.")

            processed_files += 1
            end_time_file = time.time()

        except Exception as e:
            print(f"오류 발생: 파일 처리 중 - {os.path.basename(file)}")
            traceback.print_exc()
            continue # 오류 발생 시 다음 파일 처리 계속
        finally:
             # 메모리 관리: 사용 완료된 데이터프레임 삭제
             try: del df_final
             except NameError: pass

    # --- 모든 파일 처리 루프 종료 후 ---
    joined_df_admin_area.unpersist() # 캐시된 데이터 메모리 해제
    print("행정구역 데이터 캐시 해제됨.")

    # 최종 결과 요약 출력
    end_time_total = time.time()
    if processed_files > 0: print(f"\n총 {processed_files}개 파일 처리 완료.")
    else: print("\n처리할 유효한 평일 파일(2020년)을 찾지 못했습니다.")
    print(f"총 {skipped_files_holiday_weekend}개 파일 건너뜀 (공휴일/주말).")
    total_minutes = (end_time_total - start_time_total) / 60
    print(f"전체 작업 완료. 총 소요 시간: {total_minutes:.2f} 분.")

# 4. 파일 경로 정의 및 처리 함수 실행
raw_file_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Trajectory\*.csv"  # 원본 CSV 파일 경로 패턴
parquet_output_base_dir_all_days = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_90km)"

# 출력 디렉토리 생성 (없는 경우)
if not os.path.exists(parquet_output_base_dir_all_days):
    try:
        os.makedirs(parquet_output_base_dir_all_days)
        print(f"출력 디렉토리 생성: {parquet_output_base_dir_all_days}")
    except OSError as e:
        print(f"오류: 출력 디렉토리 생성 실패 - {parquet_output_base_dir_all_days}. 에러: {e}")
        spark.stop()
        exit()

# 메인 처리 함수 호출
process_csv_files(raw_file_path, raw_schema, parquet_output_base_dir_all_days, kr_holidays_2020)

# 5. SparkSession 종료
spark.stop()
print("SparkSession 종료됨.")
