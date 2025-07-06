# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
import os
import time
import datetime
import traceback

# 시각화 라이브러리 임포트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 최적화된 설정을 포함한 SparkSession 생성
spark = (
    SparkSession.builder.appName("PathAnalysis_Monthly_V1")
    .config("spark.executor.memory", "24g")
    .config("spark.driver.memory", "16g")  # Pandas 변환을 위해 드라이버 메모리 증량
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.default.parallelism", "400")
    .config("spark.memory.fraction", "0.8")
    .config("spark.executor.cores", "4")
    .config("spark.python.worker.reuse", "true")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") # toPandas 성능 향상
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)
print("SparkSession 생성 완료.")

# 2. 원본 CSV 파일 스키마 정의
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

# 3. 시각화 함수 정의 (Gantt Chart)
def plot_vehicle_gantt_chart(df_spark, output_dir, month_str, num_vehicles=100):
    """지정된 월의 차량 활동을 샘플링하여 간트 차트로 시각화합니다."""
    print(f"  - 시각화 시작: {month_str} (최대 {num_vehicles}대 차량 샘플링)...")
    try:
        # 시각화할 OBU_ID 샘플링
        obu_id_df = df_spark.select("OBU_ID").distinct().limit(num_vehicles)
        sampled_ids = [row.OBU_ID for row in obu_id_df.collect()]

        if not sampled_ids:
            print("  - 경고: 시각화할 OBU_ID를 찾을 수 없습니다.")
            return

        # 샘플링된 ID에 해당하는 데이터를 Pandas DataFrame으로 변환
        df_sample_pd = df_spark.filter(F.col("OBU_ID").isin(sampled_ids)).toPandas()

        if df_sample_pd.empty:
            print("  - 경고: 샘플링된 데이터가 비어있어 시각화를 건너뜁니다.")
            return

        # 플롯 생성
        fig, ax = plt.subplots(figsize=(20, max(10, len(sampled_ids) * 0.3)))
        y_labels = sorted(df_sample_pd['OBU_ID'].unique())
        y_pos = np.arange(len(y_labels))
        id_to_ypos = {obu_id: pos for pos, obu_id in enumerate(y_labels)}

        driving_label_added = False
        stopping_label_added = False

        for obu_id in y_labels:
            df_vehicle = df_sample_pd[df_sample_pd['OBU_ID'] == obu_id]
            driving_bars = []
            stopping_bars = []

            for _, row in df_vehicle.iterrows():
                # 시작 시간을 시간 단위로 변환
                start_hour = row['START_TIME_MINUTES'] / 60.0
                
                # 주행 시간 바 추가
                if row['DRIVING_TIME_MINUTES'] > 0:
                    duration_hour = row['DRIVING_TIME_MINUTES'] / 60.0
                    driving_bars.append((start_hour, duration_hour))

                # 정지 시간 바 추가 (정지 시작 시점은 주행 종료 시점과 동일)
                if row['STOPPING_TIME'] > 0:
                    # 정지 시작 시점 = 현재 이벤트 시작 시간 + 주행 시간
                    stop_start_minutes = row['START_TIME_MINUTES'] + row['DRIVING_TIME_MINUTES']
                    stop_start_hour = stop_start_minutes / 60.0
                    duration_hour = row['STOPPING_TIME'] / 60.0
                    stopping_bars.append((stop_start_hour, duration_hour))

            y = id_to_ypos[obu_id]

            # broken_barh로 그리기
            if driving_bars:
                ax.broken_barh(driving_bars, (y - 0.4, 0.8), facecolors='tab:blue', label='Driving' if not driving_label_added else "")
                driving_label_added = True
            if stopping_bars:
                ax.broken_barh(stopping_bars, (y - 0.4, 0.8), facecolors='tab:orange', label='Stopping' if not stopping_label_added else "")
                stopping_label_added = True

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Time (Hours Since First Trip of the Month)")
        ax.set_ylabel("Vehicle ID (OBU_ID)")
        ax.set_title(f"Vehicle Activity Timeline for {month_str} (Sample: {len(sampled_ids)} vehicles)", fontsize=16)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        
        output_filename = os.path.join(output_dir, f"timeline_chart_{month_str}.png")
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"  - 타임라인 차트 저장 완료: {output_filename}")

    except Exception as e:
        print(f"오류: 시각화 중 에러 발생 - {e}")
        traceback.print_exc()

# 4. 메인 처리 함수 정의 (월 단위)
def process_monthly_data(file_path_pattern, schema, parquet_output_base_dir, year, month):
    """
    OBU_ID별 월간 활동 기준으로 원본 궤적 CSV 파일을 처리합니다.
    월 누적 주행거리가 임계값 미만인 OBU는 필터링합니다.
    처리된 월별 결과를 Parquet 파일로 저장하고 간트 차트를 생성합니다.
    """
    start_time_total = time.time()
    month_str = f"{year}-{month:02d}"
    print(f"\n{month_str} 데이터 처리 시작...")

    # 4-1. 해당 월의 파일 목록 가져오기
    # 파일명 패턴 예시: Trajectory_20200101.csv
    monthly_file_pattern = os.path.join(os.path.dirname(file_path_pattern), f"*_{year}{month:02d}*.csv")
    file_list = glob.glob(monthly_file_pattern)
    
    if not file_list:
        print(f"오류: {month_str}에 해당하는 CSV 파일을 찾을 수 없습니다. (패턴: {monthly_file_pattern})")
        return

    print(f"총 {len(file_list)}개의 {month_str} CSV 파일 발견.")

    # --- 4-2. 행정구역/지역 조회 데이터 로드 및 캐싱 ---
    # (기존 코드와 동일, 최초 한 번만 실행되도록 개선 가능하지만 여기서는 단순화)
    try:
        admin_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Metropolitan area\LINK_ID_DATASET.csv"
        admin_schema = StructType([StructField("sido_id", IntegerType(), True), StructField("sigungu_id", IntegerType(), True), StructField("emd_id", IntegerType(), True)])
        admin_df_raw = spark.read.csv(admin_path, header=True, schema=admin_schema)
        admin_df = admin_df_raw.withColumn("SIGUNGU_ID_ADMIN", F.substring(F.col("sigungu_id").cast("string"), 1, 4))

        area_csv_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Metropolitan area\AREA_ID_DATASET.csv"
        area_schema = StructType([StructField("AREA_ID", IntegerType(), True), StructField("SIGUNGU_ID", StringType(), True)])
        area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)

        joined_df_admin_area = admin_df.join(area_df, admin_df["SIGUNGU_ID_ADMIN"] == area_df["SIGUNGU_ID"], "left")
        joined_df_admin_area = joined_df_admin_area.select(F.col("emd_id"), F.col("sido_id"), F.col("SIGUNGU_ID_ADMIN").alias("SIGUNGU_ID"), F.col("AREA_ID"))
        joined_df_admin_area.cache()
        print("행정구역 및 지역 ID 데이터 로드, 조인 및 캐싱 완료.")
    except Exception as e:
        print(f"오류: 행정구역 데이터 로드 실패 - {e}")
        traceback.print_exc()
        return

    # --- 4-3. 메인 데이터 처리 블록 ---
    try:
        # 4-3-1: 월간 모든 CSV 읽기, 정제, 기본 계산
        df = (
            spark.read.csv(file_list, header=True, schema=schema, sep=",", timestampFormat="yyyy-MM-dd HH:mm:ss")
            .na.fill(0, subset=["OBU_ID", "GROUP_ID"])
            .drop("VEH_TYPE", "SEQ", "LINK_SPEED", "POINT_LENGTH", "POINT_SPEED", "DAS_FAKE_TYPE", "REAL_TYPE")
            .withColumn("LINK_ID", F.col("LINK_ID").cast(IntegerType()))
            .filter(F.col("DATETIME").isNotNull() & F.col("GROUP_ID").isNotNull())
        )

        df = df.withColumn("IN_TIME_TS", F.when(F.col("IN_TIME").isNotNull() & (F.length(F.trim(F.col("IN_TIME"))) == 6), F.to_timestamp(F.concat(F.date_format(F.col("DATETIME"), "yyyy-MM-dd"), F.lit(" "), F.col("IN_TIME")), "yyyy-MM-dd HHmmss")).otherwise(None)) \
              .withColumn("OUT_TIME_TS", F.when(F.col("OUT_TIME").isNotNull() & (F.length(F.trim(F.col("OUT_TIME"))) == 6), F.to_timestamp(F.concat(F.date_format(F.col("DATETIME"), "yyyy-MM-dd"), F.lit(" "), F.col("OUT_TIME")), "yyyy-MM-dd HHmmss")).otherwise(None)) \
              .na.drop(subset=["IN_TIME_TS", "OUT_TIME_TS"]) \
              .withColumn("DRIVING_TIME_SEC", (F.unix_timestamp("OUT_TIME_TS") - F.unix_timestamp("IN_TIME_TS")).cast("long")) \
              .filter((F.col("DRIVING_TIME_SEC") >= 0) & (F.col("DRIVING_TIME_SEC") < 86400)) \
              .withColumn("DRIVING_TIME_MINUTES", F.round(F.col("DRIVING_TIME_SEC") / 60.0, 1)) \
              .withColumn("LINK_LENGTH_KM", F.round(F.col("LINK_LENGTH") / 1000.0, 1)) \
              .filter(F.col("LINK_ID") != -1) \
              .drop("IN_TIME", "OUT_TIME", "IN_TIME_TS", "OUT_TIME_TS", "DRIVING_TIME_SEC", "LINK_LENGTH")

        # OBU_ID별 시간순 정렬 윈도우 정의
        window_obu_time_ordered = Window.partitionBy("OBU_ID").orderBy("DATETIME")

        # --- 4-3-2. 정지 시간 계산 로직 (GROUP_ID 또는 날짜 변경 기준) ---
        # 날짜 컬럼 추가
        df = df.withColumn("DATE", F.to_date(F.col("DATETIME")))

        # lag 함수로 이전 행 정보 가져오기 (OBU_ID 파티션 내)
        df = df.withColumn("prev_GROUP_ID", F.lag("GROUP_ID", 1).over(window_obu_time_ordered)) \
               .withColumn("prev_DATE", F.lag("DATE", 1).over(window_obu_time_ordered)) \
               .withColumn("prev_DATETIME", F.lag("DATETIME", 1).over(window_obu_time_ordered)) \
               .withColumn("prev_DRIVING_TIME_MINUTES", F.lag("DRIVING_TIME_MINUTES", 1).over(window_obu_time_ordered)) \
               .withColumn("prev_LINK_ID", F.lag("LINK_ID", 1).over(window_obu_time_ordered))
        
        # 이전 행의 링크 통행 종료 시각 계산
        df = df.withColumn("prev_datetime_end_ts",
                           F.when(F.col("prev_DATETIME").isNotNull() & F.col("prev_DRIVING_TIME_MINUTES").isNotNull(),
                                  (F.unix_timestamp("prev_DATETIME") + F.col("prev_DRIVING_TIME_MINUTES") * 60).cast("timestamp"))
                           .otherwise(None))

        # ** 핵심 변경: GROUP_ID가 바뀌거나 날짜가 바뀌면 변경으로 간주 **
        df = df.withColumn("is_change",
                           (F.col("OBU_ID") == F.lag("OBU_ID", 1).over(window_obu_time_ordered)) & \
                           ((F.col("GROUP_ID") != F.col("prev_GROUP_ID")) | (F.col("DATE") != F.col("prev_DATE"))))

        # 변경 발생 시 정지 시간 계산 (현재 행 시작 - 이전 행 종료)
        df = df.withColumn("stopping_time_at_change",
                           F.when(F.col("is_change") & F.col("prev_datetime_end_ts").isNotNull(),
                                  (F.unix_timestamp("DATETIME") - F.unix_timestamp("prev_datetime_end_ts")) / 60.0)
                           .otherwise(0.0))
        df = df.withColumn("stopping_time_at_change",
                           F.when(F.col("stopping_time_at_change") < 0, 0.0)
                           .otherwise(F.round(F.col("stopping_time_at_change"), 1)))

        # --- 4-3-3. 주행/정지 통합 데이터프레임 생성 ---
        # 주행 데이터와 정지 데이터를 하나의 레코드로 통합
        # lead 함수로 다음 행의 정지 시간(현재 행에 귀속될)을 가져옴
        df = df.withColumn("STOPPING_TIME", F.lead("stopping_time_at_change", 1, 0.0).over(window_obu_time_ordered))
        
        # 이전 그룹의 마지막 LINK_ID를 정지 위치로 사용
        # 정지 이벤트는 주행 이벤트에 귀속되므로, 현재 행의 LINK_ID가 정지 위치가 됨.
        # (A 주행 -> 정지 -> B 주행) 에서 정지는 A 주행 레코드에 포함.
        
        df_processed = df.select(
            "OBU_ID", "DATETIME", "LINK_ID", "LINK_LENGTH_KM", "DRIVING_TIME_MINUTES", "STOPPING_TIME", "EMD_CODE"
        )
        
        # 행정구역 데이터 조인
        df_processed = df_processed.join(F.broadcast(joined_df_admin_area), df_processed["EMD_CODE"] == joined_df_admin_area["emd_id"], "left") \
                                   .drop("EMD_CODE", "emd_id")

        # --- 4-3-4. 월 누적값 계산 ---
        df_processed = df_processed.withColumn("CUMULATIVE_DRIVING_TIME_MINUTES", F.round(F.sum("DRIVING_TIME_MINUTES").over(window_obu_time_ordered), 1)) \
                                   .withColumn("CUMULATIVE_STOPPING_TIME_MINUTES", F.round(F.sum("STOPPING_TIME").over(window_obu_time_ordered), 1)) \
                                   .withColumn("CUMULATIVE_LINK_LENGTH_KM", F.round(F.sum("LINK_LENGTH_KM").over(window_obu_time_ordered), 1))

        # --- 4-3-5. 월 누적 주행 거리 필터링 ---
        monthly_distance_threshold = 22.4 * 90.0  # 2016.0 km
        window_obu_max = Window.partitionBy("OBU_ID")
        df_with_max = df_processed.withColumn("max_cum_length_per_obu", F.max("CUMULATIVE_LINK_LENGTH_KM").over(window_obu_max))
        
        df_filtered = df_with_max.filter(F.col("max_cum_length_per_obu") > monthly_distance_threshold) \
                                 .drop("max_cum_length_per_obu")
        print(f"  - 최종 필터링 완료 (OBU_ID 별 월 누적 거리 > {monthly_distance_threshold:.1f} km)")

        # --- 4-3-6. 시각화를 위한 상대 시간 컬럼 추가 ---
        window_obu = Window.partitionBy("OBU_ID")
        df_final = df_filtered.withColumn("first_event_ts", F.min("DATETIME").over(window_obu)) \
                              .withColumn("START_TIME_MINUTES", 
                                          F.round((F.unix_timestamp("DATETIME") - F.unix_timestamp("first_event_ts")) / 60.0, 1)) \
                              .drop("first_event_ts")

        # --- 4-3-7. 최종 컬럼 선택 및 이름 변경 ---
        df_final = df_final.select(
            "OBU_ID",
            F.date_format(F.col("DATETIME"), "yyyy-MM-dd HH:mm:ss").alias("DATETIME"),
            "LINK_ID",
            F.col("LINK_LENGTH_KM").alias("LINK_LENGTH"),
            "DRIVING_TIME_MINUTES",
            "STOPPING_TIME",
            "CUMULATIVE_DRIVING_TIME_MINUTES",
            "CUMULATIVE_STOPPING_TIME_MINUTES",
            F.col("CUMULATIVE_LINK_LENGTH_KM").alias("CUMULATIVE_LINK_LENGTH"),
            "SIGUNGU_ID", "AREA_ID",
            "START_TIME_MINUTES" # 시각화용 컬럼
        ).orderBy("OBU_ID", "DATETIME")

        print("  - 최종 데이터 처리 및 컬럼 정리 완료.")

        # --- 4-3-8. 처리된 데이터를 Parquet 형식으로 저장 ---
        parquet_output_path = os.path.join(parquet_output_base_dir, month_str)
        print(f"  - Parquet 저장 경로: {parquet_output_path}")
        start_time_write = time.time()
        
        df_final.coalesce(20).write.parquet(parquet_output_path, mode="overwrite")
        
        end_time_write = time.time()
        print(f"  - {month_str} Parquet 데이터 저장 완료. 쓰기 시간: {end_time_write - start_time_write:.2f} 초.")

        # --- 4-3-9. 시각화 함수 호출 ---
        # parquet_output_base_dir의 상위 폴더에 시각화 결과 저장
        visualization_output_dir = os.path.dirname(parquet_output_base_dir) 
        plot_vehicle_gantt_chart(df_final, visualization_output_dir, month_str, num_vehicles=100)

    except Exception as e:
        print(f"오류 발생: {month_str} 데이터 처리 중 - {e}")
        traceback.print_exc()
    finally:
        # 메모리 관리
        try: del df, df_processed, df_with_max, df_filtered, df_final
        except NameError: pass
        joined_df_admin_area.unpersist()
        print("행정구역 데이터 캐시 해제됨.")

    end_time_total = time.time()
    total_minutes = (end_time_total - start_time_total) / 60
    print(f"{month_str} 처리 완료. 총 소요 시간: {total_minutes:.2f} 분.")

# 5. 파일 경로 정의 및 처리 함수 실행
raw_file_path_base = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Trajectory\*.csv"
# 월 누적 2016km 기준으로 폴더명 변경
parquet_output_base_dir = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(MONTH_90KM)"

# 출력 디렉토리 생성 (없는 경우)
if not os.path.exists(parquet_output_base_dir):
    try:
        os.makedirs(parquet_output_base_dir)
        print(f"출력 디렉토리 생성: {parquet_output_base_dir}")
    except OSError as e:
        print(f"오류: 출력 디렉토리 생성 실패 - {e}")
        spark.stop()
        exit()

# --- 처리할 연도와 월 지정 ---
target_year = 2020
# 예시: 2020년 1월부터 3월까지 처리
target_months = [1]
#target_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 

for month in target_months:
    process_monthly_data(raw_file_path_base, raw_schema, parquet_output_base_dir, target_year, month)

# 6. SparkSession 종료
spark.stop()
print("\n전체 작업 완료. SparkSession 종료됨.")