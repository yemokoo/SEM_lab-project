# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark import StorageLevel
from pyspark.sql.functions import * # col, lit, year, month, etc. 포함
from pyspark.sql.types import *
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import traceback # 오류 추적용
import shutil # CSV 저장 시 디렉토리 정리용

# --- 1. SparkSession 생성 (대용량 설정) ---
start_time_spark = time.time()
spark = (
    SparkSession.builder.appName("StopAnalysisIncreasedSampling") # App 이름 변경
    # 대용량 처리 위한 설정 유지 (필요시 조정)
    .config("spark.executor.memory", "32g")
    .config("spark.driver.memory", "16g") # 샘플링 증가로 인해 더 많은 메모리 필요할 수 있음
    .config("spark.sql.shuffle.partitions", "400")
    .config("spark.default.parallelism", "200")
    .config("spark.memory.fraction", "0.7")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.executor.cores", "4")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") # Arrow 비활성화 유지
    .config("spark.driver.maxResultSize", "0") # 충분한 결과 크기 허용 (주의)
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
)
end_time_spark = time.time()
print(f"SparkSession 생성 완료 (대용량 설정, Arrow 비활성화) - 소요 시간: {end_time_spark - start_time_spark:.2f} 초")

# --- 2. 경로 정의 (Parquet 입력 경로 및 모든 LINK_ID 기반 분석 경로) ---
parquet_input_path = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY_RAW)"
output_base_path = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\Data_Analysis\Stopping_time_and_location"
os.makedirs(output_base_path, exist_ok=True)
# 분석 결과 저장 경로 정의
stop_time_hist_path = os.path.join(output_base_path, "stop_time_histogram_stacked.png")
stop_type_hist_path = os.path.join(output_base_path, "stop_type_histogram.png")
location_link_freq_csv_path = os.path.join(output_base_path, "stop_link_frequency_all.csv")
location_link_type_pivot_csv_path = os.path.join(output_base_path, "stop_link_type_pivot_all.csv")
# *** 파일명 원복 ***
kmeans_plot_path = os.path.join(output_base_path, "kmeans_clusters_boxplot.png")
kmeans_sampled_scatter_path = os.path.join(output_base_path, "kmeans_sampled_scatter.png")
stop_scatter_plot_path = os.path.join(output_base_path, "stop_duration_type_scatter.png")
# *** 요약 정보 텍스트 파일 경로 추가 ***
summary_stats_txt_path = os.path.join(output_base_path, "stop_analysis_summary_stats.txt")

print(f"입력 Parquet 경로: {parquet_input_path}")
print(f"출력 경로: {output_base_path}")
print(f"요약 정보 텍스트 파일 경로: {summary_stats_txt_path}")


# --- 3. Parquet 데이터 읽기 및 최소 컬럼 선택 ---
start_time_read = time.time()
try:
    print(f"Parquet 데이터 로딩 시작: {parquet_input_path}")
    df_read_parquet = spark.read.parquet(parquet_input_path)

    print("로드된 Parquet 스키마:")
    df_read_parquet.printSchema()

    columns_needed_minimal = ["OBU_ID", "TRIP_ID", "TOTAL_MINUTES", "LINK_ID"]
    missing_cols = [col_name for col_name in columns_needed_minimal if col_name not in df_read_parquet.columns]
    if missing_cols:
        raise ValueError(f"오류: Parquet 파일에 필요한 컬럼이 없습니다 - {missing_cols}. 첫 번째 스크립트에서 해당 컬럼을 저장했는지 확인하세요.")

    all_processed_df_minimal = df_read_parquet.select(*columns_needed_minimal)

    print(f"Parquet 데이터 로딩 및 최소 컬럼 선택 완료. 스키마 확인:")
    all_processed_df_minimal.printSchema()
    all_processed_df_minimal.cache()
    num_records = all_processed_df_minimal.count()
    end_time_read = time.time()
    print(f"데이터 로딩 및 캐싱 완료. 총 레코드 수: {num_records}. 소요 시간: {end_time_read - start_time_read:.2f} 초")

except Exception as e:
    print(f"Parquet 데이터 로딩 또는 전처리 중 오류 발생: {parquet_input_path}")
    print(e)
    spark.stop()
    exit()


# --- 4. 각 Trip의 시작과 끝 식별 ---
start_time_endpoints = time.time()
print("Trip 시작/종료 지점 식별 시작...")
window_trip_time_asc = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(col("TOTAL_MINUTES").asc())
window_trip_time_desc = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(col("TOTAL_MINUTES").desc())

trip_endpoints_df = all_processed_df_minimal.withColumn("rn_asc", row_number().over(window_trip_time_asc)) \
                                           .withColumn("rn_desc", row_number().over(window_trip_time_desc)) \
                                           .filter((col("rn_asc") == 1) | (col("rn_desc") == 1)) \
                                           .withColumn("trip_marker", when(col("rn_asc") == 1, "start").otherwise("end")) \
                                           .select("OBU_ID", "TRIP_ID", "TOTAL_MINUTES", "LINK_ID", "trip_marker")

if all_processed_df_minimal.is_cached:
    all_processed_df_minimal.unpersist()
    print("원본 all_processed_df_minimal 캐시 해제됨.")
end_time_endpoints = time.time()
print(f"Trip 시작/종료 지점 식별 완료. 소요 시간: {end_time_endpoints - start_time_endpoints:.2f} 초.")


# --- 5. OBU별로 Trip 간 정지 시간 및 위치(LINK_ID) 계산 ---
start_time_stops = time.time()
print("Trip 간 정지 정보 계산 시작...")
window_obu_time = Window.partitionBy("OBU_ID").orderBy(col("TOTAL_MINUTES").asc())

stops_intermediate_df = trip_endpoints_df.withColumn("prev_TOTAL_MINUTES", lag("TOTAL_MINUTES", 1).over(window_obu_time)) \
                                         .withColumn("prev_TRIP_ID", lag("TRIP_ID", 1).over(window_obu_time)) \
                                         .withColumn("prev_LINK_ID", lag("LINK_ID", 1).over(window_obu_time)) \
                                         .withColumn("prev_trip_marker", lag("trip_marker", 1).over(window_obu_time)) \
                                         .filter((col("trip_marker") == "start") & (col("prev_trip_marker") == "end") & (col("TRIP_ID") != col("prev_TRIP_ID")))

stops_df = stops_intermediate_df.withColumn("STOP_DURATION_MINUTES", col("TOTAL_MINUTES") - col("prev_TOTAL_MINUTES")) \
                                .filter(col("STOP_DURATION_MINUTES") > 0) \
                                .select(
                                    col("OBU_ID"),
                                    col("prev_TOTAL_MINUTES").alias("STOP_START_TIME_MIN"),
                                    col("TOTAL_MINUTES").alias("STOP_END_TIME_MIN"),
                                    col("STOP_DURATION_MINUTES"),
                                    col("prev_LINK_ID").alias("STOP_LINK_ID")
                                )

stops_df.persist(StorageLevel.MEMORY_AND_DISK)
num_stops = stops_df.count()
end_time_stops = time.time()
print(f"Trip 간 정지 정보 계산 완료. 총 정지 건수: {num_stops}. 소요 시간: {end_time_stops - start_time_stops:.2f} 초. DF 캐시됨.")


# --- 6. 정지 유형 분류 (K-Means Clustering) ---
start_time_kmeans = time.time()
print("K-Means 클러스터링 시작...")
kmeans_input_df = stops_df.select("STOP_DURATION_MINUTES", "OBU_ID",
                                 "STOP_START_TIME_MIN", "STOP_END_TIME_MIN", "STOP_LINK_ID") \
                         .withColumn("STOP_DURATION_MINUTES_DOUBLE", col("STOP_DURATION_MINUTES").cast("double"))

assembler = VectorAssembler(inputCols=["STOP_DURATION_MINUTES_DOUBLE"], outputCol="features")
stops_vector_df = assembler.transform(kmeans_input_df)

if stops_df.is_cached:
    stops_df.unpersist()
    print("원본 stops_df 캐시 해제됨.")

kmeans = KMeans(k=4, seed=1, featuresCol="features", predictionCol="STOP_TYPE_PREDICTION", initSteps=5, maxIter=30)
model = kmeans.fit(stops_vector_df)
predictions_df = model.transform(stops_vector_df)
centers = model.clusterCenters()
print(f"K-Means 모델 학습 완료. 클러스터 중심: {centers}")

sorted_centers = sorted([(i, center[0]) for i, center in enumerate(centers)], key=lambda x: x[1])
print(f"정렬된 클러스터 중심 (원본 인덱스, 중심값): {sorted_centers}")

# STOP_TYPE을 0부터 시작하는 정렬된 인덱스로 매핑
when_clause = None
for sorted_idx, (original_idx, _) in enumerate(sorted_centers):
    if when_clause is None:
        when_clause = when(col("STOP_TYPE_PREDICTION") == original_idx, sorted_idx)
    else:
        when_clause = when_clause.when(col("STOP_TYPE_PREDICTION") == original_idx, sorted_idx)
when_clause = when_clause.otherwise(None) # 혹시 모를 예외 처리

predictions_with_type = predictions_df.withColumn("STOP_TYPE", when_clause.cast(IntegerType())) # STOP_TYPE은 0, 1, 2, 3

final_stops_df = predictions_with_type.select(
    "OBU_ID",
    "STOP_START_TIME_MIN",
    "STOP_END_TIME_MIN",
    col("STOP_DURATION_MINUTES_DOUBLE").alias("STOP_DURATION_MINUTES"), # 원래 이름으로 복구
    "STOP_LINK_ID",
    "STOP_TYPE"
)

final_stops_df.persist(StorageLevel.MEMORY_AND_DISK)
num_final_stops = final_stops_df.count()
end_time_kmeans = time.time()
print(f"K-Means 클러스터링 및 유형 매핑 완료 (숫자형 타입). 최종 정지 건수: {num_final_stops}. 소요 시간: {end_time_kmeans - start_time_kmeans:.2f} 초. 최종 DF 캐시됨.")
final_stops_df.show(5)

# --- 6.1. 정지 유형별 실제 시간 범위 확인 (텍스트 파일 저장을 위해 결과 저장) ---
start_time_ranges = time.time()
print("\n--- 정지 유형별 실제 시간 범위 계산 시작 ---")
type_labels_en_range = {
    0: "Short Stop(0)",
    1: "Work Stop(1)",
    2: "End-of-Day Stop(2)",
    3: "Long Stop(3)"
}
stop_ranges_output_lines = ["[정지 유형별 실제 포함된 정지 시간 범위]"] # 텍스트 파일 저장용 리스트
try:
    stop_ranges = final_stops_df.groupBy("STOP_TYPE") \
        .agg(
            min("STOP_DURATION_MINUTES").alias("min_duration_minutes"),
            max("STOP_DURATION_MINUTES").alias("max_duration_minutes"),
            count("*").alias("count")
        ).orderBy("STOP_TYPE")

    stop_ranges_collected = stop_ranges.collect()

    # 콘솔 출력 및 파일 저장용 문자열 생성
    for row in stop_ranges_collected:
        stop_type = row["STOP_TYPE"]
        type_label = type_labels_en_range.get(stop_type, f"Unknown Type({stop_type})")
        min_dur = row["min_duration_minutes"]
        max_dur = row["max_duration_minutes"]
        count_val = row["count"]
        line1 = f"- {type_label}:"
        line2 = f"  - Count: {count_val}"
        line3 = f"  - Min Duration: {min_dur:.2f} 분 (~ {min_dur/60:.2f} 시간)"
        line4 = f"  - Max Duration: {max_dur:.2f} 분 (~ {max_dur/60:.2f} 시간)"
        print(line1)
        print(line2)
        print(line3)
        print(line4)
        stop_ranges_output_lines.extend([line1, line2, line3, line4]) # 리스트에 추가

except Exception as e:
    error_line = f"오류: 정지 유형별 시간 범위 계산 중 - {e}"
    print(error_line)
    stop_ranges_output_lines.append(error_line) # 오류 메시지도 추가
end_time_ranges = time.time()
print(f"--- 정지 유형별 실제 시간 범위 계산 완료. 소요 시간: {end_time_ranges - start_time_ranges:.2f} 초 ---\n")


# --- 7. 분석 및 결과 저장 (모든 LINK_ID 기반 위치 분석 포함) ---
start_time_analysis = time.time()
print("분석 및 결과 저장 시작...")

type_labels_en = {
    0: "Short Stop(0)",
    1: "Work Stop(1)",
    2: "End-of-Day Stop(2)",
    3: "Long Stop(3)"
}

# 파일 저장용 문자열 리스트 초기화
hist_counts_output_lines = []
scatter_counts_output_lines = []

# 7.1 정지 시간 분포 - 누적 히스토그램 (범례에 개수/비율 추가)
print(" - 정지 시간 분포 분석 (유형별 누적) 중...")
try:
    sample_fraction = 0.1 # 10% 샘플링
    print(f"   [주의] 히스토그램 샘플링 비율: {sample_fraction*100}%. 드라이버 메모리 사용량 주의.")
    hist_sample_df = final_stops_df.select("STOP_DURATION_MINUTES", "STOP_TYPE") \
                                   .sample(False, sample_fraction) \
                                   .toPandas()

    if not hist_sample_df.empty:
        hist_sample_df['STOP_TYPE_LABEL'] = hist_sample_df['STOP_TYPE'].map(type_labels_en)
        type_counts_hist_sample = hist_sample_df['STOP_TYPE_LABEL'].value_counts()
        total_samples_hist = len(hist_sample_df)

        # 콘솔 출력 및 파일 저장용 문자열 생성
        hist_counts_header = "--- 누적 히스토그램 샘플 내 유형별 개수 ---"
        print(hist_counts_header)
        hist_counts_output_lines.append(hist_counts_header)
        print(type_counts_hist_sample)
        hist_counts_output_lines.append(type_counts_hist_sample.to_string()) # Series를 문자열로
        hist_total_line = f"누적 히스토그램 샘플 전체 개수: {total_samples_hist}"
        print(hist_total_line)
        hist_counts_output_lines.append(hist_total_line)

        data_to_stack = []
        labels_to_stack = []
        for i in range(4):
            type_data = hist_sample_df[hist_sample_df['STOP_TYPE'] == i]['STOP_DURATION_MINUTES']
            if not type_data.empty:
                data_to_stack.append(type_data)
                labels_to_stack.append(type_labels_en[i])

        if data_to_stack:
            plt.figure(figsize=(12, 7))
            plt.hist(data_to_stack, bins=50, stacked=True, label=labels_to_stack, log=True)
            plt.title(f'Stacked Stop Duration Distribution by Type (Sampled {sample_fraction*100}%, Log Scale)')
            plt.xlabel('Stop Duration (Minutes)')
            plt.ylabel('Frequency (Log Scale)')
            plt.grid(True, axis='y')
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = []
            for label in labels:
                count = type_counts_hist_sample.get(label, 0)
                percentage = (count / total_samples_hist * 100) if total_samples_hist > 0 else 0
                new_labels.append(f"{label} (n={count}, {percentage:.1f}%)")
            plt.legend(handles=handles, labels=new_labels, title='Stop Type')
            plt.tight_layout()
            plt.savefig(stop_time_hist_path)
            plt.close()
            print(f"   정지 시간 (유형별 누적) 히스토그램 저장 완료: {stop_time_hist_path}")
        else:
            print("   스택할 데이터가 없어 정지 시간 누적 히스토그램을 생성하지 않습니다.")
    else:
        print("   샘플링된 데이터가 없어 정지 시간 히스토그램을 생성할 수 없습니다.")

except Exception as e:
    print(f"   정지 시간 (유형별 누적) 히스토그램 생성/저장 중 오류: {e}")
    traceback.print_exc()


# 7.2 정지 유형 별 분포 히스토그램 (Spark 집계 후 Pandas)
# (이전 코드와 동일 - 출력 저장 없음)
print(" - 정지 유형별 분포 분석 중...")
try:
    stop_type_counts_df = final_stops_df.groupBy("STOP_TYPE").count().orderBy("STOP_TYPE")
    stop_type_counts_pd = stop_type_counts_df.toPandas()

    if not stop_type_counts_pd.empty:
        stop_type_counts_pd['STOP_TYPE_LABEL'] = stop_type_counts_pd['STOP_TYPE'].map(type_labels_en)
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(x='STOP_TYPE_LABEL', y='count', data=stop_type_counts_pd,
                         order=[type_labels_en[i] for i in sorted(stop_type_counts_pd['STOP_TYPE'].unique())])
        plt.title('Distribution of Stop Types (Sorted by Duration)')
        plt.xlabel('Stop Type (Cluster Index)')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(stop_type_hist_path)
        plt.close()
        print(f"   정지 유형 히스토그램 저장 완료 (값 표시됨): {stop_type_hist_path}")
    else:
        print("   정지 유형 데이터가 없어 히스토그램을 생성할 수 없습니다.")
except Exception as e:
    print(f"   정지 유형 히스토그램 생성/저장 중 오류: {e}")
    traceback.print_exc()


# 7.3 정지 위치(LINK_ID 기준) 별 빈도 (모든 결과 CSV 저장)
# (이전 코드와 동일 - 출력 저장 없음)
print(" - 정지 위치(LINK_ID)별 빈도 분석 및 저장 중 (모든 LINK)...")
try:
    location_link_freq = final_stops_df.groupBy("STOP_LINK_ID") \
                                       .agg(count("*").alias("frequency")) \
                                       .orderBy(desc("frequency"))
    start_time_csv1 = time.time()
    temp_csv_path1 = location_link_freq_csv_path + "_temp"
    final_csv_path1 = location_link_freq_csv_path
    location_link_freq.coalesce(1).write.csv(temp_csv_path1, header=True, mode="overwrite")
    csv_files1 = glob.glob(os.path.join(temp_csv_path1, "part-*.csv"))
    if csv_files1:
        if os.path.exists(final_csv_path1): os.remove(final_csv_path1)
        os.rename(csv_files1[0], final_csv_path1)
        shutil.rmtree(temp_csv_path1)
        end_time_csv1 = time.time()
        print(f"   모든 정지 LINK 빈도 CSV 저장 완료: {final_csv_path1}. 소요 시간: {end_time_csv1 - start_time_csv1:.2f} 초")
    else:
        print(f"   오류: CSV part 파일 찾기 실패 - {temp_csv_path1}")
except Exception as e:
    print(f"   정지 위치(LINK_ID) 빈도 분석/저장 중 오류: {e}")
    traceback.print_exc()


# 7.4 정지 위치(LINK_ID) 및 정지 유형별 빈도 (Pivot 형태로 수정)
# (이전 코드와 동일 - 출력 저장 없음)
print(" - 정지 위치(LINK_ID) 및 유형별 빈도 분석(Pivot) 및 저장 중...")
try:
    stop_type_values = [0, 1, 2, 3]
    location_link_type_pivot = final_stops_df.groupBy("STOP_LINK_ID") \
        .pivot("STOP_TYPE", stop_type_values) \
        .agg(count("*")) \
        .na.fill(0)
    pivot_cols = [str(val) for val in stop_type_values]
    location_link_type_pivot = location_link_type_pivot.withColumn(
        "total_freq", sum(col(c) for c in pivot_cols))
    select_expr = ["STOP_LINK_ID"] + [col(c).alias(f"Type_{c}") for c in pivot_cols] + ["total_freq"]
    location_link_type_pivot = location_link_type_pivot.selectExpr(*select_expr).orderBy(desc("total_freq"))
    start_time_csv2 = time.time()
    temp_csv_path2 = location_link_type_pivot_csv_path + "_temp"
    final_csv_path2 = location_link_type_pivot_csv_path
    location_link_type_pivot.coalesce(1).write.csv(temp_csv_path2, header=True, mode="overwrite")
    csv_files2 = glob.glob(os.path.join(temp_csv_path2, "part-*.csv"))
    if csv_files2:
        if os.path.exists(final_csv_path2): os.remove(final_csv_path2)
        os.rename(csv_files2[0], final_csv_path2)
        shutil.rmtree(temp_csv_path2)
        end_time_csv2 = time.time()
        print(f"   모든 정지 LINK-TYPE 빈도 (Pivot) CSV 저장 완료: {final_csv_path2}. 소요 시간: {end_time_csv2 - start_time_csv2:.2f} 초")
        location_link_type_pivot.show(5, truncate=False)
    else:
        print(f"   오류: CSV part 파일 찾기 실패 - {temp_csv_path2}")
except Exception as e:
    print(f"   정지 위치(LINK_ID) 및 유형별 빈도 (Pivot) CSV 저장 중 오류: {e}")
    traceback.print_exc()

end_time_analysis = time.time()
print(f"분석 및 결과 저장 단계 완료. 소요 시간: {end_time_analysis - start_time_analysis:.2f} 초")

# --- 8. K-Means 결과 시각화 (Box Plot 수정 및 주석 추가) ---
start_time_viz = time.time()
print("K-Means 클러스터 시각화 시작...")
try:
    sample_fraction_plot = 0.1 # 10% 샘플링
    print(f"   [주의] K-Means 시각화 샘플링 비율: {sample_fraction_plot*100}%. 드라이버 메모리 사용량 주의.")
    sampled_for_plot = final_stops_df.select("STOP_DURATION_MINUTES", "STOP_TYPE").sample(False, sample_fraction_plot)
    sampled_for_plot_pd = sampled_for_plot.toPandas()

    if not sampled_for_plot_pd.empty:
        sampled_for_plot_pd['STOP_TYPE_LABEL'] = sampled_for_plot_pd['STOP_TYPE'].map(type_labels_en)
        stop_type_label_order = [type_labels_en[i] for i in range(4)]

        summary_stats_kmeans_pd = sampled_for_plot_pd.groupby('STOP_TYPE_LABEL')['STOP_DURATION_MINUTES'].agg(['min', 'max', 'mean', 'count']).reindex(stop_type_label_order)
        print("   샘플링 데이터 기반 K-Means 클러스터별 통계:")
        print(summary_stats_kmeans_pd)

        print(" - K-Means Box Plot 생성 중 (선형 스케일, 주석 포함)...")
        fig_kmeans_box, ax_kmeans_box = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='STOP_TYPE_LABEL', y='STOP_DURATION_MINUTES', data=sampled_for_plot_pd, order=stop_type_label_order, ax=ax_kmeans_box)
        # *** 로그 스케일 제거 ***
        # ax_kmeans_box.set_yscale('log')

        ax_kmeans_box.set_title(f'Distribution of Stop Durations by K-Means Cluster (Sampled {sample_fraction_plot*100}%)') # 제목에서 Log Scale 제거
        ax_kmeans_box.set_xlabel('Stop Type (Sorted Cluster Index)')
        ax_kmeans_box.set_ylabel('Stop Duration (Minutes)') # Y축 레이블에서 Log Scale 제거
        ax_kmeans_box.tick_params(axis='x', rotation=45)

        # 주석 추가 로직 (선형 스케일용 Y축 조정)
        current_bottom_kb, current_top_kb = ax_kmeans_box.get_ylim()
        # 선형 스케일에 맞는 상단 여백 조정 (예: 1.25배)
        new_top_limit_kb = current_top_kb * 1.25
        ax_kmeans_box.set_ylim(bottom=current_bottom_kb, top=new_top_limit_kb)
        annotation_y_pos_kb = new_top_limit_kb * 0.98 # 상단 근처 위치 (98% 지점)

        x_coords_kb = range(len(stop_type_label_order))
        for i, label in enumerate(stop_type_label_order):
            if label in summary_stats_kmeans_pd.index and pd.notna(summary_stats_kmeans_pd.loc[label, 'count']):
                stats = summary_stats_kmeans_pd.loc[label]
                # Min/Max/Avg 값 포맷팅 개선 (소수점 1자리)
                label_text = f"Min: {stats['min']:,.1f}\nMax: {stats['max']:,.1f}\nAvg: {stats['mean']:,.1f}\n(N={int(stats['count'])})"
                ax_kmeans_box.text(x_coords_kb[i], annotation_y_pos_kb, label_text,
                                 horizontalalignment='center', verticalalignment='top', size=8,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7))

        plt.subplots_adjust(top=0.90, bottom=0.15)
        # *** 파일명 원복 ***
        plt.savefig(kmeans_plot_path)
        plt.close(fig_kmeans_box)
        print(f"   K-Means 클러스터 Box Plot (샘플, 주석 포함, 선형 스케일) 저장 완료: {kmeans_plot_path}")

        # --- K-Means Scatter Plot (로그 스케일 유지) ---
        print(" - K-Means Scatter Plot 생성 중 (로그 스케일)...")
        plt.figure(figsize=(12, 8))
        sns.stripplot(x='STOP_TYPE_LABEL', y='STOP_DURATION_MINUTES', data=sampled_for_plot_pd, order=stop_type_label_order, jitter=0.3, alpha=0.5, size=3)
        plt.yscale('log') # Scatter는 로그 스케일 유지
        plt.title(f'Sampled Stop Durations by K-Means Cluster ({sample_fraction_plot*100}%, Log Scale)')
        plt.xlabel('Stop Type (Sorted Cluster Index)')
        plt.ylabel('Stop Duration (Minutes, Log Scale)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(kmeans_sampled_scatter_path)
        plt.close()
        print(f"   K-Means 클러스터 샘플 Scatter Plot 저장 완료: {kmeans_sampled_scatter_path}")
        # --- Scatter Plot 끝 ---

    else:
        print("   샘플링된 데이터가 없어 K-Means 시각화를 생성할 수 없습니다.")

except Exception as e:
    print(f"   K-Means 클러스터 시각화 중 오류: {e}")
    traceback.print_exc()
end_time_viz = time.time()
print(f"K-Means 시각화 완료. 소요 시간: {end_time_viz - start_time_viz:.2f} 초")


# --- 8.1 요청하신 산점도 생성 (범례에 개수 및 비율 추가) ---
start_time_scatter = time.time()
print("요청하신 정지 유형별 산점도 생성 시작 (범례 개수 및 비율 포함)...")
try:
    sample_fraction_scatter = 0.01 # 1% 샘플링
    print(f"   [주의] 산점도 샘플링 비율: {sample_fraction_scatter*100}%. 드라이버 메모리 사용량 주의.")
    scatter_sample = final_stops_df.select("STOP_DURATION_MINUTES", "STOP_TYPE").sample(False, sample_fraction_scatter)
    scatter_sample_df = scatter_sample.toPandas()

    if not scatter_sample_df.empty:
        scatter_sample_df["STOP_DURATION_HOURS"] = scatter_sample_df["STOP_DURATION_MINUTES"] / 60
        scatter_sample_df = scatter_sample_df.reset_index().rename(columns={'index': 'Stop Sequence Index'})
        scatter_sample_df['STOP_TYPE_LABEL'] = scatter_sample_df['STOP_TYPE'].map(type_labels_en)
        stop_type_label_order = [type_labels_en[i] for i in range(4)]

        # 콘솔 출력 및 파일 저장용 문자열 생성
        type_counts_scatter_sample = scatter_sample_df['STOP_TYPE_LABEL'].value_counts()
        total_samples_scatter = len(scatter_sample_df)
        scatter_counts_header = "--- 산점도 샘플 내 유형별 개수 ---"
        print(scatter_counts_header)
        scatter_counts_output_lines.append(scatter_counts_header)
        print(type_counts_scatter_sample)
        scatter_counts_output_lines.append(type_counts_scatter_sample.to_string()) # Series를 문자열로
        scatter_total_line = f"산점도 샘플 전체 개수: {total_samples_scatter}"
        print(scatter_total_line)
        scatter_counts_output_lines.append(scatter_total_line)

        plt.figure(figsize=(15, 8))
        ax_scatter = sns.scatterplot(
            data=scatter_sample_df, x="Stop Sequence Index", y="STOP_DURATION_HOURS",
            hue="STOP_TYPE_LABEL", style="STOP_TYPE_LABEL",
            hue_order=stop_type_label_order, style_order=stop_type_label_order,
            s=50, alpha=0.7)
        plt.title(f'Stop Duration vs. Sequence Index by Type (Sampled {sample_fraction_scatter*100}%)')
        plt.xlabel('Stop Sequence Index (Sampled)')
        plt.ylabel('Stop Duration (Hours)')
        plt.grid(True)

        handles, labels = ax_scatter.get_legend_handles_labels()
        new_labels = []
        legend_title = "STOP_TYPE_LABEL"
        processed_labels = set()
        final_handles = []
        final_labels = []
        for handle, label in zip(handles, labels):
            if label == legend_title: continue
            if label in processed_labels: continue
            count = type_counts_scatter_sample.get(label, 0)
            percentage = (count / total_samples_scatter * 100) if total_samples_scatter > 0 else 0
            new_label_text = f"{label} (n={count}, {percentage:.1f}%)"
            final_handles.append(handle)
            final_labels.append(new_label_text)
            processed_labels.add(label)
        ax_scatter.legend(handles=final_handles, labels=final_labels, title='Stop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(stop_scatter_plot_path)
        plt.close()
        print(f"   정지 유형별 산점도 저장 완료: {stop_scatter_plot_path}")
    else:
        print("   샘플링된 데이터가 없어 산점도를 생성할 수 없습니다.")

except Exception as e:
    print(f"   정지 유형별 산점도 생성 중 오류: {e}")
    traceback.print_exc()
end_time_scatter = time.time()
print(f"정지 유형별 산점도 생성 완료. 소요 시간: {end_time_scatter - start_time_scatter:.2f} 초")

# --- 8.2. 요약 정보 텍스트 파일 저장 ---
print(f"요약 정보 텍스트 파일 저장 시작: {summary_stats_txt_path}")
try:
    # 모든 출력 라인들을 하나의 리스트로 결합
    all_output_lines = stop_ranges_output_lines + ["\n"] + hist_counts_output_lines + ["\n"] + scatter_counts_output_lines

    with open(summary_stats_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_output_lines))
    print(f"요약 정보 텍스트 파일 저장 완료: {summary_stats_txt_path}")
except Exception as e:
    print(f"오류: 요약 정보 텍스트 파일 저장 중 - {e}")
    traceback.print_exc()

# --- 9. SparkSession 종료 ---
start_time_stop = time.time()
print("캐시 해제 및 SparkSession 종료 시작...")
if 'final_stops_df' in locals() and final_stops_df.is_cached:
    final_stops_df.unpersist()
    print("final_stops_df 캐시 해제됨.")
spark.stop()
end_time_stop = time.time()
print(f"SparkSession 종료됨. 소요 시간: {end_time_stop - start_time_stop:.2f} 초")

total_end_time = time.time()
print(f"\n전체 스크립트 실행 완료. 총 소요 시간: {total_end_time - start_time_spark:.2f} 초")
