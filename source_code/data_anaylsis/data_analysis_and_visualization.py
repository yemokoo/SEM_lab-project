# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
import itertools

# --- 1. 경로 및 파라미터 설정 ---
print("--- 1. 경로 및 파라미터 설정 ---")

PARQUET_BASE_PATH = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(MONTH_FULL)"
SHAPEFILE_PATH = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\main_road_network_level_5.5\level5_5_link_probe_32_2020.shp"
BASE_OUTPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Data_Analysis"
STATISTICS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Statistics")
VISUALIZATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Visualization")
YEARLY_LINK_VOLUME_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "link_yearly_volume_2020")
DAILY_TOTAL_VOLUME_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "daily_total_volume_2020")
CLUSTERED_STOPS_PARQUET_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "clustered_stops.parquet")
HEATMAP_HTML = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure1_traffic_heatmap_ALL_LINKS.html")
BOXPLOT_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure2_daily_volume_boxplot.png")
SCATTER_PLOT_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure3_stop_duration_scatter.png")
TARGET_CRS = "EPSG:4326"

os.makedirs(STATISTICS_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

try:
    plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)
except Exception as e:
    print(f"경고: 한글 폰트('Malgun Gothic') 설정 실패. 오류: {e}")

processed_schema = StructType([
    StructField("OBU_ID", IntegerType(), True),
    StructField("DATETIME", StringType(), True),
    StructField("LINK_ID", IntegerType(), True),
    StructField("LINK_LENGTH", DoubleType(), True),
    StructField("DRIVING_TIME_MINUTES", DoubleType(), True),
    StructField("STOPPING_TIME", DoubleType(), True),
    StructField("CUMULATIVE_DRIVING_TIME_MINUTES", DoubleType(), True),
    StructField("CUMULATIVE_STOPPING_TIME_MINUTES", DoubleType(), True),
    StructField("CUMULATIVE_LINK_LENGTH", DoubleType(), True),
    StructField("SIGUNGU_ID", StringType(), True),
    StructField("AREA_ID", IntegerType(), True),
    StructField("START_TIME_MINUTES", DoubleType(), True),
])


# --- 2. Spark을 이용한 데이터 분석 (월별 순차 처리 방식으로 전면 수정) ---
def run_spark_analysis():
    """Spark을 사용하여 연간/일별 통행량 및 정지 유형 분석을 수행합니다."""
    print("\n--- 2. Spark 분석 시작 (월별 순차 처리) ---")
    
    spark = (
        SparkSession.builder.appName("CombinedTrafficAnalysis_Monthly")
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
    
    # 최종 결과를 누적할 비어있는 데이터프레임 초기화
    yearly_link_volume_acc = None
    daily_total_volume_acc = None
    stopping_df_acc = None
    
    # 1월부터 12월까지 순차적으로 처리
    for month in range(1, 13):
        year = 2020
        month_str = f"{year}-{month:02d}"
        monthly_path = os.path.join(PARQUET_BASE_PATH, month_str)
        
        print(f"\n--- {month_str} 데이터 처리 시작 ---")
        
        try:
            # 해당 월의 데이터만 읽기
            df_month = spark.read.schema(processed_schema).parquet(monthly_path)
            # 월별 데이터 캐싱 (메모리 부담이 적음)
            df_month.cache()
            
            count = df_month.count()
            if count == 0:
                print(f"  - {month_str} 폴더에 데이터가 없습니다. 건너뜁니다.")
                df_month.unpersist()
                continue
            
            print(f"  - {month_str} 데이터 {count:,} 건 로드 완료.")

            # 2a. 월간 링크별 통행량 집계 및 누적
            monthly_link_volume = df_month.groupBy("LINK_ID").count()
            if yearly_link_volume_acc is None:
                yearly_link_volume_acc = monthly_link_volume
            else:
                # full_outer join을 통해 두 데이터프레임의 count를 합산
                yearly_link_volume_acc = yearly_link_volume_acc.join(monthly_link_volume, "LINK_ID", "full_outer") \
                    .select(
                        F.col("LINK_ID"),
                        (F.coalesce(yearly_link_volume_acc["count"], F.lit(0)) + F.coalesce(monthly_link_volume["count"], F.lit(0))).alias("count")
                    )
            print(f"  - {month_str} 링크 통행량 누적 완료.")

            # 2b. 월간 일별 총 통행량 집계 및 누적
            monthly_daily_volume = df_month.withColumn("date", F.to_date("DATETIME")).groupBy("date").count().withColumnRenamed("count", "Daily_Volume")
            if daily_total_volume_acc is None:
                daily_total_volume_acc = monthly_daily_volume
            else:
                daily_total_volume_acc = daily_total_volume_acc.unionByName(monthly_daily_volume)
            print(f"  - {month_str} 일별 통행량 누적 완료.")

            # 2c. 월간 정지 시간 데이터 누적
            monthly_stopping_df = df_month.filter(F.col("STOPPING_TIME") > 0)
            if stopping_df_acc is None:
                stopping_df_acc = monthly_stopping_df
            else:
                stopping_df_acc = stopping_df_acc.unionByName(monthly_stopping_df)
            print(f"  - {month_str} 정지 데이터 누적 완료.")
            
            df_month.unpersist() # 월별 캐시 해제

        except Exception as e:
            print(f"  - 🚨 {month_str} 처리 중 오류 발생: {e}. 다음 월로 진행합니다.")
            continue
    
    print("\n--- 모든 월 데이터 누적 완료. 최종 저장 및 분석 시작 ---")
    try:
        # 최종 집계된 데이터로 파일 저장 및 분석 수행
        # 2a. 최종 연간 링크별 통행량 저장
        if yearly_link_volume_acc:
            yearly_link_volume_acc.withColumnRenamed("count", "Yearly_Volume") \
                .withColumn("Year", F.lit(2020)).select("LINK_ID", "Yearly_Volume", "Year") \
                .coalesce(1).write.mode("overwrite").option("header", "true").csv(YEARLY_LINK_VOLUME_CSV_PATH)
            print("  - ✅ 최종 연간 링크별 통행량 CSV 폴더 저장 완료.")
        
        # 2b. 최종 일별 총 통행량 저장
        if daily_total_volume_acc:
            daily_total_volume_acc.withColumn("Day_of_Week", F.date_format(F.col("date"), "E")) \
                .coalesce(1).write.mode("overwrite").option("header", "true").csv(DAILY_TOTAL_VOLUME_CSV_PATH)
            print("  - ✅ 최종 일별 총 통행량 CSV 폴더 저장 완료.")
        
        # 2c. 정지 시간 군집 분석 및 저장 (안정성 및 성능 강화 버전)
        print("  - 2c. 정지 시간 군집 분석 및 저장 시작...")

        # 최종 누적된 데이터가 있는지, 그리고 내용이 비어있지 않은지 확인
        if stopping_df_acc and stopping_df_acc.count() > 0:
            
            # ❗ [수정 1] 빈 파티션 문제를 원천 차단하기 위해 데이터를 단일 파티션으로 통합
            # dropna()로 null 값을 먼저 제거하고, coalesce(1)로 파티션을 하나로 합쳐 안정성 확보
            stopping_df_final = stopping_df_acc.dropna(subset=["STOPPING_TIME"]).coalesce(1)
            
            # ❗ [수정 2] 반복적인 ML 학습을 위해 정제된 데이터를 메모리에 고정(캐시)
            stopping_df_final.persist()
            
            try:
                print(f"  - ✅ 최종 군집 분석 대상 데이터 {stopping_df_final.count():,} 건 준비 완료. 모델 학습을 시작합니다.")
                
                pipeline = Pipeline(stages=[VectorAssembler(inputCols=["STOPPING_TIME"], outputCol="features"), KMeans(k=4, seed=1)])
                model = pipeline.fit(stopping_df_final)
                clustered_df = model.transform(stopping_df_final)
                
                cluster_summary = clustered_df.groupBy("prediction").agg(F.mean("STOPPING_TIME").alias("avg_stop_time")).orderBy("avg_stop_time")
                ranked_clusters = cluster_summary.withColumn("rank", F.row_number().over(Window.orderBy("avg_stop_time")) - 1).collect()
                
                # --- ❗ [수정 2] label_mapping 및 mapping_expr 생성 방식 변경 ---
                
                # 1. Python 딕셔너리로 예측-레이블 맵을 먼저 생성
                label_map_dict = {
                    row['prediction']: f"{stop_type}({row['rank']})" 
                    for row, stop_type in zip(ranked_clusters, ["Short Stop", "Work Stop", "End-of-Day Stop", "Long Stop"])
                }
                
                # 2. 딕셔너리를 [키1, 값1, 키2, 값2, ...] 형태의 평평한 리스트로 변환
                flattened_map = list(itertools.chain.from_iterable(label_map_dict.items()))
                
                # 3. create_map 함수에 올바른 형태로 전달
                mapping_expr = F.create_map(*[F.lit(x) for x in flattened_map])
                
                # ----------------------------------------------------------------

                final_stops_df = clustered_df.withColumn("stop_type_label", mapping_expr[F.col("prediction")])
                final_stops_df.write.mode("overwrite").parquet(CLUSTERED_STOPS_PARQUET_PATH)
                
                print("  - ✅ 정지 시간 군집 분석 데이터 Parquet 저장 완료.")

            finally:
                stopping_df_final.unpersist()
                
        else:
            print("  - ⚠️ 최종 누적된 정지 데이터가 없어 K-Means 군집 분석을 건너뜁니다.")

    finally:
        spark.stop()
        print("\n--- Spark 세션 종료 ---")


# --- 3. 시각화 함수 ---
def create_traffic_heatmap():
    """(5분위수 버전) 도로 네트워크 통행량 Heatmap을 흰색 배경과 5분위 색상으로 생성합니다."""
    print("\n--- 3a. 도로 네트워크 Heatmap 생성 시작 (5분위수 방식) ---")
    try:
        csv_folder_path = YEARLY_LINK_VOLUME_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: Heatmap을 위한 CSV 폴더가 없습니다: {csv_folder_path}. 이 단계를 건너뜁니다.")
            return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 해당 폴더에 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])
        
        gdf_roads = gpd.read_file(SHAPEFILE_PATH)
        df_volume = pd.read_csv(csv_file_path)
        
        gdf_roads['link_id'] = gdf_roads['link_id'].astype(str)
        df_volume.rename(columns={'LINK_ID': 'link_id'}, inplace=True)
        df_volume['link_id'] = df_volume['link_id'].astype(str)
        merged_gdf = gdf_roads.merge(df_volume, on='link_id', how='inner')
        if merged_gdf.crs != TARGET_CRS: merged_gdf = merged_gdf.to_crs(TARGET_CRS)
        
        map_data = merged_gdf[merged_gdf['Yearly_Volume'] > 0].copy()
        print(f"  - 모든 도로 링크({len(map_data):,}개)를 시각화합니다.")

        # ❗ [수정] LinearColormap을 직접 사용하여 객체 생성
        custom_colors = ['blue', 'green', 'yellow', 'orange', 'red']
        colormap = LinearColormap(
            colors=custom_colors,
            vmin=map_data['Yearly_Volume'].min(),
            vmax=map_data['Yearly_Volume'].quantile(0.95)
        )
        colormap.caption = 'Yearly Freight Truck Volume'
        
        m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron', control_scale=True)
        
        folium.GeoJson(
            map_data,
            style_function=lambda x: {
                'color': colormap(x['properties']['Yearly_Volume']),
                'weight': 2.5,
                'opacity': 0.8
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['link_id', 'road_name', 'Yearly_Volume'],
                aliases=['Link ID:', 'Road Name:', 'Yearly Volume:']
            )
        ).add_to(m)
        
        m.add_child(colormap)
        m.save(HEATMAP_HTML)
        print(f"  - ✅ Heatmap HTML 저장 완료: {HEATMAP_HTML}")

    except Exception as e:
        print(f"오류: Heatmap 생성 중 문제가 발생했습니다 - {e}")

def create_daily_volume_boxplot():
    """요일별 통행량 Boxplot을 생성합니다."""
    print("\n--- 3b. 요일별 통행량 Boxplot 생성 시작 ---")
    try:
        csv_folder_path = DAILY_TOTAL_VOLUME_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: Boxplot을 위한 CSV 폴더가 없습니다: {csv_folder_path}. 이 단계를 건너뜁니다.")
            return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 해당 폴더에 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])

        df = pd.read_csv(csv_file_path)
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=day_order, ordered=True)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.boxplot(x='Day_of_Week', y='Daily_Volume', data=df, ax=ax, palette='viridis', width=0.6)

        for i, day in enumerate(day_order):
            subset = df[df['Day_of_Week'] == day]
            if subset.empty: continue
            min_v, max_v, avg_v, n_v = subset['Daily_Volume'].min(), subset['Daily_Volume'].max(), subset['Daily_Volume'].mean(), len(subset)
            stat_text = f"Min: {min_v:,.0f}\nMax: {max_v:,.0f}\nAvg: {avg_v:,.1f}\nN = {n_v}"
            ax.text(i, max_v * 1.05, stat_text, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8), fontsize=10)
        
        ax.set_title('요일별 일일 통행량 분포 (2020년)', fontsize=20, pad=20)
        ax.set_xlabel('요일 (Day of Week)', fontsize=14)
        ax.set_ylabel('일일 통행량 (Daily Volume)', fontsize=14)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(BOXPLOT_PNG, dpi=150)
        print(f"  - ✅ Boxplot 이미지 저장 완료: {BOXPLOT_PNG}")

    except Exception as e:
        print(f"오류: Boxplot 생성 중 문제가 발생했습니다 - {e}")


def create_stop_duration_scatter_plot():
    """정지 유형별 지속 시간 Scatter Plot을 생성합니다."""
    print("\n--- 3c. 정지 유형별 Scatter Plot 생성 시작 ---")
    try:
        if not os.path.exists(CLUSTERED_STOPS_PARQUET_PATH):
             print(f"경고: Scatter Plot을 위한 Parquet 파일이 없습니다: {CLUSTERED_STOPS_PARQUET_PATH}. 이 단계를 건너뜁니다.")
             return

        df = pd.read_parquet(CLUSTERED_STOPS_PARQUET_PATH)
        
        sampled_df = df.sample(frac=0.00001, random_state=42).reset_index(drop=True)
        sampled_df['Stop_Duration_Hours'] = sampled_df['STOPPING_TIME'] / 60
        
        type_order = sorted(sampled_df['stop_type_label'].unique(), key=lambda x: int(x[-2]))
        
        plt.figure(figsize=(18, 9))
        sns.scatterplot(
            data=sampled_df, x=sampled_df.index, y='Stop_Duration_Hours',
            hue='stop_type_label', style='stop_type_label',
            hue_order=type_order, style_order=type_order, s=50
        )
        
        plt.title('정지 유형별 지속 시간 분포 (0.1% 샘플링)', fontsize=16)
        plt.xlabel('정지 순서 인덱스 (샘플링)', fontsize=12)
        plt.ylabel('정지 지속 시간 (시간)', fontsize=12)
        plt.legend(title='정지 유형 (Stop Type)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(SCATTER_PLOT_PNG, dpi=150)
        print(f"  - ✅ Scatter Plot 이미지 저장 완료: {SCATTER_PLOT_PNG}")
        
    except Exception as e:
        print(f"오류: Scatter Plot 생성 중 문제가 발생했습니다 - {e}")


# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    total_start_time = time.time()
    
    # Spark 분석 실행
    run_spark_analysis()
    
    # 시각화 실행
    create_traffic_heatmap()
    create_daily_volume_boxplot()
    create_stop_duration_scatter_plot()
    
    print(f"\n✨ 모든 작업 완료. 총 소요 시간: {(time.time() - total_start_time) / 60:.2f} 분.")