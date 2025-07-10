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
BASE_OUTPUT_DIR = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\Data_Analysis"

STATISTICS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Statistics")
VISUALIZATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Visualization")

# 통계 파일 경로
YEARLY_LINK_VOLUME_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "link_yearly_volume_2020")
DAILY_TOTAL_VOLUME_WITH_HOLIDAYS_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "daily_total_volume_with_holidays")
DAILY_TOTAL_VOLUME_WITHOUT_HOLIDAYS_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "daily_total_volume_without_holidays")
CLUSTERED_STOPS_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "clustered_stops_csv")
DAILY_DISTANCE_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "daily_driving_distance")
DAILY_DISTANCE_STATS_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "daily_driving_distance_stats.csv")
ALL_DRIVING_SEGMENTS_CSV_PATH = os.path.join(STATISTICS_OUTPUT_DIR, "all_driving_segments")

# 시각화 파일 경로
HEATMAP_VIS_DIR = os.path.join(VISUALIZATION_OUTPUT_DIR, "Heatmaps")
BOXPLOT_WITH_HOLIDAYS_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure2_daily_volume_boxplot_with_holidays_en.png")
BOXPLOT_WITHOUT_HOLIDAYS_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure2_daily_volume_boxplot_without_holidays_en.png")
SCATTER_PLOT_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure3_stop_duration_scatter_en.png")
DAILY_DISTANCE_HISTOGRAM_PNG = os.path.join(VISUALIZATION_OUTPUT_DIR, "figure4_daily_distance_histogram.png")
STOP_LOCATION_MAP_DIR = os.path.join(VISUALIZATION_OUTPUT_DIR, "Stop_Location_Maps")

TARGET_CRS = "EPSG:4326"

os.makedirs(STATISTICS_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(HEATMAP_VIS_DIR, exist_ok=True)
os.makedirs(STOP_LOCATION_MAP_DIR, exist_ok=True)

processed_schema = StructType([
    StructField("OBU_ID", IntegerType(), True), StructField("DATETIME", StringType(), True),
    StructField("LINK_ID", IntegerType(), True), StructField("LINK_LENGTH", DoubleType(), True),
    StructField("DRIVING_TIME_MINUTES", DoubleType(), True), StructField("STOPPING_TIME", DoubleType(), True),
    StructField("CUMULATIVE_DRIVING_TIME_MINUTES", DoubleType(), True), StructField("CUMULATIVE_STOPPING_TIME_MINUTES", DoubleType(), True),
    StructField("CUMULATIVE_LINK_LENGTH", DoubleType(), True), StructField("SIGUNGU_ID", StringType(), True),
    StructField("AREA_ID", IntegerType(), True), StructField("START_TIME_MINUTES", DoubleType(), True),
])


# --- 2. Spark을 이용한 데이터 분석 ---
def run_spark_analysis():
    """Spark을 사용하여 연간/일별 통행량, 정지 유형, 일일 주행거리 분석을 수행합니다."""
    print("\n--- 2. Spark 분석 시작 (월별 순차 처리) ---")
    
    spark = SparkSession.builder.appName("CombinedTrafficAnalysis_Monthly").config("spark.driver.memory", "32g").config("spark.driver.host", "127.0.0.1").getOrCreate()
    
    yearly_link_volume_acc, daily_total_volume_acc, stopping_df_acc, daily_distance_acc = None, None, None, None
    all_driving_segments_acc = None
    
    for month in range(1, 13):
        year, month_str = 2020, f"2020-{month:02d}"
        monthly_path = os.path.join(PARQUET_BASE_PATH, month_str)
        print(f"\n--- {month_str} 데이터 처리 시작 ---")
        
        try:
            df_month = spark.read.schema(processed_schema).parquet(monthly_path)
            df_month.cache()
            count = df_month.count()
            if count == 0:
                print(f"  - {month_str} 폴더에 데이터가 없습니다. 건너뜁니다."); df_month.unpersist(); continue
            print(f"  - {month_str} 데이터 {count:,} 건 로드 완료.")

            monthly_link_volume = df_month.groupBy("LINK_ID").count()
            if yearly_link_volume_acc is None: yearly_link_volume_acc = monthly_link_volume
            else: yearly_link_volume_acc = yearly_link_volume_acc.join(monthly_link_volume, "LINK_ID", "full_outer").select(F.col("LINK_ID"), (F.coalesce(yearly_link_volume_acc["count"], F.lit(0)) + F.coalesce(monthly_link_volume["count"], F.lit(0))).alias("count"))

            monthly_daily_volume = df_month.withColumn("date", F.to_date("DATETIME")).groupBy("date").count().withColumnRenamed("count", "Daily_Volume")
            if daily_total_volume_acc is None: daily_total_volume_acc = monthly_daily_volume
            else: daily_total_volume_acc = daily_total_volume_acc.unionByName(monthly_daily_volume)

            monthly_stopping_df = df_month.filter(F.col("STOPPING_TIME") > 0)
            if stopping_df_acc is None: stopping_df_acc = monthly_stopping_df
            else: stopping_df_acc = stopping_df_acc.unionByName(monthly_stopping_df)
            
            monthly_daily_distance = df_month.withColumn("date", F.to_date("DATETIME")).groupBy("OBU_ID", "date").agg(F.sum("LINK_LENGTH").alias("daily_distance_km"))
            if daily_distance_acc is None: daily_distance_acc = monthly_daily_distance
            else: daily_distance_acc = daily_distance_acc.unionByName(monthly_daily_distance)

            if all_driving_segments_acc is None: all_driving_segments_acc = df_month
            else: all_driving_segments_acc = all_driving_segments_acc.unionByName(df_month)
            
            df_month.unpersist()
        except Exception as e:
            print(f"  - 🚨 {month_str} 처리 중 오류 발생: {e}. 다음 월로 진행합니다."); continue
    
    print("\n--- 모든 월 데이터 누적 완료. 최종 저장 및 분석 시작 ---")
    try:
        if all_driving_segments_acc:
             all_driving_segments_acc.coalesce(1).write.mode("overwrite").option("header", "true").csv(ALL_DRIVING_SEGMENTS_CSV_PATH)
             print("  - ✅ 전체 주행 기록 데이터 CSV 저장 완료.")

        if yearly_link_volume_acc:
            yearly_link_volume_acc.withColumnRenamed("count", "Yearly_Volume").withColumn("Year", F.lit(2020)).coalesce(1).write.mode("overwrite").option("header", "true").csv(YEARLY_LINK_VOLUME_CSV_PATH)
            print("  - ✅ 최종 연간 링크별 통행량 CSV 저장 완료.")
        
        if daily_total_volume_acc:
            daily_total_volume_acc.withColumn("Day_of_Week", F.date_format(F.col("date"), "E")).coalesce(1).write.mode("overwrite").option("header", "true").csv(DAILY_TOTAL_VOLUME_WITH_HOLIDAYS_CSV_PATH)
            holidays_2020 = ['2020-01-01', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-03-01', '2020-04-15', '2020-04-30', '2020-05-01', '2020-05-05', '2020-06-06', '2020-08-15', '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-09', '2020-12-25']
            daily_total_volume_acc.filter(~F.col("date").cast("string").isin(holidays_2020)).withColumn("Day_of_Week", F.date_format(F.col("date"), "E")).coalesce(1).write.mode("overwrite").option("header", "true").csv(DAILY_TOTAL_VOLUME_WITHOUT_HOLIDAYS_CSV_PATH)
            print("  - ✅ 최종 일별 통행량(공휴일 포함/제외) CSV 2종 저장 완료.")
            
        if daily_distance_acc:
            daily_distance_acc.coalesce(1).write.mode("overwrite").option("header", "true").csv(DAILY_DISTANCE_CSV_PATH)
            print("  - ✅ 최종 OBU별 일일 주행거리 CSV 저장 완료.")

        if stopping_df_acc and stopping_df_acc.count() > 0:
            stopping_df_final = stopping_df_acc.dropna(subset=["STOPPING_TIME"]).coalesce(1)
            stopping_df_final.persist()
            try:
                print(f"  - ✅ 최종 군집 분석 대상 데이터 {stopping_df_final.count():,} 건 준비 완료.")
                pipeline = Pipeline(stages=[VectorAssembler(inputCols=["STOPPING_TIME"], outputCol="features"), KMeans(k=4, seed=1)])
                model = pipeline.fit(stopping_df_final)
                clustered_df = model.transform(stopping_df_final)
                cluster_summary = clustered_df.groupBy("prediction").agg(F.mean("STOPPING_TIME").alias("avg_stop_time")).orderBy("avg_stop_time")
                ranked_clusters = cluster_summary.withColumn("rank", F.row_number().over(Window.orderBy("avg_stop_time")) - 1).collect()
                label_map_dict = {row['prediction']: f"{stop_type}({row['rank']})" for row, stop_type in zip(ranked_clusters, ["Short Stop", "Work Stop", "End-of-Day Stop", "Long Stop"])}
                flattened_map = list(itertools.chain.from_iterable(label_map_dict.items()))
                mapping_expr = F.create_map(*[F.lit(x) for x in flattened_map])
                final_stops_df = clustered_df.withColumn("stop_type_label", mapping_expr[F.col("prediction")])
                
                final_stops_df.drop("features", "prediction") \
                    .coalesce(1).write.mode("overwrite").option("header", "true").csv(CLUSTERED_STOPS_CSV_PATH)
                    
                print("  - ✅ 정지 시간 군집 분석 데이터 CSV 저장 완료.")
            finally: 
                stopping_df_final.unpersist()
        else: 
            print("  - ⚠️ 최종 누적된 정지 데이터가 없어 K-Means 군집 분석을 건너뜁니다.")
    finally:
        spark.stop()
        print("\n--- Spark 세션 종료 ---")


# --- 3. 시각화 함수 ---
def create_traffic_heatmap():
    """여러 조건에 따라 다양한 버전의 Heatmap을 생성합니다."""
    print("\n--- 3a. 도로 네트워크 Heatmap 생성 시작 ---")
    try:
        csv_folder_path = YEARLY_LINK_VOLUME_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: Heatmap CSV 폴더가 없습니다: {csv_folder_path}. 건너뜁니다."); return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])
        gdf_roads, df_volume = gpd.read_file(SHAPEFILE_PATH), pd.read_csv(csv_file_path)
        
        gdf_roads['link_id'] = gdf_roads['link_id'].astype(str)
        df_volume.rename(columns={'LINK_ID': 'link_id'}, inplace=True)
        df_volume['link_id'] = df_volume['link_id'].astype(str)
        merged_gdf = gdf_roads.merge(df_volume, on='link_id', how='inner')
        if merged_gdf.crs != TARGET_CRS: merged_gdf = merged_gdf.to_crs(TARGET_CRS)
        
        map_data = merged_gdf[merged_gdf['Yearly_Volume'] > 0].copy()
        map_data['quantile_all'] = pd.qcut(map_data['Yearly_Volume'], q=5, labels=False, duplicates='drop')
        color_dict = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'orange', 4: 'red'}
        print("  - ✅ 데이터 준비 및 전체 5분위수 계산 완료.")

        print("\n  --- 등급별 누적 지도 생성 중 ---")
        style_function_all = lambda x: {'color': color_dict.get(x['properties']['quantile_all'], 'grey'), 'weight': 2.5, 'opacity': 0.8}
        for i in range(4, -1, -1):
            filtered_data = map_data[map_data['quantile_all'] >= i]
            percentage = (5 - i) * 20
            filename = os.path.join(HEATMAP_VIS_DIR, f"heatmap_top{percentage}pct.html")
            if i == 0: filename = os.path.join(HEATMAP_VIS_DIR, "heatmap_all.html")
            print(f"    - 상위 {percentage}% 지도 생성 중... ({len(filtered_data):,}개 링크)")
            m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron', control_scale=True)
            folium.GeoJson(filtered_data, style_function=style_function_all, tooltip=folium.GeoJsonTooltip(fields=['link_id', 'road_name', 'Yearly_Volume', 'quantile_all'], aliases=['Link ID:', 'Road Name:', 'Yearly Volume:', 'Overall Quantile:'])).add_to(m)
            m.save(filename)

        print("\n  --- 통행량 상위 25% 지도 생성 중 ---")
        top_25_threshold = map_data['Yearly_Volume'].quantile(0.75)
        top_25_data = map_data[map_data['Yearly_Volume'] >= top_25_threshold].copy()
        top_25_data['quantile_top25'] = pd.qcut(top_25_data['Yearly_Volume'], q=5, labels=False, duplicates='drop')
        style_function_top25 = lambda x: {'color': color_dict.get(x['properties']['quantile_top25'], 'grey'),'weight': 2.5, 'opacity': 0.8}
        filename = os.path.join(HEATMAP_VIS_DIR, "heatmap_top25vol.html")
        print(f"    - 상위 25% 지도 생성 중... ({len(top_25_data):,}개 링크)")
        m_top25 = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron', control_scale=True)
        folium.GeoJson(top_25_data, style_function=style_function_top25, tooltip=folium.GeoJsonTooltip(fields=['link_id', 'road_name', 'Yearly_Volume', 'quantile_top25'], aliases=['Link ID:', 'Road Name:', 'Yearly Volume:', 'Top 25% Quantile:'])).add_to(m_top25)
        m_top25.save(filename)
        print(f"  - ✅ 모든 Heatmap 저장 완료.")

    except Exception as e:
        print(f"오류: Heatmap 생성 중 문제가 발생했습니다 - {e}")

def create_daily_volume_boxplot():
    """공휴일 포함/제외 두 가지 버전의 Boxplot을 모두 생성합니다."""
    print("\n--- 3b. 요일별 통행량 Boxplot 생성 시작 ---")
    plot_configs = [
        {"title": "Daily Volume Distribution by Day of Week (Holidays INCLUDED)", "csv_path": DAILY_TOTAL_VOLUME_WITH_HOLIDAYS_CSV_PATH, "output_png": BOXPLOT_WITH_HOLIDAYS_PNG},
        {"title": "Daily Volume Distribution by Day of Week (Holidays EXCLUDED)", "csv_path": DAILY_TOTAL_VOLUME_WITHOUT_HOLIDAYS_CSV_PATH, "output_png": BOXPLOT_WITHOUT_HOLIDAYS_PNG}
    ]
    for config in plot_configs:
        try:
            print(f"\n  - '{config['title']}' 그래프 생성 중...")
            csv_folder_path = config["csv_path"]
            if not os.path.exists(csv_folder_path):
                print(f"    경고: CSV 폴더가 없습니다: {csv_folder_path}. 건너뜁니다."); continue
            csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
            if not csv_files: raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_folder_path}")
            csv_file_path = os.path.join(csv_folder_path, csv_files[0])
            df = pd.read_csv(csv_file_path)
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=day_order, ordered=True)
            
            fig, ax = plt.subplots(figsize=(16, 10))
            sns.boxplot(x='Day_of_Week', y='Daily_Volume', data=df, ax=ax, palette='viridis', width=0.6, hue='Day_of_Week', legend=False)
            max_y_for_text = 0
            for i, day in enumerate(day_order):
                subset = df[df['Day_of_Week'] == day]
                if subset.empty: continue
                min_v, max_v, avg_v, n_v = subset['Daily_Volume'].min(), subset['Daily_Volume'].max(), subset['Daily_Volume'].mean(), len(subset)
                stat_text = f"Min: {min_v:,.0f}\nMax: {max_v:,.0f}\nAvg: {avg_v:,.1f}\nN = {n_v}"
                text_y_position = max_v
                max_y_for_text = max(max_y_for_text, text_y_position)
                ax.text(i, text_y_position * 1.05, stat_text, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8), fontsize=10)
            
            if max_y_for_text > 0: ax.set_ylim(top=max_y_for_text * 1.35)
            
            ax.set_title(config["title"], fontsize=20, pad=20)
            ax.set_xlabel('Day of Week', fontsize=14); ax.set_ylabel('Daily Volume', fontsize=14)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout(); plt.savefig(config["output_png"], dpi=150); plt.close(fig)
            print(f"    ✅ Boxplot 이미지 저장 완료: {config['output_png']}")
        except Exception as e:
            print(f"    오류: '{config['title']}' 그래프 생성 중 문제가 발생했습니다 - {e}")

def create_daily_distance_histogram():
    """일일 주행거리 분포 히스토그램을 생성하고, 요약 통계량을 CSV로 저장합니다."""
    print("\n--- 3c. 일일 주행거리 히스토그램 및 통계 분석 시작 ---")
    try:
        csv_folder_path = DAILY_DISTANCE_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: 일일 주행거리 CSV 폴더가 없습니다: {csv_folder_path}. 건너뜁니다."); return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])
        df = pd.read_csv(csv_file_path)

        daily_distance_stats = df['daily_distance_km'].describe()
        daily_distance_stats.to_csv(DAILY_DISTANCE_STATS_CSV_PATH)
        print(f"  - ✅ 일일 주행거리 요약 통계 저장 완료: {os.path.basename(DAILY_DISTANCE_STATS_CSV_PATH)}")
        
        plt.figure(figsize=(12, 7))
        upper_bound = df['daily_distance_km'].quantile(0.99)
        sns.histplot(df[df['daily_distance_km'] <= upper_bound]['daily_distance_km'], bins=50, kde=True)
        avg_dist = df['daily_distance_km'].mean()
        plt.axvline(avg_dist, color='red', linestyle='--', label=f'Average: {avg_dist:.1f} km')
        plt.title('Distribution of Daily Driving Distance per Truck (up to 99th percentile)', fontsize=16)
        plt.xlabel('Daily Driving Distance (km)', fontsize=12); plt.ylabel('Number of Daily Trips', fontsize=12)
        plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(DAILY_DISTANCE_HISTOGRAM_PNG, dpi=150); plt.close()
        print(f"  - ✅ 일일 주행거리 히스토그램 저장 완료.")
    except Exception as e:
        print(f"오류: 일일 주행거리 히스토그램 생성 중 문제가 발생했습니다 - {e}")

def create_stop_duration_scatter_plot():
    """정지 유형별 Scatter Plot의 범례에 시간 범위와 개수를 추가하여 2종 생성합니다."""
    print("\n--- 3d. 정지 유형별 Scatter Plot 생성 시작 ---")
    try:
        csv_folder_path = CLUSTERED_STOPS_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: Scatter Plot용 CSV 폴더가 없습니다. 건너뜁니다."); return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        if df.empty: print(f"경고: 정지 데이터가 비어있어 Scatter Plot을 생성할 수 없습니다."); return
            
        total_counts = df['stop_type_label'].value_counts()
        time_ranges = df.groupby('stop_type_label')['STOPPING_TIME'].agg(['min', 'max'])
        
        sampling_frac = 0.001
        sampled_df = df.sample(frac=sampling_frac, random_state=42).reset_index(drop=True)
        if sampled_df.empty: print(f"경고: 샘플링된 데이터가 없어 Scatter Plot을 생성할 수 없습니다."); return
        
        sampled_counts = sampled_df['stop_type_label'].value_counts()
        sampled_df['Stop_Duration_Hours'] = sampled_df['STOPPING_TIME'] / 60
        type_order_original = sorted(sampled_df['stop_type_label'].unique(), key=lambda x: int(x[-2]))

        def create_legend_label_sampled(label):
            count = sampled_counts.get(label, 0)
            time_range = time_ranges.loc[label]
            return f"{label} (n={count}) (Range: {time_range['min']:.1f}-{time_range['max']:.1f} min)"

        sampled_df['legend_label_sampled'] = sampled_df['stop_type_label'].apply(create_legend_label_sampled)
        hue_order_sampled = [create_legend_label_sampled(x) for x in type_order_original]

        plt.figure(figsize=(18, 9))
        sns.scatterplot(data=sampled_df, x=sampled_df.index, y='Stop_Duration_Hours', hue='legend_label_sampled', style='legend_label_sampled', hue_order=hue_order_sampled, style_order=hue_order_sampled, s=50)
        plt.title(f'Stop Duration vs. Sequence Index by Type (Sampled {sampling_frac:.1%}) - Sampled Counts')
        plt.xlabel('Stop Sequence Index (Sampled)'); plt.ylabel('Stop Duration (Hours)')
        plt.legend(title='Stop Type (n = count in sample)'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename_sampled = SCATTER_PLOT_PNG.replace('.png', '_sampled_counts.png')
        plt.savefig(filename_sampled, dpi=150); plt.close()
        print(f"  - ✅ Scatter Plot (샘플링 개수 버전) 저장 완료.")

        def create_legend_label_total(label):
            count = total_counts.get(label, 0)
            time_range = time_ranges.loc[label]
            return f"{label} (N={count}) (Range: {time_range['min']:.1f}-{time_range['max']:.1f} min)"

        sampled_df['legend_label_total'] = sampled_df['stop_type_label'].apply(create_legend_label_total)
        hue_order_total = [create_legend_label_total(x) for x in type_order_original]

        plt.figure(figsize=(18, 9))
        sns.scatterplot(data=sampled_df, x=sampled_df.index, y='Stop_Duration_Hours', hue='legend_label_total', style='legend_label_total', hue_order=hue_order_total, style_order=hue_order_total, s=50)
        plt.title(f'Stop Duration vs. Sequence Index by Type (Sampled {sampling_frac:.1%}) - Total Counts')
        plt.xlabel('Stop Sequence Index (Sampled)'); plt.ylabel('Stop Duration (Hours)')
        plt.legend(title='Stop Type (N = count in total data)'); plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename_total = SCATTER_PLOT_PNG.replace('.png', '_total_counts.png')
        plt.savefig(filename_total, dpi=150); plt.close()
        print(f"  - ✅ Scatter Plot (전체 개수 버전) 저장 완료.")
    except Exception as e:
        print(f"오류: Scatter Plot 생성 중 문제가 발생했습니다 - {e}")
        
def create_stop_location_maps():
    """정지 유형별 위치 지도를 생성합니다."""
    print("\n--- 3e. 정지 유형별 위치 지도 생성 시작 ---")
    try:
        csv_folder_path = CLUSTERED_STOPS_CSV_PATH
        if not os.path.exists(csv_folder_path):
            print(f"경고: 정지 위치 지도를 위한 CSV 파일이 없습니다. 건너뜁니다."); return
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        if not csv_files: raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_folder_path}")
        csv_file_path = os.path.join(csv_folder_path, csv_files[0])
        df_stops = pd.read_csv(csv_file_path)

        gdf_roads = gpd.read_file(SHAPEFILE_PATH, columns=['link_id', 'geometry'])
        gdf_roads.rename(columns={'link_id': 'LINK_ID'}, inplace=True)

        df_stops['LINK_ID'] = df_stops['LINK_ID'].astype(str)
        gdf_roads['LINK_ID'] = gdf_roads['LINK_ID'].astype(str)
        
        stop_counts = df_stops.groupby(['LINK_ID', 'stop_type_label']).size().reset_index(name='stop_count')
        merged_gdf = gdf_roads.merge(stop_counts, on='LINK_ID', how='inner')
        if merged_gdf.crs != TARGET_CRS: merged_gdf = merged_gdf.to_crs(TARGET_CRS)
        
        merged_gdf['lon'] = merged_gdf.geometry.centroid.x
        merged_gdf['lat'] = merged_gdf.geometry.centroid.y
        
        stop_types = sorted(merged_gdf['stop_type_label'].unique(), key=lambda x: int(x[-2]))
        type_colors = {'Short Stop(0)': 'blue', 'Work Stop(1)': 'green', 'End-of-Day Stop(2)': 'purple', 'Long Stop(3)': 'red'}

        for stop_type in stop_types:
            type_data = merged_gdf[merged_gdf['stop_type_label'] == stop_type]
            if type_data.empty: continue
            
            m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron')
            for _, row in type_data.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], radius=max(2, min(10, row['stop_count'] / 5)),
                    color=type_colors.get(stop_type, 'gray'), fill=True, fill_color=type_colors.get(stop_type, 'gray'),
                    tooltip=f"Link: {row['LINK_ID']}<br>Type: {row['stop_type_label']}<br>Count: {row['stop_count']}"
                ).add_to(m)
            
            filename = os.path.join(STOP_LOCATION_MAP_DIR, f"stop_locations_{stop_type.replace('(', '').replace(')', '')}.html")
            m.save(filename)
            print(f"  - ✅ {stop_type} 정지 위치 지도 저장 완료.")
            
        m_all = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron')
        for stop_type in stop_types:
            type_data = merged_gdf[merged_gdf['stop_type_label'] == stop_type]
            fg = folium.FeatureGroup(name=stop_type)
            for _, row in type_data.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], radius=max(2, min(10, row['stop_count'] / 5)),
                    color=type_colors.get(stop_type, 'gray'), fill=True, fill_color=type_colors.get(stop_type, 'gray'),
                    tooltip=f"Link: {row['LINK_ID']}<br>Type: {row['stop_type_label']}<br>Count: {row['stop_count']}"
                ).add_to(fg)
            fg.add_to(m_all)
        
        folium.LayerControl().add_to(m_all)
        filename_all = os.path.join(STOP_LOCATION_MAP_DIR, "stop_locations_all_types.html")
        m_all.save(filename_all)
        print("  - ✅ 모든 유형 종합 지도 저장 완료.")

    except Exception as e:
        print(f"오류: 정지 위치 지도 생성 중 문제가 발생했습니다 - {e}")

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    total_start_time = time.time()
    
    print("\n✨ 데이터 분석 및 시각화 스크립트 시작")
    run_spark_analysis()
    
    create_traffic_heatmap()
    create_daily_volume_boxplot()
    create_daily_distance_histogram()
    create_stop_duration_scatter_plot()
    create_stop_location_maps()
    
    print(f"\n✨ 모든 작업 완료. 총 소요 시간: {(time.time() - total_start_time) / 60:.2f} 분.")