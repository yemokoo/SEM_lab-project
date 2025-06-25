# -*- coding: utf-8 -*-
import glob
import re
from pyspark.sql import SparkSession, Window
# Import necessary SQL functions
from pyspark.sql.functions import (
    col, lit, year, month, quarter, dayofweek, date_format,
    to_date, avg, count, first, last, when, desc, min as spark_min, max as spark_max, # Renamed min, max to avoid conflict
    approx_count_distinct
)
# Import approxQuantile is REMOVED as we will use the DataFrame method
# from pyspark.sql.functions import approxQuantile # <-- 제거됨
from pyspark.sql.types import *
import os
import time # 시간 측정
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import Seaborn
from datetime import datetime
import traceback # 오류 추적용
import shutil # CSV 저장 시 디렉토리 정리용

# Matplotlib will use its default font which supports English
plt.rcParams['axes.unicode_minus'] = False
print("Using default Matplotlib font for English labels.")


# 1. SparkSession 생성 (튜닝된 설정 적용)
spark = (
    SparkSession.builder.appName("TrafficAnalysisFromParquet2020_Tuned_Quartiles_v2") # Updated App Name
    .config("spark.executor.memory", "28g")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.default.parallelism", "200")
    .config("spark.memory.fraction", "0.7")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.executor.cores", "4")
    .config("spark.python.worker.reuse", "true")
    .config("spark.executor.heartbeatInterval", "300s")
    .config("spark.network.timeout", "1000s")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .config("spark.sql.session.timeZone", "Asia/Seoul")
    .getOrCreate()
)
print("SparkSession 생성 완료 (튜닝된 설정 적용).")
print(f"spark.sql.shuffle.partitions = {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"spark.memory.storageFraction = {spark.conf.get('spark.memory.storageFraction')}")


# --- 설정값 ---
# 분석할 연도 (2020년 고정)
analysis_year = 2020

# *** 경로 수정: Parquet 파일이 저장된 경로 ***
parquet_input_path = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY_RAW)"
# *** 경로 수정: 분석 결과 CSV 저장 기본 경로 ***
analysis_output_base_path = r"D:\project\HDT_EVCS_Opt\Data\Processed_Data\Data_Analysis\Traffic_OD" # Base path for all CSVs and plots

# *** 경로 수정: OD 분석 결과 CSV 저장 경로 (연도 포함) ***
od_output_base_path = os.path.join(analysis_output_base_path, f"OD_Frequency_{analysis_year}")
# *** 경로 수정: 통행량 그래프 저장 경로 (이제 analysis_output_base_path 사용) ***
plot_output_base_path = analysis_output_base_path # Use the common base path
plot_output_quarterly = os.path.join(plot_output_base_path, f"Traffic_Volume_Quarterly_{analysis_year}_EN.png")
plot_output_monthly = os.path.join(plot_output_base_path, f"Traffic_Volume_Monthly_SimpleLegend_{analysis_year}_EN.png")
plot_output_dow_incl = os.path.join(plot_output_base_path, f"Traffic_Volume_Dow_Boxplot_InclHoliday_{analysis_year}_EN_v2.png")
plot_output_dow_excl = os.path.join(plot_output_base_path, f"Traffic_Volume_Dow_Boxplot_ExclHoliday_{analysis_year}_EN_v2.png")
plot_output_yearly = os.path.join(plot_output_base_path, f"Traffic_Volume_YearlyType_Boxplot_{analysis_year}_EN.png")

# *** 경로 추가: 연간 총 통행량 CSV 저장 경로 (별도 폴더 지정) ***
total_volume_output_dir = os.path.join(analysis_output_base_path, f"Total_Volume_{analysis_year}")
total_volume_csv_path = os.path.join(total_volume_output_dir, f"total_yearly_volume_{analysis_year}.csv")

# *** 경로 수정: LINK_ID별 연간 통행량 및 그룹 CSV 저장 경로 (별도 폴더 지정) ***
link_volume_output_dir = os.path.join(analysis_output_base_path, f"Link_Volume_Grouped_{analysis_year}") # Updated directory name
link_volume_csv_path = os.path.join(link_volume_output_dir, f"link_yearly_volume_grouped_{analysis_year}.csv") # Updated file name


# 필요한 컬럼 정의
columns_needed_minimal = ["OBU_ID", "TRIP_ID", "TOTAL_MINUTES", "LINK_ID", "processing_date"]

# --- 공휴일 정의 (2020년 고정) ---
holidays_2020_set = {
    datetime(2020, 1, 1).date(), datetime(2020, 1, 24).date(), datetime(2020, 1, 25).date(),
    datetime(2020, 1, 26).date(), datetime(2020, 1, 27).date(), datetime(2020, 3, 1).date(),
    datetime(2020, 4, 15).date(), datetime(2020, 4, 30).date(), datetime(2020, 5, 5).date(),
    datetime(2020, 6, 6).date(), datetime(2020, 8, 15).date(), datetime(2020, 8, 17).date(),
    datetime(2020, 9, 30).date(), datetime(2020, 10, 1).date(), datetime(2020, 10, 2).date(),
    datetime(2020, 10, 3).date(), datetime(2020, 10, 9).date(), datetime(2020, 12, 25).date()
}
holidays_2020_list = list(holidays_2020_set)

# --- 데이터 로딩 및 전처리 ---
try:
    print(f"Parquet 파일 로딩 시작: {parquet_input_path}")
    df_raw = spark.read.parquet(parquet_input_path)

    if "processing_date" in df_raw.columns:
        df_raw = df_raw.withColumn("processing_date", to_date(col("processing_date"), "yyyy-MM-dd"))
        df_raw = df_raw.filter(year(col("processing_date")) == analysis_year)
    else:
        print("오류: 'processing_date' 파티션 컬럼을 찾을 수 없습니다.")
        spark.stop()
        exit()

    # Select only necessary columns and filter out null LINK_IDs early if possible
    df_selected = df_raw.select(columns_needed_minimal).filter(col("LINK_ID").isNotNull()).cache()

    initial_count = df_selected.count() # Get the total count for the year (after filtering null LINK_ID)
    if initial_count == 0:
        print(f"경고: {analysis_year}년 데이터가 입력 경로에 없거나 LINK_ID가 모두 null입니다. 경로 또는 연도를 확인하세요.")
        spark.stop()
        exit()
    print(f"데이터 로딩 및 {analysis_year}년 필터링 완료 (Null LINK_ID 제외). 총 {initial_count} 건")

    # --- 0. Saving Total Yearly Volume (Single Value) ---
    print("\n--- 0. Saving Total Yearly Volume ---")
    start_time_total_vol = time.time()
    try:
        # Ensure the specific output directory exists for total volume
        os.makedirs(total_volume_output_dir, exist_ok=True) # Use the new directory path
        print(f"Ensured total volume output directory exists: {total_volume_output_dir}")

        # Create a Pandas DataFrame for the total volume
        total_volume_df = pd.DataFrame({'Year': [analysis_year], 'Total_Volume': [initial_count]})

        # Save to CSV in the specific directory
        total_volume_df.to_csv(total_volume_csv_path, index=False, encoding='utf-8-sig') # Path already points to the correct location
        end_time_total_vol = time.time()
        print(f"Total yearly volume saved successfully: {total_volume_csv_path}")
        print(f"Total yearly volume saving duration: {end_time_total_vol - start_time_total_vol:.2f} seconds")

    except Exception as e:
        print(f"Error: Failed to save total yearly volume CSV - {e}")
        traceback.print_exc()
    # --- END SECTION 0 ---


except Exception as e:
    print(f"오류: Parquet 파일 로딩 또는 초기 필터링 실패 - {e}")
    traceback.print_exc() # Print full traceback for loading errors
    spark.stop()
    exit()


# --- 1. 통행량 분석 및 그래프 생성 ---
# (Section 1 remains unchanged)
print("\n--- 1. Daily Volume Analysis and Plotting ---")
start_time_volume = time.time()
try:
    os.makedirs(plot_output_base_path, exist_ok=True)
    print(f"Ensured base plot output directory exists: {plot_output_base_path}")
except OSError as e:
    print(f"Error creating base plot directory '{plot_output_base_path}': {e}")
df_daily_volume = df_selected.groupBy("processing_date").count().withColumnRenamed("count", "daily_volume")
df_date_info = df_daily_volume.withColumn("month", month(col("processing_date"))) \
                              .withColumn("quarter", quarter(col("processing_date"))) \
                              .withColumn("day_of_week", date_format(col("processing_date"), "E")) \
                              .withColumn("day_type",
                                  when(col("processing_date").isin(holidays_2020_list), "Holiday")
                                  .when(dayofweek(col("processing_date")).isin([1, 7]), "Weekend")
                                  .otherwise("Weekday")
                              )
df_date_info.cache()
df_avg_volume_quarterly = df_date_info.groupBy("quarter", "day_type") \
                                      .agg(avg("daily_volume").alias("avg_volume"),
                                           count("*").alias("num_days")) \
                                      .orderBy("quarter", "day_type")
print("1.3 Average volume calculation per quarter and day type completed.")
avg_volume_quarterly_pd = df_avg_volume_quarterly.toPandas()
print("1.4 Generating QUARTERLY volume plot...")
try:
    quarters = sorted(avg_volume_quarterly_pd['quarter'].unique())
    day_types_q = ["Weekday", "Weekend", "Holiday"]
    colors_q = {"Weekday": "blue", "Weekend": "green", "Holiday": "red"}
    fig_q, ax_q = plt.subplots(figsize=(12, 7))
    for day_type in day_types_q:
        plot_data = avg_volume_quarterly_pd[avg_volume_quarterly_pd['day_type'] == day_type].set_index('quarter')
        plot_data = plot_data.reindex(quarters)
        valid_quarters = [q for q in quarters if q in plot_data.index and pd.notna(plot_data.loc[q, 'num_days']) and plot_data.loc[q, 'num_days'] > 0]
        if valid_quarters:
            label_parts = [f"Q{q}:{int(plot_data.loc[q, 'num_days'])} days" for q in valid_quarters]
            legend_label = f"{day_type} ({', '.join(label_parts)})"
            ax_q.plot(plot_data.loc[valid_quarters].index, plot_data.loc[valid_quarters, 'avg_volume'], marker='o', linestyle='-', color=colors_q[day_type], label=legend_label)
    ax_q.set_title(f'Average Daily Volume per Quarter by Day Type ({analysis_year})')
    ax_q.set_xlabel('Quarter')
    ax_q.set_ylabel('Average Daily Volume')
    ax_q.set_xticks(quarters)
    ax_q.set_xticklabels([f'Q{q}' for q in quarters])
    ax_q.legend(title="Day Type (Days per Quarter)")
    ax_q.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(plot_output_quarterly)
    print(f"Quarterly volume plot saved successfully: {plot_output_quarterly}")
    plt.close(fig_q)
except Exception as e:
    print(f"Error: Failed to generate or save QUARTERLY plot - {e}")
    traceback.print_exc()
df_avg_volume_monthly = df_date_info.groupBy("month", "day_type") \
                                    .agg(avg("daily_volume").alias("avg_volume"),
                                         count("*").alias("num_days")) \
                                    .orderBy("month", "day_type")
print("1.5 Calculating average volume per MONTH and day type...")
avg_volume_monthly_pd = df_avg_volume_monthly.toPandas()
print("1.6 Generating MONTHLY volume plot (Simple Legend)...")
try:
    months = range(1, 13)
    day_types_m = ["Weekday", "Weekend", "Holiday"]
    colors_m = {"Weekday": "blue", "Weekend": "green", "Holiday": "red"}
    fig_m, ax_m = plt.subplots(figsize=(14, 7))
    for day_type in day_types_m:
        plot_data_type = avg_volume_monthly_pd[avg_volume_monthly_pd['day_type'] == day_type].set_index('month')
        plot_data_reindexed = plot_data_type.reindex(months)
        valid_months_data = plot_data_reindexed.dropna(subset=['num_days']).query('num_days > 0')
        if not valid_months_data.empty:
            legend_label = day_type
            ax_m.plot(valid_months_data.index, valid_months_data['avg_volume'], marker='o', linestyle='-', color=colors_m[day_type], label=legend_label)
    ax_m.set_title(f'Average Daily Volume per Month by Day Type ({analysis_year})')
    ax_m.set_xlabel('Month')
    ax_m.set_ylabel('Average Daily Volume')
    ax_m.set_xticks(months)
    ax_m.legend(title="Day Type")
    ax_m.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(plot_output_monthly)
    print(f"Monthly volume plot (Simple Legend) saved successfully: {plot_output_monthly}")
    plt.close(fig_m)
except Exception as e:
    print(f"Error: Failed to generate or save MONTHLY plot - {e}")
    traceback.print_exc()
dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("1.7 Preparing data for Day of Week box plot (including holidays)...")
daily_volumes_incl_pd = df_date_info.select("day_of_week", "daily_volume").toPandas()
daily_volumes_incl_pd['day_of_week'] = pd.Categorical(daily_volumes_incl_pd['day_of_week'], categories=dow_order, ordered=True)
daily_volumes_incl_pd = daily_volumes_incl_pd.sort_values('day_of_week')
summary_stats_incl_pd = daily_volumes_incl_pd.groupby('day_of_week', observed=False)['daily_volume'].agg(['min', 'max', 'mean', 'count']).reindex(dow_order)
print("1.8 Generating Day of Week volume box plot (including holidays)...")
try:
    fig_dow_incl, ax_dow_incl = plt.subplots(figsize=(16, 9))
    sns.boxplot(x='day_of_week', y='daily_volume', data=daily_volumes_incl_pd,
                order=dow_order, ax=ax_dow_incl, palette="coolwarm", width=0.6)
    ax_dow_incl.set_title(f'Daily Volume Distribution by Day of Week ({analysis_year}, Holidays Included)')
    ax_dow_incl.set_xlabel('Day of Week')
    ax_dow_incl.set_ylabel('Daily Volume')
    ax_dow_incl.grid(True, axis='y', linestyle='--')
    current_bottom, current_top = ax_dow_incl.get_ylim()
    new_top_limit = current_top * 1.20
    ax_dow_incl.set_ylim(bottom=current_bottom, top=new_top_limit)
    annotation_y_pos = new_top_limit * 0.98
    x_coords = range(len(dow_order))
    for i, day in enumerate(dow_order):
        if day in summary_stats_incl_pd.index and pd.notna(summary_stats_incl_pd.loc[day, 'count']):
            stats = summary_stats_incl_pd.loc[day]
            label_text = f"Min: {stats['min']:,.0f}\nMax: {stats['max']:,.0f}\nAvg: {stats['mean']:,.1f}\n(N={int(stats['count'])})"
            ax_dow_incl.text(x_coords[i], annotation_y_pos, label_text,
                             horizontalalignment='center', verticalalignment='top', size=8,
                             bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6))
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.savefig(plot_output_dow_incl)
    print(f"Day of Week box plot (incl. holidays) saved successfully: {plot_output_dow_incl}")
    plt.close(fig_dow_incl)
except Exception as e:
    print(f"Error: Failed to generate or save Day of Week box plot (incl. holidays) - {e}")
    traceback.print_exc()
print("1.9 Preparing data for Day of Week box plot (excluding holidays)...")
daily_volumes_excl_pd = df_date_info.filter(col("day_type") != "Holiday") \
                                    .select("day_of_week", "daily_volume").toPandas()
if not daily_volumes_excl_pd.empty:
    daily_volumes_excl_pd['day_of_week'] = pd.Categorical(daily_volumes_excl_pd['day_of_week'], categories=dow_order, ordered=True)
    daily_volumes_excl_pd = daily_volumes_excl_pd.sort_values('day_of_week')
    summary_stats_excl_pd = daily_volumes_excl_pd.groupby('day_of_week', observed=False)['daily_volume'].agg(['min', 'max', 'mean', 'count']).reindex(dow_order)
else:
    print("Warning: No non-holiday data found for step 1.9/1.10.")
    summary_stats_excl_pd = pd.DataFrame(columns=['min', 'max', 'mean', 'count'], index=pd.Index([], name='day_of_week'))
print("1.10 Generating Day of Week volume box plot (excluding holidays)...")
if not daily_volumes_excl_pd.empty:
    try:
        fig_dow_excl, ax_dow_excl = plt.subplots(figsize=(16, 9))
        sns.boxplot(x='day_of_week', y='daily_volume', data=daily_volumes_excl_pd,
                    order=dow_order, ax=ax_dow_excl, palette="viridis", width=0.6)
        ax_dow_excl.set_title(f'Daily Volume Distribution by Day of Week ({analysis_year}, Holidays Excluded)')
        ax_dow_excl.set_xlabel('Day of Week')
        ax_dow_excl.set_ylabel('Daily Volume')
        ax_dow_excl.grid(True, axis='y', linestyle='--')
        current_bottom, current_top = ax_dow_excl.get_ylim()
        new_top_limit = current_top * 1.20
        ax_dow_excl.set_ylim(bottom=current_bottom, top=new_top_limit)
        annotation_y_pos = new_top_limit * 0.98
        x_coords = range(len(dow_order))
        for i, day in enumerate(dow_order):
            if day in summary_stats_excl_pd.index and pd.notna(summary_stats_excl_pd.loc[day, 'count']):
                stats = summary_stats_excl_pd.loc[day]
                label_text = f"Min: {stats['min']:,.0f}\nMax: {stats['max']:,.0f}\nAvg: {stats['mean']:,.1f}\n(N={int(stats['count'])})"
                ax_dow_excl.text(x_coords[i], annotation_y_pos, label_text,
                                 horizontalalignment='center', verticalalignment='top', size=8,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcyan", alpha=0.6))
        plt.subplots_adjust(top=0.92, bottom=0.08)
        plt.savefig(plot_output_dow_excl)
        print(f"Day of Week box plot (excl. holidays) saved successfully: {plot_output_dow_excl}")
        plt.close(fig_dow_excl)
    except Exception as e:
        print(f"Error: Failed to generate or save Day of Week box plot (excl. holidays) - {e}")
        traceback.print_exc()
else:
    print("Skipping Day of Week box plot (excluding holidays) due to lack of non-holiday data.")
print("1.11 Preparing data for Yearly volume box plot by Day Type...")
yearly_volumes_pd = df_date_info.select("day_type", "daily_volume").toPandas()
type_order = ["Weekday", "Weekend", "Holiday"]
yearly_volumes_pd['day_type'] = pd.Categorical(yearly_volumes_pd['day_type'], categories=type_order, ordered=True)
yearly_volumes_pd = yearly_volumes_pd.sort_values('day_type')
summary_stats_yearly_pd = yearly_volumes_pd.groupby('day_type', observed=False)['daily_volume'].agg(['min', 'max', 'mean', 'count']).reindex(type_order)
print("   Yearly volume data preparation complete.")
print("   Yearly volume summary stats:")
print(summary_stats_yearly_pd)
print("1.12 Generating Yearly Volume Box Plot by Day Type...")
if not yearly_volumes_pd.empty:
    try:
        colors_y = {"Weekday": "blue", "Weekend": "green", "Holiday": "red"}
        fig_yearly, ax_yearly = plt.subplots(figsize=(10, 7))
        sns.boxplot(x='day_type', y='daily_volume', data=yearly_volumes_pd,
                    order=type_order, ax=ax_yearly, palette=colors_y, width=0.5)
        ax_yearly.set_title(f'Daily Volume Distribution by Day Type ({analysis_year})')
        ax_yearly.set_xlabel('Day Type')
        ax_yearly.set_ylabel('Daily Volume')
        ax_yearly.grid(True, axis='y', linestyle='--')
        current_bottom_y, current_top_y = ax_yearly.get_ylim()
        new_top_limit_y = current_top_y * 1.25
        ax_yearly.set_ylim(bottom=current_bottom_y, top=new_top_limit_y)
        annotation_y_pos_y = new_top_limit_y * 0.98
        x_coords_y = range(len(type_order))
        for i, day_type in enumerate(type_order):
            if day_type in summary_stats_yearly_pd.index and pd.notna(summary_stats_yearly_pd.loc[day_type, 'count']):
                stats = summary_stats_yearly_pd.loc[day_type]
                label_text = f"Min: {stats['min']:,.0f}\nMax: {stats['max']:,.0f}\nAvg: {stats['mean']:,.1f}\n(N={int(stats['count'])} days)"
                ax_yearly.text(x_coords_y[i], annotation_y_pos_y, label_text,
                               horizontalalignment='center', verticalalignment='top', size=9,
                               bbox=dict(boxstyle="round,pad=0.3", fc=colors_y.get(day_type, 'gray'), alpha=0.3))
        plt.subplots_adjust(top=0.90, bottom=0.10)
        plt.savefig(plot_output_yearly)
        print(f"Yearly Volume Box Plot by Type saved successfully: {plot_output_yearly}")
        plt.close(fig_yearly)
    except Exception as e:
        print(f"Error: Failed to generate or save Yearly Volume Box Plot by Type - {e}")
        traceback.print_exc()
else:
    print("Skipping Yearly Volume Box Plot by Type due to lack of data.")
end_time_volume = time.time()
print(f"Total Volume analysis and plotting duration: {end_time_volume - start_time_volume:.2f} seconds")


# --- 2. OD 데이터 분석 ---
# (Section 2 remains unchanged)
print("\n--- 2. OD Data Analysis ---")
start_time_od = time.time()
window_spec = Window.partitionBy("processing_date", "OBU_ID", "TRIP_ID").orderBy("TOTAL_MINUTES")
df_od_pairs = df_selected.withColumn("O_LINK_ID", first("LINK_ID", ignorenulls=True).over(window_spec)) \
                         .withColumn("D_LINK_ID", last("LINK_ID", ignorenulls=True).over(window_spec)) \
                         .select("processing_date", "OBU_ID", "TRIP_ID", "O_LINK_ID", "D_LINK_ID") \
                         .distinct()
df_od_pairs = df_od_pairs.filter(col("O_LINK_ID").isNotNull() & col("D_LINK_ID").isNotNull())
df_od_pairs = df_od_pairs.withColumn("month", month(col("processing_date"))) \
                         .withColumn("quarter", quarter(col("processing_date"))) \
                         .withColumn("year", lit(analysis_year))
print("OD pair extraction completed.")
df_od_pairs.cache()
df_origins = df_od_pairs.select(col("O_LINK_ID").alias("LINK_ID"), "year", "month", "quarter")
df_destinations = df_od_pairs.select(col("D_LINK_ID").alias("LINK_ID"), "year", "month", "quarter")
df_all_od_links = df_origins.unionByName(df_destinations)
print("Origin/Destination link combination completed.")
time_units = {"monthly": ["year", "month"], "quarterly": ["year", "quarter"], "yearly": ["year"]}
os.makedirs(od_output_base_path, exist_ok=True)
print(f"Ensured OD output directory exists: {od_output_base_path}")
for unit, group_cols in time_units.items():
    start_time_agg = time.time()
    print(f"   - Calculating {unit.capitalize()} OD frequency...")
    df_freq = df_all_od_links.groupBy(*group_cols, "LINK_ID") \
                             .agg(count("*").alias("Freq")) \
                             .orderBy(*group_cols, desc("Freq"))
    temp_output_dir = os.path.join(od_output_base_path, f"temp_od_frequency_{unit}_{analysis_year}")
    final_csv_name = f"od_frequency_{unit}_{analysis_year}.csv"
    final_csv_path = os.path.join(od_output_base_path, final_csv_name)
    print(f"   - Preparing to write {unit} data to temp dir: {temp_output_dir}")
    print(f"   - Final CSV will be: {final_csv_path}")
    try:
        df_freq.select("LINK_ID", "Freq", *group_cols) \
               .repartition(1) \
               .write \
               .option("header", "true") \
               .mode("overwrite") \
               .csv(temp_output_dir)
        print(f"   - Successfully wrote data to temp dir: {temp_output_dir}")
        csv_files = glob.glob(os.path.join(temp_output_dir, "part-*.csv"))
        if csv_files:
            source_csv_path = csv_files[0]
            print(f"   - Found part file: {source_csv_path}")
            if os.path.exists(final_csv_path):
                print(f"   - Removing existing final file: {final_csv_path}")
                os.remove(final_csv_path)
            print(f"   - Renaming {source_csv_path} to {final_csv_path}")
            os.rename(source_csv_path, final_csv_path)
            try:
                shutil.rmtree(temp_output_dir)
                print(f"   - Temporary directory '{temp_output_dir}' removed.")
            except OSError as e:
                print(f"   - Warning: Could not remove directory '{temp_output_dir}' completely: {e}")
            end_time_agg = time.time()
            print(f"   - {unit.capitalize()} OD frequency CSV saved: {final_csv_path}")
            print(f"   - {unit.capitalize()} aggregation and save duration: {end_time_agg - start_time_agg:.2f} seconds")
        else:
            print(f"   - Warning: No CSV part file found in '{temp_output_dir}'")
            try:
                shutil.rmtree(temp_output_dir)
                print(f"   - Empty temporary directory '{temp_output_dir}' removed.")
            except OSError as e:
                print(f"   - Warning: Could not remove empty directory '{temp_output_dir}': {e}")
            end_time_agg = time.time()
    except Exception as e:
        print(f"Error: Failed to process or save {unit.capitalize()} OD frequency CSV - {e}")
        traceback.print_exc()
        if os.path.exists(temp_output_dir):
            try:
                shutil.rmtree(temp_output_dir)
                print(f"   - Cleaned up temporary directory '{temp_output_dir}' after error.")
            except Exception as cleanup_e:
                print(f"   - Error cleaning up temporary directory '{temp_output_dir}': {cleanup_e}")
if 'df_od_pairs' in locals() and df_od_pairs.is_cached:
    df_od_pairs.unpersist()
    print("Unpersisted df_od_pairs.")
end_time_od = time.time()
print(f"Total OD analysis and save duration: {end_time_od - start_time_od:.2f} seconds")


# --- 3. Yearly Volume per LINK_ID with Quartile Grouping ---
print("\n--- 3. Yearly Volume per LINK_ID with Quartile Grouping ---")
start_time_link_vol = time.time()

try:
    print(f"   - Calculating yearly volume per LINK_ID...")
    # Group df_selected by LINK_ID and count, renaming to 'Volume'
    df_link_yearly_volume = df_selected.groupBy("LINK_ID") \
                                       .agg(count("*").alias("Volume"))

    # Calculate Quartiles for the 'Volume' column using the DataFrame method
    # This method is called on the DataFrame directly.
    # It takes the column name, list of probabilities, and relative error as arguments.
    try:
        # *** CHANGED HERE: Use DataFrame method instead of imported function ***
        quartiles = df_link_yearly_volume.approxQuantile("Volume", [0.25, 0.5, 0.75], 0.01) # Adjust relativeError if needed

        # Check if quartiles were calculated successfully (list should not be empty)
        if not quartiles or len(quartiles) != 3:
             raise ValueError(f"Failed to calculate quartiles. Result: {quartiles}")

        q1 = quartiles[0]
        q2 = quartiles[1]
        q3 = quartiles[2]
        print(f"   - Calculated Volume Quartiles: Q1={q1}, Q2={q2}, Q3={q3}")

        # Add Volume_Group column based on quartiles
        df_link_grouped = df_link_yearly_volume.withColumn("Volume_Group",
            when(col("Volume") > q3, 3)
            .when((col("Volume") > q2) & (col("Volume") <= q3), 2)
            .when((col("Volume") > q1) & (col("Volume") <= q2), 1)
            .otherwise(0)
        )

        # Order by Volume descending
        df_link_grouped = df_link_grouped.orderBy(desc("Volume"))

        # Ensure the specific output directory exists
        os.makedirs(link_volume_output_dir, exist_ok=True)
        print(f"   - Ensured LINK_ID volume output directory exists: {link_volume_output_dir}")

        # Define paths for saving
        temp_output_dir = os.path.join(link_volume_output_dir, f"temp_link_yearly_volume_grouped_{analysis_year}")
        # Final CSV path is already defined as link_volume_csv_path

        print(f"   - Preparing to write LINK_ID yearly volume (grouped) data to temp dir: {temp_output_dir}")
        print(f"   - Final CSV will be: {link_volume_csv_path}")

        # Save the result to CSV
        df_link_grouped.select("LINK_ID", "Volume", "Volume_Group") \
               .repartition(1) \
               .write \
               .option("header", "true") \
               .mode("overwrite") \
               .csv(temp_output_dir)
        print(f"   - Successfully wrote data to temp dir: {temp_output_dir}")

        # Find the generated part file and rename
        csv_files = glob.glob(os.path.join(temp_output_dir, "part-*.csv"))
        if csv_files:
            source_csv_path = csv_files[0]
            print(f"   - Found part file: {source_csv_path}")
            if os.path.exists(link_volume_csv_path):
                print(f"   - Removing existing final file: {link_volume_csv_path}")
                os.remove(link_volume_csv_path)
            print(f"   - Renaming {source_csv_path} to {link_volume_csv_path}")
            os.rename(source_csv_path, link_volume_csv_path)
            try:
                shutil.rmtree(temp_output_dir)
                print(f"   - Temporary directory '{temp_output_dir}' removed.")
            except OSError as e:
                print(f"   - Warning: Could not remove directory '{temp_output_dir}' completely: {e}")

            end_time_link_vol = time.time()
            print(f"   - LINK_ID yearly volume (grouped) CSV saved: {link_volume_csv_path}")
            print(f"   - LINK_ID yearly volume calculation, grouping, and save duration: {end_time_link_vol - start_time_link_vol:.2f} seconds")
        else:
            print(f"   - Warning: No CSV part file found in '{temp_output_dir}'")
            try:
                shutil.rmtree(temp_output_dir)
                print(f"   - Empty temporary directory '{temp_output_dir}' removed.")
            except OSError as e:
                print(f"   - Warning: Could not remove empty directory '{temp_output_dir}': {e}")
            end_time_link_vol = time.time() # Still record time

    except Exception as q_err:
         print(f"Error: Failed during quartile calculation or processing - {q_err}")
         traceback.print_exc()
         # Attempt cleanup if temp dir was created before error
         if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
             try:
                 shutil.rmtree(temp_output_dir)
                 print(f"   - Cleaned up temporary directory '{temp_output_dir}' after quartile error.")
             except Exception as cleanup_e:
                 print(f"   - Error cleaning up temporary directory '{temp_output_dir}' after quartile error: {cleanup_e}")


except Exception as e:
    print(f"Error: Failed to process or save LINK_ID yearly volume (grouped) CSV - {e}")
    traceback.print_exc()
    # Clean up temp directory on general error in section 3
    if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
            print(f"   - Cleaned up temporary directory '{temp_output_dir}' after error.")
        except Exception as cleanup_e:
            print(f"   - Error cleaning up temporary directory '{temp_output_dir}': {cleanup_e}")

# --- END SECTION 3 ---


# --- Cleanup ---
# Unpersist cached dataframes that are no longer needed
if 'df_selected' in locals() and df_selected.is_cached:
    df_selected.unpersist()
    print("\nUnpersisted df_selected.")
# df_od_pairs was already unpersisted after section 2
if 'df_date_info' in locals() and df_date_info.is_cached:
    df_date_info.unpersist()
    print("Unpersisted df_date_info.")


# --- SparkSession 종료 ---
print("\nAll tasks completed. Stopping SparkSession...")
spark.stop()
print("SparkSession stopped.")
