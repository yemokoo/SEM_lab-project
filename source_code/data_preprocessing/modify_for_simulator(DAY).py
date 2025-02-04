from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *

# 1. SparkSession 생성
spark = (
    SparkSession.builder.appName("PathAnalysis")
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

# 2. CSV 파일 스키마 정의
schema = StructType([
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

# 3. 데이터 불러오기 (OBU_ID 결측치 처리)
file_path = r"C:/Users/wngud/Desktop/project/heavy_duty_truck_charging_infra/resource/*.csv"
df = (
    spark.read.csv(file_path, header=True, schema=schema, sep=",")
    .na.fill(0, subset=["OBU_ID"]) 
)

# 4. 데이터 정제 및 변환
df = (
    df.withColumn("LINK_ID", col("LINK_ID").cast(IntegerType()))
    .withColumn("IN_TIME", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("IN_TIME")), "yyyy-MM-dd HHmmss"))
    .withColumn("OUT_TIME", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("OUT_TIME")), "yyyy-MM-dd HHmmss"))
    .na.drop(subset=["IN_TIME", "OUT_TIME"]) 
    .withColumn(
        "DRIVING_TIME",
        (unix_timestamp("OUT_TIME") - unix_timestamp("IN_TIME")).cast("double")
    )
)

filtered_df = df.filter((col("LINK_ID") != -1) & (col("DRIVING_TIME") >= 1))

# 5. TRIP_ID 생성 및 필요한 열 선택 (DATETIME 열 추가 및 포맷 변경)
# DRIVING_TIME_MINUTES 열 추가, LINK_LENGTH 열 추가
# Window 함수를 사용하여 누적 합 계산
window_spec = Window.partitionBy("DATE", "OBU_ID", "TRIP_ID").orderBy("DATETIME")
result_df = (
    filtered_df.withColumn("TRIP_ID", concat(col("OBU_ID"), lit("_"), col("GROUP_ID")))
    .withColumn("DATE", date_format(col("DATETIME"), "yyyy-MM-dd")) 
    .withColumn("DATETIME", date_format(col("DATETIME"), "HH:mm")) 
    .withColumn(
        "DRIVING_TIME_MINUTES", 
        round(col("DRIVING_TIME") / 60, 1)
    )
    .withColumn(
        "CUMULATIVE_DRIVING_TIME_MINUTES",
        round(sum("DRIVING_TIME_MINUTES").over(window_spec), 1)
    )
    .withColumn(
        "CUMULATIVE_LINK_LENGTH",
        round(sum("LINK_LENGTH").over(window_spec), 1) 
    )
    .select(
        "DATE",
        "DATETIME",
        "OBU_ID",
        "TRIP_ID",
        "LINK_ID",
        round(col("LINK_LENGTH") / 1000, 1).alias("LINK_LENGTH"), 
        "DRIVING_TIME_MINUTES",
        "CUMULATIVE_DRIVING_TIME_MINUTES",
        round(col("CUMULATIVE_LINK_LENGTH") / 1000, 1).alias("CUMULATIVE_LINK_LENGTH"),
        "EMD_CODE"
    )
)

# CSV 파일에서 행정구역 데이터 읽어오기
csv_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\resource\LINK_ID_DATASET.csv"

# CSV 파일 스키마 정의
admin_schema = StructType([
    StructField("sido_id", IntegerType(), True),
    StructField("sigungu_id", IntegerType(), True),
    StructField("emd_id", IntegerType(), True),
    StructField("sido_name", StringType(), True),
    StructField("sigungu_name", StringType(), True),
    StructField("emd_name", StringType(), True)
])

admin_df = spark.read.csv(csv_path, header=True, schema=admin_schema)

#  SIGUNGU_ID 앞 4자리만 사용 (StringType으로 변환 후 처리)
admin_df = admin_df.withColumn("SIGUNGU_ID", substring(col("sigungu_id").cast("string"), 1, 4))

# AREA_ID_DATASET.csv 파일 읽어오기
area_csv_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\resource\AREA_ID_DATASET.csv"

# AREA_ID_DATASET.csv 파일 스키마 정의
area_schema = StructType([
    StructField("AREA_ID", IntegerType(), True),
    StructField("SIGUNGU_ID", StringType(), True),  # 4자리 문자열로 처리
    StructField("SIGUNGU_NAME", StringType(), True) 
])

area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)

# admin_df와 area_df 조인
joined_df = admin_df.join(area_df, admin_df["SIGUNGU_ID"] == area_df["SIGUNGU_ID"], "left")

# EMD_CODE를 기준으로 result_df와 joined_df 조인 (emd_id와 EMD_CODE를 일치시켜 조인)
final_df = result_df.join(joined_df, result_df["EMD_CODE"] == joined_df["emd_id"], "left")

# 필요한 열 선택 (SIGUNGU_ID, AREA_ID 포함)
final_df = final_df.select(
    "DATE",
    "DATETIME",
    "OBU_ID",
    "TRIP_ID",
    "LINK_ID",
    "LINK_LENGTH",
    "DRIVING_TIME_MINUTES",
    "CUMULATIVE_DRIVING_TIME_MINUTES",
    "CUMULATIVE_LINK_LENGTH",
    admin_df["SIGUNGU_ID"],  # admin_df의 SIGUNGU_ID를 명시적으로 지정
    "AREA_ID"
)

# Window 함수를 사용하여 그룹별 마지막 행의 CUMULATIVE_DRIVING_TIME_MINUTES 확인
window_spec = Window.partitionBy("DATE", "OBU_ID", "TRIP_ID").orderBy(desc("DATETIME"))
filtered_df = (
    final_df.withColumn("last_cumulative_driving_time", last("CUMULATIVE_DRIVING_TIME_MINUTES").over(window_spec))
    .filter(col("last_cumulative_driving_time") >= 15)
    .groupBy("DATE", "OBU_ID", "TRIP_ID")
    .agg(count("*").alias("count"))
    .filter(col("count") >= 10)
    .select("DATE", "OBU_ID", "TRIP_ID")
)

# 필터링된 TRIP_ID를 기반으로 final_df에서 데이터 필터링
filtered_final_df = final_df.join(filtered_df, ["DATE", "OBU_ID", "TRIP_ID"])

# 6. 결과 정렬
sorted_df = filtered_final_df.orderBy("DATE", "OBU_ID", "TRIP_ID","DATETIME")

# 7. 결과 저장 (날짜별 폴더 생성 및 저장)
base_output_path = "C:/Users/wngud/Desktop/project/heavy_duty_truck_charging_infra/data analysis/analyzed_paths_for_simulator"
sorted_df.write.partitionBy("DATE").csv(base_output_path, header=True, mode="overwrite")

# 8. SparkSession 종료
spark.stop()