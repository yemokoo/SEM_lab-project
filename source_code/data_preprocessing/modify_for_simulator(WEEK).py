import glob
import re
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os

# 1. SparkSession 생성 (기존 코드 사용)
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

# 2. CSV 파일 스키마 정의 (기존 코드 사용)
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

# 3. 파일 목록 가져오기, 정렬 및 처리 함수 정의
def process_csv_files(file_path, schema, batch_size=7):
    """
    CSV 파일 목록을 날짜순으로 정렬하고, batch_size 개수만큼 묶어서 Spark로 처리합니다.
    각 주차별 데이터를 순차적으로 저장합니다.

    Args:
        file_path (str): CSV 파일이 있는 디렉토리 경로 (와일드카드 사용 가능).
        schema (StructType): CSV 파일의 스키마.
        batch_size (int): 한 번에 처리할 파일 개수.
    """

    # 3-1. 파일 목록 가져오기
    file_list = glob.glob(file_path)

    # 3-2. 파일 이름에서 날짜 추출 및 정렬을 위한 함수
    def get_date_from_filename(filename):
        """
        파일 이름에서 AUTO_P1_TRUCK_SERO_YYYYMMDD 형식의 날짜를 추출합니다.
        예: AUTO_P1_TRUCK_SERO_20200101.csv -> 20200101
        """
        match = re.search(r"AUTO_P1_TRUCK_SERO_(\d{8})", filename)  # 파일 이름에서 8자리 숫자(날짜) 추출
        if match:
            return int(match.group(1))
        else:
            return 0  # 날짜를 찾을 수 없는 경우, 정렬 순서에서 뒤로 밀림

    # 3-3. 날짜 기준으로 파일 목록 정렬
    file_list.sort(key=get_date_from_filename)

    # 3-4. batch_size 개수만큼 파일을 묶어서 처리 (마지막 주차 처리 포함)
    week_num = 1  # 주차 번호 초기화
    for i in range(0, len(file_list), batch_size):
        if i + batch_size < len(file_list):
            batch_files = file_list[i:i + batch_size]
        else:
            batch_files = file_list[i:] # 마지막 주차 파일 개수가 batch_size보다 작을 수 있도록 수정

        # 3-5. Spark로 파일 읽기, 데이터 처리
        try:
            df = (
                spark.read.csv(batch_files, header=True, schema=schema, sep=",")
                .na.fill(0, subset=["OBU_ID"])
                .drop("VEH_TYPE", "SEQ", "LINK_SPEED", "POINT_LENGTH", "POINT_SPEED", "DAS_FAKE_TYPE", "REAL_TYPE")
                .withColumn("LINK_ID", col("LINK_ID").cast(IntegerType()))
                .withColumn("IN_TIME", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("IN_TIME")), "yyyy-MM-dd HHmmss"))
                .withColumn("OUT_TIME", to_timestamp(concat(date_format(col("DATETIME"), "yyyy-MM-dd"), lit(" "), col("OUT_TIME")), "yyyy-MM-dd HHmmss"))
                .na.drop(subset=["IN_TIME", "OUT_TIME"])
                .withColumn("DRIVING_TIME", (unix_timestamp("OUT_TIME") - unix_timestamp("IN_TIME")).cast("double"))
                .filter((col("LINK_ID") != -1) & (col("DRIVING_TIME") >= 1))
                .drop("IN_TIME", "OUT_TIME")
                .withColumn("DRIVING_TIME_MINUTES", round(col("DRIVING_TIME") / 60, 1))
                .drop("DRIVING_TIME")
                .withColumn("LINK_LENGTH", round(col("LINK_LENGTH") / 1000, 1))
            )
            print(f"Processing files: {', '.join(batch_files)}")

            # 3-6. 데이터 집계 및 변환
            df = df.withColumn("TRIP_ID", concat(col("OBU_ID"), lit("_"), col("GROUP_ID")))
            df = df.withColumn("DATE", date_format(col("DATETIME"), "yyyy-MM-dd"))
            window_spec = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy("DATE", "DATETIME")
            min_date = df.select(min("DATE")).first()[0]
            df = (
                df
                .withColumn(
                    "CUMULATIVE_DRIVING_TIME_MINUTES",
                    round(sum("DRIVING_TIME_MINUTES").over(window_spec), 1)
                )
                .withColumn(
                    "CUMULATIVE_LINK_LENGTH",
                    round(sum("LINK_LENGTH").over(window_spec), 1)
                )
                .withColumn("DAY_DIFF", datediff(col("DATE"), lit(min_date)))
                .withColumn("TOTAL_MINUTES", col("DAY_DIFF") * 1440 + hour(col("DATETIME")) * 60 + minute(col("DATETIME")))
                .withColumn("COMBINED_DATETIME", concat(floor(col("TOTAL_MINUTES") / 60), lit(":"), lpad(col("TOTAL_MINUTES") % 60, 2, "0")))
                .drop("DAY_DIFF", "TOTAL_MINUTES")
                .withColumn("DATETIME", date_format(col("DATETIME"), "HH:mm"))
            )

            # 3-7. 행정구역 및 지역 ID 데이터 조인
            admin_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\resource\광역자치단체 지도\LINK_ID_DATASET.csv"
            admin_schema = StructType([
                StructField("sido_id", IntegerType(), True),
                StructField("sigungu_id", IntegerType(), True),
                StructField("emd_id", IntegerType(), True),
                StructField("sido_name", StringType(), True),
                StructField("sigungu_name", StringType(), True),
                StructField("emd_name", StringType(), True)
            ])
            admin_df = spark.read.csv(admin_path, header=True, schema=admin_schema)
            admin_df = admin_df.withColumn("SIGUNGU_ID", substring(col("sigungu_id").cast("string"), 1, 4))

            area_csv_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\resource\도시 권역\AREA_ID_DATASET.csv"
            area_schema = StructType([
                StructField("AREA_ID", IntegerType(), True),
                StructField("SIGUNGU_ID", StringType(), True),
                StructField("SIGUNGU_NAME", StringType(), True)
            ])
            area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)

            joined_df = admin_df.join(area_df, admin_df["SIGUNGU_ID"] == area_df["SIGUNGU_ID"], "left")
            df = df.join(joined_df, df["EMD_CODE"] == joined_df["emd_id"], "left")

            # 3-8. 데이터 필터링, 열 이름 변경 및 선택
            # 윈도우 함수 적용
            window_spec = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(desc("DATETIME"))
            
            # 조건에 맞는 TRIP_ID 필터링 및 필요한 열 선택
            df = (
                df.withColumn("last_cumulative_driving_time", last("CUMULATIVE_DRIVING_TIME_MINUTES").over(window_spec))
                .filter(col("last_cumulative_driving_time") > 3) # 각 그룹의 마지막 CUMULATIVE_DRIVING_TIME_MINUTES 값이 3min 이상
                .drop("last_cumulative_driving_time")  # 임시 열 제거
                .withColumn("last_cumulative_link_length", last("CUMULATIVE_LINK_LENGTH").over(window_spec))
                .filter(col("last_cumulative_link_length") > 1) # 각 그룹의 마지막 CUMULATIVE_LINK_LENGTH 값이 1km 이상
                .drop("last_cumulative_link_length") # 임시 열 제거
                .groupBy("OBU_ID", "TRIP_ID")
                .agg(count("*").alias("count"))
                #.filter(col("count") >= 10) # 각 그룹의 행 개수가 10개 이상
                .drop("count") # 임시 열 제거
                .join(df, ["OBU_ID", "TRIP_ID"], "inner")  # 필터링된 TRIP_ID를 기준으로 조인
                .drop("DATE", "DATETIME", "EMD_CODE", "emd_id", "sido_id", "sido_name", "sigungu_name", "emd_name")
                .withColumnRenamed("COMBINED_DATETIME", "DATETIME")
                .select("OBU_ID", "TRIP_ID", "DATETIME", "LINK_ID", "LINK_LENGTH", "DRIVING_TIME_MINUTES", "CUMULATIVE_DRIVING_TIME_MINUTES", "CUMULATIVE_LINK_LENGTH", admin_df["sigungu_id"], "AREA_ID")
            )

            df.orderBy("OBU_ID", "TRIP_ID", "DATETIME")

            # 3-9. 파일 내용 처리 및 저장 (파티션 개수를 10개로 줄여서 저장)
            output_path = f"C:/Users/wngud/Desktop/project/heavy_duty_truck_charging_infra/data analysis/analyzed_paths_for_simulator(WEEK)/{week_num}_weeks"
            df.coalesce(10).write.csv(output_path, header=True, mode="overwrite")
            print(f"Saved {week_num}_weeks data to {output_path}")

            week_num += 1  # 주차 번호 증가

        except Exception as e:
            print(f"Error processing files: {', '.join(batch_files)}")
            print(e)

# 4. 함수 호출
file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\resource\트럭 이동 경로\*.csv"
process_csv_files(file_path, schema, batch_size=7)