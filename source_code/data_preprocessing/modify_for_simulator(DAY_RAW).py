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
def process_csv_files(file_path, schema):
    """
    CSV 파일 목록을 날짜순으로 정렬하고, 하루 단위로 Spark로 처리합니다.
    각 날짜별 데이터를 순차적으로 저장합니다.

    Args:
        file_path (str): CSV 파일이 있는 디렉토리 경로 (와일드카드 사용 가능).
        schema (StructType): CSV 파일의 스키마.
    """

    # 3-1. 파일 목록 가져오기
    file_list = glob.glob(file_path)

    # 3-2. 파일 이름에서 날짜 추출 및 정렬을 위한 함수 (기존 함수 사용)
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

    # 3-3. 날짜 기준으로 파일 목록 정렬 (기존 코드 사용)
    file_list.sort(key=get_date_from_filename)

    # 3-4. 파일 하나씩 처리 (일 단위 처리)
    for file in file_list:
        date_int = get_date_from_filename(file)
        if date_int == 0: # 날짜 추출 실패 시 파일 건너뛰기
            print(f"Skipping file due to date extraction failure: {file}")
            continue

        date_str = str(date_int)
        output_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}" # YYYY-MM-DD 형식 날짜 문자열 생성
        batch_files = [file] # 이제 batch_files는 항상 파일 하나만 포함

        # 3-5. Spark로 파일 읽기, 데이터 처리 (기존 코드와 동일)
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
                .filter(col("LINK_ID") != -1)
                .drop("IN_TIME", "OUT_TIME")
                .withColumn("DRIVING_TIME_MINUTES", round(col("DRIVING_TIME") / 60, 1))
                .drop("DRIVING_TIME")
                .withColumn("LINK_LENGTH", round(col("LINK_LENGTH") / 1000, 1))
            )
            print(f"Processing file: {file}")

            # 3-6. 데이터 집계 및 변환 (기존 코드와 동일)
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

            # 3-7. 행정구역 및 지역 ID 데이터 조인 (기존 코드와 동일)
            admin_path = r"C:\Users\wngud\Desktop\project\HDT_EVCS_Opt\Data\Raw_Data\Metropolitan area\LINK_ID_DATASET.csv"
            admin_schema = StructType([
                StructField("sido_id", IntegerType(), True),
                StructField("sigungu_id", IntegerType(), True),
                StructField("emd_id", IntegerType(), True)
            ])
            admin_df = spark.read.csv(admin_path, header=True, schema=admin_schema)
            admin_df = admin_df.withColumn("SIGUNGU_ID", substring(col("sigungu_id").cast("string"), 1, 4))

            area_csv_path = r"C:\Users\wngud\Desktop\project\HDT_EVCS_Opt\Data\Raw_Data\Metropolitan area\AREA_ID_DATASET.csv"
            area_schema = StructType([
                StructField("AREA_ID", IntegerType(), True),
                StructField("SIGUNGU_ID", StringType(), True)
            ])
            area_df = spark.read.csv(area_csv_path, header=True, schema=area_schema)

            joined_df = admin_df.join(area_df, admin_df["SIGUNGU_ID"] == area_df["SIGUNGU_ID"], "left")
            df = df.join(joined_df, df["EMD_CODE"] == joined_df["emd_id"], "left")

            # 3-8. 데이터 필터링, 열 이름 변경 및 선택 (기존 코드와 동일)
            # 윈도우 함수 적용
            window_spec = Window.partitionBy("OBU_ID", "TRIP_ID").orderBy(desc("DATETIME"))

            # 조건에 맞는 TRIP_ID 필터링 및 필요한 열 선택
            df = (
                df.withColumn("last_cumulative_link_length", last("CUMULATIVE_LINK_LENGTH").over(window_spec))
                .drop("last_cumulative_link_length") # 임시 열 제거
                .groupBy("OBU_ID", "TRIP_ID")
                .join(df, ["OBU_ID", "TRIP_ID"], "inner")  # 필터링된 TRIP_ID를 기준으로 조인
                .drop("DATE", "DATETIME", "EMD_CODE", "emd_id", "sido_id")
                .withColumnRenamed("COMBINED_DATETIME", "DATETIME")
                .select("OBU_ID", "TRIP_ID", "DATETIME", "LINK_ID", "LINK_LENGTH", "DRIVING_TIME_MINUTES", "CUMULATIVE_DRIVING_TIME_MINUTES", "CUMULATIVE_LINK_LENGTH", admin_df["sigungu_id"], "AREA_ID")
            )

            df.orderBy("OBU_ID", "TRIP_ID", "DATETIME")

            # 3-9. 파일 내용 처리 및 저장 (파티션 개수를 10개로 줄여서 저장)
            output_path = f"C:\Users\wngud\Desktop\project\HDT_EVCS_Opt\Data\Processed_Data\simulator\Trajectory(DAY_RAW){output_date}" # output_date 사용
            df.coalesce(10).write.csv(output_path, header=True, mode="overwrite")
            print(f"Saved {output_date} data to {output_path}")


        except Exception as e:
            print(f"Error processing file: {file}")
            print(e)

# 4. 함수 호출
file_path = r"C:\Users\wngud\Desktop\project\HDT_EVCS_Opt\Data\Raw_Data\Trajectory*.csv"
process_csv_files(file_path, schema) 