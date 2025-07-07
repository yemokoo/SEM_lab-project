# -*- coding: utf-8 -*-
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, regexp_extract, count, avg
from pyspark.sql.types import StructType, StructField, IntegerType

def calculate_daily_activity_probability_from_csv():
    """
    1년간의 Raw CSV 데이터를 기반으로 각 차량(OBU_ID)이 하루에 운행할 평균 확률을 계산합니다.

    로직:
    1. 지정된 경로의 모든 CSV 파일(..._YYYYMMDD.csv)을 한 번에 읽어들입니다.
    2. 파일 이름에서 날짜 정보(YYYYMMDD)를 추출하여 'date' 컬럼을 생성합니다.
    3. 데이터가 존재하는 총 고유 날짜 수(전체 운행일수)를 계산합니다.
    4. 각 OBU_ID별로 고유한 운행일수를 집계합니다.
    5. (OBU_ID별 운행일수 / 총 운행일수) 를 통해 OBU_ID별 '일일 운행 확률'을 계산합니다.
    6. 모든 OBU_ID의 '일일 운행 확률'의 평균을 계산하여 최종 결과를 도출합니다.
    """
    spark = (
        SparkSession.builder.appName("CalculateOBUActivityProbabilityFromCSV")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    raw_data_base_path = r"D:/연구실/연구/화물차 충전소 배치 최적화/Data/Raw_Data/Trajectory"
    
    # 분석에 OBU_ID만 필요하므로, 메모리 효율성을 위해 최소한의 스키마만 정의
    minimal_schema = StructType([
        StructField("OBU_ID", IntegerType(), True)
    ])
    
    try:
        # Windows 로컬 경로에서 와일드카드(*) 문제를 해결하기 위해 file:/// 스키마를 명시
        load_path = f"file:///{raw_data_base_path}/*.csv"
        
        df = spark.read.csv(
            load_path,
            header=True,
            schema=minimal_schema
        ).na.drop(subset=["OBU_ID"])

        # 파일명에서 날짜(YYYYMMDD)를 추출하여 'date' 컬럼으로 추가
        # 예: '.../AUTO_P1_TRUCK_SERO_20200101.csv' -> '20200101'
        df_with_date = df.withColumn("date", regexp_extract(input_file_name(), r'_(\d{8})\.csv', 1))
        
        distinct_obu_date = df_with_date.select("OBU_ID", "date").distinct()
        distinct_obu_date.cache()
        
        total_unique_days = distinct_obu_date.select("date").distinct().count()
        if total_unique_days == 0:
            print("오류: 분석할 데이터가 없습니다. 파일 경로나 파일명을 확인해주세요.")
            return

        active_days_per_obu = distinct_obu_date.groupBy("OBU_ID").agg(
            count("date").alias("active_days")
        )

        obu_probabilities_df = active_days_per_obu.withColumn(
            "activity_probability", col("active_days") / total_unique_days
        )
        
        print("\n[샘플] OBU_ID별 운행일수 및 운행 확률:")
        obu_probabilities_df.show(5)

        final_avg_probability_row = obu_probabilities_df.agg(
            avg("activity_probability").alias("avg_probability")
        ).first()
        final_avg_probability = final_avg_probability_row['avg_probability'] if final_avg_probability_row else 0
        
        print("\n--- 최종 결과 요약 ---")
        print(f"분석 대상 총 고유 OBU_ID 수: {obu_probabilities_df.count()}")
        print(f"분석 기간(고유 일수): {total_unique_days}일")
        print(f"📊 차량 당 평균 일일 운행 확률: {final_avg_probability:.4f} ({final_avg_probability:.2%})")

        output_dir = r"D:/연구실/연구/화물차 충전소 배치 최적화/Code/SEM_lab-project/resource_data/obu_activity_analysis"
        file_name = "obu_daily_activity_probability_from_raw.csv"
        full_path = f"{output_dir}/{file_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        pandas_df_results = obu_probabilities_df.toPandas()
        pandas_df_results.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f"\nOBU_ID별 상세 결과가 다음 위치에 저장되었습니다:\n{full_path}")

    except Exception as e:
        print(f"\n스크립트 실행 중 예외 발생: {e}")
    finally:
        spark.stop()
        print("\n⏹️ SparkSession 종료됨.")

if __name__ == "__main__":
    calculate_daily_activity_probability_from_csv()
