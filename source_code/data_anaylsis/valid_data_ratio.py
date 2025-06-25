# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, IntegerType

def calculate_obu_stats():
    """
    1. 일별로 '고유 OBU_ID'의 개수를 계산합니다.
    2. 계산된 '일별 개수'들을 모두 더하여 전체 기간의 통계 및 최종 비율을 계산합니다.
    3. 일별 결과를 CSV 파일로 저장합니다.
    """
    # 1. SparkSession 생성
    spark = (
        SparkSession.builder.appName("CalculateOBUStats")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    print("SparkSession 생성 완료.")

    # 2. 파일 경로 정의
    raw_csv_base_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\Trajectory"
    processed_parquet_base_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_90km)"
    
    raw_schema_minimal = StructType([
        StructField("OBU_ID", IntegerType(), True)
    ])

    try:
        # 3. 처리된 데이터 날짜 목록 가져오기
        processed_dates = [d for d in os.listdir(processed_parquet_base_path) 
                           if os.path.isdir(os.path.join(processed_parquet_base_path, d))]
        processed_dates.sort()
        if not processed_dates:
            print("오류: 처리된 데이터가 없습니다.")
            spark.stop()
            return
        print(f"총 {len(processed_dates)}개의 처리된 날짜를 발견했습니다.")

        daily_results = []
        
        # 4. 일별 '고유 OBU_ID' 개수 계산
        print("\n--- 일별 고유 OBU ID 개수 계산 시작 ---")
        for date_str in processed_dates:
            yyyymmdd = date_str.replace('-', '')
            
            raw_file_pattern = os.path.join(raw_csv_base_path, f"*_{yyyymmdd}.csv")
            found_files = glob.glob(raw_file_pattern)
            if not found_files:
                raw_file_pattern = os.path.join(raw_csv_base_path, f"*{yyyymmdd}.csv")
                found_files = glob.glob(raw_file_pattern)
            
            if not found_files:
                print(f"[{date_str}] 건너뛰기: 원본 CSV 파일을 찾을 수 없습니다.")
                continue
            
            raw_csv_path = found_files[0]
            processed_parquet_path = os.path.join(processed_parquet_base_path, date_str)

            try:
                # 각 파일의 '고유 OBU_ID' 개수 계산
                df_raw = spark.read.csv(raw_csv_path, header=True, schema=raw_schema_minimal).na.drop(subset=["OBU_ID"])
                raw_obu_count = df_raw.select("OBU_ID").distinct().count()

                df_processed = spark.read.parquet(processed_parquet_path)
                processed_obu_count = df_processed.select("OBU_ID").distinct().count()

                retention_rate = (processed_obu_count / raw_obu_count) * 100 if raw_obu_count > 0 else 0
                
                daily_results.append({
                    "date": date_str,
                    "raw_count": raw_obu_count,
                    "processed_count": processed_obu_count,
                    "retention_rate": retention_rate
                })
                
                print(f"[{date_str}] 원본 OBU 수: {raw_obu_count}, 처리 후 OBU 수: {processed_obu_count}, 유지율: {retention_rate:.2f}%")

            except Exception as e:
                print(f"[{date_str}] 처리 중 오류 발생: {e}")
                continue
        
        print("--- 일별 계산 완료 ---")

        # 5. 일별 결과를 CSV 파일로 저장
        if daily_results:
            print("\n--- 일별 분석 결과 CSV 파일로 저장 시작 ---")
            df_results = pd.DataFrame(daily_results)
            df_results = df_results[['date', 'raw_count', 'processed_count', 'retention_rate']]
            
            output_dir = r"D:\연구실\연구\화물차 충전소 배치 최적화\Code\SEM_lab-project\resource_data\vaild_path_ratio"
            file_name = "vaild_path_ratio.csv"
            full_path = os.path.join(output_dir, file_name)
            
            os.makedirs(output_dir, exist_ok=True)
            df_results.to_csv(full_path, index=False, encoding='utf-8-sig')
            print(f"일별 결과가 다음 위치에 저장되었습니다:\n{full_path}")

        # 6. ★★★★★ 최종 계산 방식 수정 ★★★★★
        # 전체 파일을 다시 읽는 대신, '일별 계산 결과'를 모두 더하여 최종 통계 계산
        if not daily_results:
            print("\n전체 비율을 계산할 데이터가 없습니다.")
        else:
            print("\n--- 전체 기간 통계 계산 시작 ---")
            
            # 일별 '고유 OBU 수'의 총합 계산
            total_raw_count_sum = sum(item['raw_count'] for item in daily_results)
            total_processed_count_sum = sum(item['processed_count'] for item in daily_results)
            
            # 합산된 결과를 바탕으로 최종 비율 계산
            final_ratio = (total_processed_count_sum / total_raw_count_sum) * 100 if total_raw_count_sum > 0 else 0
            
            print("\n--- 최종 결과 요약 ---")
            print(f"분석 대상 기간: {daily_results[0]['date']} ~ {daily_results[-1]['date']}")
            print(f"일별 '원본 고유 OBU'의 총합: {total_raw_count_sum}")
            print(f"일별 '처리 후 고유 OBU'의 총합: {total_processed_count_sum}")
            print(f"최종 비율: {final_ratio:.2f}%")

    except Exception as e:
        print(f"스크립트 실행 중 예외 발생: {e}")
    finally:
        spark.stop()
        print("\nSparkSession 종료됨.")

# 메인 함수 실행
if __name__ == "__main__":
    calculate_obu_stats()