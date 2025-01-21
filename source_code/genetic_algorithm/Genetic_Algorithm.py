from logging.config import valid_ident
from math import floor
import os
import select
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import Simulator_for_day as si
from multiprocessing import Pool, cpu_count
import sys
import traceback
import logging
from collections import Counter
path_for_car = r"C:\Users\yemoy\SEM_화물차충전소\drive-download-20241212T004036Z-001"
path_for_car_oneday =r"C:\Users\yemoy\SEM_화물차충전소\경로폴더"
path_for_station = r"C:\Users\yemoy\SEM_화물차충전소\station_for_simulator.csv"
random.seed(42)

#100개의 솔루션 -> 토너먼트로 25개 선정 -> 25개중 상위 4개는 엘리티즘 -> 1등을 제외한 24개에 대하여 교차를 통해 96개의 해 생성  -> 100개의 다음세대 솔루션 생성.
#해당 사항으로 우선 알고리즘이 개발되어 일단은 한 세대당 100개의 솔루션이 있음을 가정하고 시뮬레이터에 적용하길 바랍니다.
# 유전 알고리즘 파라미터 설정
POPULATION_SIZE = 100  # 개체군 크기
GENERATIONS = 1000  # 최대 세대 수 (필요 시 무시됨)
TOURNAMENT_SIZE = 4 # 토너먼트 크기
MUTATION_RATE = 0.1  # 변이 확률
NUM_CANDIDATES = 239 #충전전소 위치 후보지 개수
NUM_SETS = 200  # 병렬로 계산할 세트 개수
CAR_PATHS_FOLDER = 'car_paths_folder_path'  # 차량 경로 데이터 폴더 경로
CONVERGENCE_THRESHOLD = 1e-6  # 적합도 수렴 기준
IMMIGRATION_THRESHOLD = 0.05  # 이민자 연산 실행 임계값
MAX_NO_IMPROVEMENT = 7  # 개선 없는 최대 세대 수
TOTAL_CHARGERS = 2000 #설치할 충전기의 대수
PARENTS_SIZE = round(POPULATION_SIZE/4) #부모의 수

duplicate_counts = []  # 중복 개체 수 저장용 리스트

def station_gene_initial(pop_size,num_candi,total_chargers):
    """초기 유전자 세트를 생성하는 함수."""
    population = []
    
    #일반 세대 생성
    for _ in range(pop_size):
        intial_station = np.random.multinomial(total_chargers,[1/num_candi]*num_candi)
        population.append(intial_station)
    
    return population


def evaluate_individual(args):
    """
    단일 개체에 대한 적합도를 계산하는 함수입니다.
    """
    
    individual, index, car_paths_df, station_df = args
    print(f"Evaluating individual {index}")
        
        #아래는 추후 df 길이 문제가 생기면 사용용
        #print(f"station_df 길이: {len(station_df)}")
        
        # 시뮬레이션 파라미터
    unit_minutes = 60
    simulating_hours = 12
    num_trucks = 50

        # Ensure the length of 'individual' matches the number of rows in station_df
    if len(individual) != len(station_df):
        raise ValueError(f"The length of 'individual' ({len(individual)}) does not match the number of rows in station_df ({len(station_df)}).")        
    station_df['num_of_charger'] = individual   
        # 시뮬레이션 실행 및 적합도 값 획득
    fitness_value = si.run_simulation(
                car_paths_df,
                station_df,
                unit_minutes,
                simulating_hours,
                num_trucks,
                TOTAL_CHARGERS
            )
    print(f"fitness_value for individual {index}: {fitness_value}")
    return (index, fitness_value)

   


def fitness_func(population, station_df):
    """시뮬레이션에서 리턴되는 값들을 통해 각 솔루션의 적합도를 평가하는 함수."""
    car_paths_folder = path_for_car  
    car_paths_df = si.load_car_path_df(car_paths_folder, number_of_trucks=5000)
    print("차량 경로 파일 길이", len(car_paths_df))

    args_list = [
        (individual, idx, car_paths_df, station_df)
        for idx, individual in enumerate(population)
    ]

    max_retries = 3  # 최대 재시도 횟수
    retry_count = 0
    print("cpu 코어 개수 : ",cpu_count())
    while retry_count < max_retries:
        try:
            results = []
            with Pool(processes=cpu_count()-14) as pool:
                # imap을 사용하여 작업 순서대로 결과 처리
                for i, result in enumerate(
                    pool.imap(evaluate_individual, args_list), 1
                ):
                    results.append(result) # result = (index, fitness_value)
                    # 진행 상황 출력
                    if i % 10 == 0 or i == len(args_list):
                        sys.stdout.write(f"\rProcessed {i}/{len(args_list)} items")
                        sys.stdout.flush()

                # 결과 정렬 (인덱스 기준)
                results.sort(key=lambda x: x[1], reverse=True) # 적합도 값이 큰 순서대로 정렬
                sorted_indices = [x[0] for x in sorted(results, key=lambda x: x[1], reverse=True)]
                sorted_population = [population[i] for i in sorted_indices]
                print("\n모든 작업이 완료되었습니다.")
                # 적합도 값 추출 및 반환
                fitness_values = [
                    fitness if fitness is not None else -np.inf for _, fitness in results
                ]
                return fitness_values, sorted_population

        except Exception as e:
            retry_count += 1
            logging.warning(
                f"An error occurred: {e}. Retry attempt {retry_count}/{max_retries}."
            )
            print("An error occurred: {e}. Retry attempt {retry_count}/{max_retries}.")
            time.sleep(5)  # 재시도 전 잠시 대기

    logging.error("Max retries reached. Exiting.")
    return [0] * len(population)  # 최대 재시도 횟수 초과 시 모든 적합도를 0로 설정





def choice_gene_tournament_no_duplicate(population, tournament_size, num_parents, fitness_values):
    """
    토너먼트 선택으로 부모를 중복 없이 선택하는 함수.
    남은 set의 수가 tournament_size보다 작다면 남은 개체만 가지고 진행한다.
    """
    if len(population) <= 1:
        raise ValueError("Value Error: Population size must be greater than 1")
    if tournament_size > len(population):
        raise ValueError("Value Error: Tournament size cannot be greater than population size")

    selected_parents = []
    # 아직 선택되지 않은 개체의 인덱스를 관리
    remain_indices = list(range(len(population)))

    while len(selected_parents) < num_parents and len(remain_indices) > 0:
        # 남은 개체 수가 tournament_size보다 작다면 해당 개체 수만 사용
        current_t_size = min(tournament_size, len(remain_indices))
        # 토너먼트 참가자 후보 인덱스 선택
        contestants_indices = random.sample(remain_indices, k=current_t_size)
        # fitness 값이 가장 큰 인덱스 선택
        best_index = max(contestants_indices, key=lambda i: fitness_values[i])
        # 선택된 부모를 결과 리스트에 추가
        selected_parents.append(population[best_index])
        # 이미 선택된 인덱스는 다음에는 선택되지 않도록 제거
        #for contestant in contestants_indices:
            #remain_indices.remove(contestant)
        remain_indices.remove(best_index)

    return selected_parents


def crossover_elitsm(selected_parents, num_genes, pop_size):
    """다중지점 교차를 통해 새로운 세대의 개체를 생성하는 함수."""
    crossover = []
    #elitism = selected_parents[:4]  # 상위 1개 엘리트
    #crossover.extend([parent.copy() for parent in elitism])
    # 나머지 자손 생성
    while len(crossover) < pop_size:
        # 부모 4명 선택 (예: 랜덤 선택)
        parents = random.sample(selected_parents, 6)
        
        crossover_points =sorted(random.sample(range(1, num_genes), 5))

        child = np.concatenate([
            parents[0][:crossover_points[0]],
            parents[1][crossover_points[0]:crossover_points[1]],
            parents[2][crossover_points[1]:crossover_points[2]],
            parents[3][crossover_points[2]:crossover_points[3]],
            parents[4][crossover_points[3]:crossover_points[4]],
            parents[5][crossover_points[4]:]])
        crossover.append(child)
    
    counter = Counter(tuple(ind) for ind in crossover)
    
    # 1) 총 중복 개체 수 (동일한 해가 여러 번 나온 횟수의 합)
    duplicates_count = sum((count - 1) for count in counter.values() if count > 1)
    if duplicates_count > 0:
        print(f"[중복 개체 확인] 이번 세대에서 총 {duplicates_count}개의 중복 개체가 발견되었습니다.")
        duplicate_counts.append(duplicates_count)
    else:
        print("[중복 개체 확인] 이번 세대에서는 중복된 개체가 없습니다.")
        duplicate_counts.append(0)
    
    # 2) 어떤 개체가 몇 번 등장했는지, 상세 목록을 보고 싶다면:
    for solution, cnt in counter.items():
         if cnt > 1:
             print(f"개체 {solution} 이(가) {cnt}회 등장")

    return crossover[:pop_size]

    

def mutation(crossovered,pop_size,mutation_rate,num_candi,total_chargers,adaptive_constant):
    """적응형 변이율을 적용하는 함수."""
    for i in range(pop_size):
        if random.random()<=mutation_rate + adaptive_constant:
            crossovered[i] = np.random.multinomial(total_chargers,[1/num_candi]*num_candi)
            #adaptive_constant -= 0.0001 
           #print(f"변이가 적용되었습니다. 변이 횟수 : {-adaptive_constant*10000} ")
    return crossovered

def immigration(population, num_candi, total_chargers):
    """이민자 연산을 통해 개체군 다양성을 유지하는 함수.
    하위 20%의 개체를 무작위로 교체한다."""
    num_to_replace = len(population) // 5  # 하위 20% 치환
    for i in range(len(population) - num_to_replace, len(population)):
        population[i] = np.random.multinomial(total_chargers, [1/num_candi] * num_candi)
    return population




def genetic_algorithm():
    no_improvement_count = 0
    best_fitness = float('-inf')
    fitness_history = []
    immigration_count = 0
    MAX_IMMIGRATIONS = 5

    
    # 각 세대의 전체 적합도 값 저장용 리스트
    all_fitness_history = []
    min_fitness_history = []
    max_fitness_history = []
    mean_fitness_history = []
    station_file_path = path_for_station
    #station_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\station_for_simulator(debug).csv"
    station_df = si.load_station_df(station_file_path) 

    for generation in range(GENERATIONS):
        print(f"\n세대 {generation + 1}/{GENERATIONS} 진행 중...")

        if generation == 0:
            population = station_gene_initial(POPULATION_SIZE, NUM_CANDIDATES, TOTAL_CHARGERS)
        else:
            population = mutated
        if generation == GENERATIONS:
            print("지정된 세대의 연산이 종료되었으므로 계산 결과를 출력합니다")
            break

        # 적합도 계산
        fitness_values, sorted_population = fitness_func(population,station_df)
        print('적합도 평가 완료')
        
        # 전체 적합도 값 저장
        all_fitness_history.append(fitness_values)
        print('적합도 값 저장 완료')
        
        # 세대별 최소, 최대, 평균 적합도 저장
        current_min = np.min(fitness_values) #에러발생. >= 등의 부등호를 비교할 수 없는 문제가 발생했음.
        current_max = np.max(fitness_values)
        current_mean = np.mean(fitness_values)

        min_fitness_history.append(current_min)
        max_fitness_history.append(current_max)
        mean_fitness_history.append(current_mean)

        # 현재 세대의 최고 적합도
        current_best_fitness = current_max
        fitness_history.append(current_best_fitness) # 최고 적합도 저장

        print(f"세대 {generation + 1}의 최고 적합도: {current_best_fitness}")

        # 이민자 연산용 카운트 계산 : 여기서는 범위를 넓게 잡아서 범위에 들어오면 이민자 연산을 실행함 현세대 max vs 이전 까지의 max
        # 이민자 연산의 조건은 적합도 변화가 n% 이하인 경우가 5번 연속 발생하면 이민자 연산을 실행함
        if abs(current_best_fitness - best_fitness) < IMMIGRATION_THRESHOLD:
            convergence_count += 1
        else:
            convergence_count = 0

        # 수렴 체크 // 여기서는 범위를 좁게 잡아서 범위에 들어오면 알고리즘이 종료됨됨
        if abs(current_best_fitness - best_fitness) < CONVERGENCE_THRESHOLD:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            best_fitness = max(max_fitness_history) # 최고 적합도 갱신


        if no_improvement_count >= MAX_NO_IMPROVEMENT:
            print(f"개선이 {MAX_NO_IMPROVEMENT} 세대 동안 없었으므로 알고리즘을 종료합니다.")
            break
        
        # 이민자 연산
        if convergence_count == 4 and immigration_count < MAX_IMMIGRATIONS:
            sorted_population = immigration(sorted_population, NUM_CANDIDATES, TOTAL_CHARGERS)
            immigration_count += 1
            print(f"이민자 연산 실행 ({immigration_count}/{MAX_IMMIGRATIONS})")

        # 부모 선택
        parents = choice_gene_tournament_no_duplicate(sorted_population, TOURNAMENT_SIZE, PARENTS_SIZE ,fitness_values)
        print('부모 선택 완료')

        # 교차
        children = crossover_elitsm(parents, NUM_CANDIDATES,POPULATION_SIZE)
        print('교차 연산 완료')

        # 변이
        mutated = mutation(children, POPULATION_SIZE, MUTATION_RATE, NUM_CANDIDATES, TOTAL_CHARGERS, 0)
        print('변이 연산 완료')

        

    print("\n유전 알고리즘 종료")
    print(f"최종 세대 수: {generation + 1}")
    print(f"최고 적합도: {best_fitness}")

    # 중복 개체 수 저장
    duplicate_array = np.array(duplicate_counts)
    df = pd.DataFrame(duplicate_array, columns=["Duplicate Counts"])
    df.to_csv("duplicate_counts_2.csv", index=False)  
    print("중복 개체 수를 csv 파일로 저장")

    result_dict = {
        'Generation' : np.arange(1,len(min_fitness_history)+1),
        'Min_fitness' : min_fitness_history,
        'Max_fitness' : max_fitness_history,
        'Mean_fitness' : mean_fitness_history
    }
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("ga_results.csv",index=False)
    print("각 세대별 fitness value를 csv파일로 저장")
    # 그래프 출력
    gens = np.arange(1, len(fitness_history)+1)

    fig, ax = plt.subplots(2, 1, figsize=(20, 14), sharex=True)

    # 상단: 최고, 평균, 최소-최대 범위 표시
    ax[0].plot(gens, mean_fitness_history, label='Mean Fitness', color='blue', marker='o')
    ax[0].plot(gens, fitness_history, label='Best Fitness', color='red', marker='o')
    ax[0].fill_between(gens, min_fitness_history, max_fitness_history, color='gray', alpha=0.3, label='Min-Max Range')
    ax[0].set_ylabel('Fitness')
    ax[0].legend()
    ax[0].grid(True)

    # 하단: 박스플랏으로 세대별 전체 분포 표시
    ax[1].boxplot(all_fitness_history, positions=gens)
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Fitness Distribution ()')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 로그 파일 경로
    log_file_path = os.path.join(r"C:\Users\yemoy\SEM_화물차충전소\geneticAlgorithm", "genetic_algorithm.log")

    # 로깅 설정
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',filemode='w')
    genetic_algorithm()
