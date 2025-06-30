#화물차 궤적 데이터 기반 시뮬레이션 개발 및 전기차 충전소 위치 및 규모 최적화
import glob
import os
import re
import traceback

# --- NumPy 및 관련 라이브러리 스레드 제한 설정 ---
# 중요: numpy, pandas 등을 import 하기 전에 실행되어야 합니다!
num_threads_to_set = '1'
os.environ['OMP_NUM_THREADS'] = num_threads_to_set
os.environ['MKL_NUM_THREADS'] = num_threads_to_set
os.environ['OPENBLAS_NUM_THREADS'] = num_threads_to_set
os.environ['NUMEXPR_NUM_THREADS'] = num_threads_to_set
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads_to_set
# ---------------------------------------------

import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import Simulator_for_day as si
from multiprocessing import Pool, cpu_count
import multiprocessing
import logging
from collections import Counter
from shared_memory_utils import put_df_to_shared_memory, reconstruct_df_from_shared_memory, cleanup_shms_by_info
import gc
import pprint

path_for_car = r"/home/semlab/SEM/EVCS/화물차 충전소 배치 최적화/Data/Processed_Data/simulator/Trajectory(DAY_90km)"
path_for_station = r"/home/semlab/SEM/EVCS/화물차 충전소 배치 최적화/Data/Processed_Data/simulator/Final_Candidates_Selected.csv"
path_for_result = r"/home/semlab/SEM/EVCS/화물차 충전소 배치 최적화/Data/Processed_Data/GA_result"
random.seed(42)
np.random.seed(42)

#100개의 솔루션 -> 토너먼트로 25개 선정 -> 25개중 상위 4개는 엘리티즘 -> 1등을 제외한 24개에 대하여 교차를 통해 96개의 해 생성  -> 100개의 다음세대 솔루션 생성.
#해당 사항으로 우선 알고리즘이 개발되어 일단은 한 세대당 100개의 솔루션이 있음을 가정하고 시뮬레이터에 적용하길 바랍니다.

# 유전 알고리즘 파라미터 설정
CORE_NUM = 150  # 코어 수
POPULATION_SIZE = 150  # 개체군 크기
GENERATIONS = 10000  # 최대 세대 수 (필요 시 무시됨)
TOURNAMENT_SIZE = 4 # 토너먼트 크기
MUTATION_RATE = 0.015  # 변이 확률(기본 0.015)
MUTATION_GENES_MULTIPLE = 20  # 중복된 해에 들어간 유전자 정보의 변이 배수
NUM_CANDIDATES = 500 # 충전소 위치 후보지 개수
CONVERGENCE_CHECK_START_GENERATIONS = 800  # 수렴 체크 시작 세대
MAX_NO_IMPROVEMENT = 10  # 개선 없는 최대 세대 수
ELECTRIFICATION_RATE = 1.0 # 전동화율 가정(원본이 10%임)
TRUCK_NUMBERS = int(5946 * ELECTRIFICATION_RATE) # 전체 화물차 대수 / 5946대는 10%의 전동화율 기준 대수
INITIAL_CHARGERS = TRUCK_NUMBERS # 설치할 충전기의 대수 충전기 
TOTAL_CHARGERS = 10000 # 총 충전기 대수
PARENTS_SIZE = round(POPULATION_SIZE/2) # 부모의 수



# 저장 세대 간격
SAVE_INTERVAL = 100
# 중복 정보를 저장할 DataFrame 초기화
duplicate_info_df = pd.DataFrame(columns=['Generation', 'Solution', 'Indices', 'Count'])

def station_gene_initial(pop_size,num_candi,total_chargers):
    """초기 유전자 세트를 생성하는 함수."""
    population = []
    
    #일반 세대 생성
    for _ in range(pop_size):
        intial_station = np.random.multinomial(total_chargers,[1/num_candi]*num_candi)
        population.append(intial_station)
    
    return population


def evaluate_individual_shared(args):
    global worker_original_station_df 

    individual, index, shm_infos_car_paths = args

    unit_minutes = 20   
    simulating_hours = 36  
    truck_step_freqency = 3  # 트럭의 시간 단계 주기 
    num_trucks = TRUCK_NUMBERS # GA.py의 전역변수
    
    #try:
        # 1. 공유 메모리에서 car_paths_df 재구성 및 복사
    car_paths_df_copy = reconstruct_df_from_shared_memory(shm_infos_car_paths)

    station_df_for_sim = worker_original_station_df.copy()
    station_df_for_sim['num_of_charger'] = individual # 유전자 적용

    fitness_value = si.run_simulation( 
        car_paths_df_copy,
        station_df_for_sim,
        unit_minutes,
        simulating_hours,
        num_trucks,
        TOTAL_CHARGERS, # GA.py의 전역변수
        truck_step_freqency
    )

    # 사용한 DataFrame 명시적 삭제
    del car_paths_df_copy
    del station_df_for_sim
    gc.collect()

    return (index, fitness_value)


worker_original_station_df = None

def init_worker_station(original_station_df_data):
    global worker_original_station_df
    # print(f"Worker {os.getpid()} initializing with station_df...")
    worker_original_station_df = original_station_df_data
    # print(f"Worker {os.getpid()} station_df initialized.")

def fitness_func(population, original_station_df_ref_for_fitness_func, path_history, pool, current_generation_number):
    """
    시뮬레이션에서 리턴되는 값들을 통해 각 솔루션의 적합도를 평가하는 함수.
    공유 메모리 사용 및 세대별 car_paths_df 로딩 반영.
    original_station_df_ref_for_fitness_func: 메인 프로세스의 원본 station_df 참조 (워커는 initializer로 받음)
    current_generation_number: 공유 메모리 이름 고유성 확보를 위한 세대 번호
    """
    car_paths_folder = path_for_car  # 전역 변수 사용

    load_start_time = time.time()
    # 매 세대마다 car_paths_df 새로 로드 (사용자 요구사항)
    car_paths_df_current_gen = si.load_car_path_df(car_paths_folder, TRUCK_NUMBERS)
    load_end_time = time.time()
    print(f"  - 데이터 로딩 시간: {load_end_time - load_start_time:.2f}초")
    
    if car_paths_df_current_gen is None or car_paths_df_current_gen.empty:
        print("Warning: load_car_path_df returned empty or None. Skipping fitness evaluation for this generation.")
        return [0] * len(population), population # 또는 적절한 에러 처리/기본값 반환

    path_history.append(len(car_paths_df_current_gen)) # 경로 길이 기록

    shm_infos_car_paths = None
    shm_objects_to_cleanup_in_main = [] # 메인 프로세스에서 close/unlink할 shm 객체들

    results = [] # imap 결과를 담을 리스트

    try:
        # 1. 현재 세대의 car_paths_df를 공유 메모리에 올림
        # unique_prefix는 세대별로 달라야 하므로 current_generation_number 사용
        shm_unique_prefix = f"gen{current_generation_number}"
        shm_infos_car_paths, shm_objects_to_cleanup_in_main = put_df_to_shared_memory(car_paths_df_current_gen, unique_prefix=shm_unique_prefix)
        
        del car_paths_df_current_gen # 공유 메모리에 올렸으므로 원본은 삭제 (메모리 절약)
        gc.collect()

        # 2. 워커에 전달할 인자 리스트 생성
        #    individual (유전자), idx (인덱스), shm_infos_car_paths (공유 car_paths_df 정보)
        #    station_df는 워커가 initializer를 통해 worker_original_station_df를 사용함
        args_list = [
            (individual, idx, shm_infos_car_paths)
            for idx, individual in enumerate(population)
        ]

        # 3. 멀티프로세싱 실행 (pool.imap 사용)
        map_start_time = time.time()
        calculated_chunksize = max(1, len(population) // CORE_NUM) if CORE_NUM > 0 else 1

        results = list(pool.imap(evaluate_individual_shared, args_list, chunksize=calculated_chunksize))
        map_end_time = time.time()
        print(f"  - 멀티프로세싱 시간: {map_end_time - map_start_time:.2f}초")
        
    except Exception as e:
        print(f"ERROR in fitness_func during shared memory setup or pool.imap: {e}")
        print(f"!!! Exception Type: {type(e)} !!!")
        print(f"!!! Exception Message: {str(e)} !!!")
        # 예외 발생 시 로깅하거나 기본값을 반환할 수 있습니다.
        # 이 경우, 생성된 공유 메모리가 있다면 정리해야 합니다.
        raise
    finally:
        for shm_obj in shm_objects_to_cleanup_in_main:
            try:
                shm_obj.close() # 메인 프로세스에서 연결 해제
                shm_obj.unlink()# 시스템에서 제거 (이것이 중요)
            except FileNotFoundError:
                print(f"SHM object {shm_obj.name if shm_obj else 'Unknown'} already unlinked or not fully created.")
            except Exception as e_clean:
                print(f"Error cleaning up SHM object {shm_obj.name if shm_obj else 'Unknown'} in fitness_func: {e_clean}")
        
    # 5. 결과 처리 및 반환
    if not results: # imap에서 에러가 나서 비어있을 경우
        print("Warning: pool.imap returned empty results.")
        return [-np.inf] * len(population), population
   
    # 결과 정렬 (기존 로직)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sorted_results]
    sorted_population = [population[i] for i in sorted_indices]
    fitness_values = [fitness if fitness is not None else -np.inf for _, fitness in sorted_results]
    
    return fitness_values, sorted_population


def choice_gene_tournament_no_duplicate(population, tournament_size, num_parents, fitness_values):
    """
    토너먼트 선택을 사용하여 부모를 중복 없이 선택하고, 적합도(fitness)를 기준으로 내림차순 정렬하여 반환합니다.

    Args:
        population (list): 부모 개체군 리스트. 각 요소는 NumPy 배열(유전자)입니다.
        tournament_size (int): 토너먼트 크기. 한 번에 경쟁할 후보 개체의 수입니다.
        num_parents (int): 선택할 부모의 수입니다.
        fitness_values (list): 각 부모 개체의 적합도(fitness)를 나타내는 리스트입니다. `population`과 같은 순서로 정렬되어 있습니다.

    Returns:
        list: 적합도(fitness)를 기준으로 내림차순 정렬된 부모 개체(NumPy 배열) 리스트를 반환합니다.

    Raises:
        ValueError: `population`의 크기가 1보다 작거나 같을 때, 또는 `tournament_size`가 `population`의 크기보다 클 때 발생합니다.
    """

    # 입력 값 검증: population 크기, tournament_size 유효성 확인
    if len(population) <= 1:
        raise ValueError("Value Error: Population size must be greater than 1")
    if tournament_size > len(population):
        raise ValueError("Value Error: Tournament size cannot be greater than population size")

    # 1. population과 fitness_values를 DataFrame으로 묶기
    df = pd.DataFrame({'parent': population, 'fitness': fitness_values})

    # 2. 선택된 부모와 남은 인덱스 초기화
    selected_parents = []  # 선택된 부모를 저장할 빈 리스트
    remain_indices = list(df.index)  # 아직 선택되지 않은 개체의 인덱스를 추적하기 위한 리스트

    # 3. 토너먼트 선택을 통해 부모 선택
    while len(selected_parents) < num_parents and len(remain_indices) > 0:
        # 3-1. 현재 토너먼트 크기 결정
        current_t_size = min(tournament_size, len(remain_indices))  # 남은 인덱스가 tournament_size보다 적으면, 남은 인덱스 수를 사용

        # 3-2. 토너먼트 참가자 무작위 선택
        contestants_indices = random.sample(remain_indices, k=current_t_size)  # remain_indices에서 current_t_size만큼 무작위로 인덱스를 선택

        # 3-3. DataFrame에서 토너먼트 수행
        contestants_df = df.loc[contestants_indices]  # 선택된 인덱스에 해당하는 행만 추출하여 새로운 DataFrame 생성

        # 3-4. 토너먼트 우승자(최고 적합도) 결정
        best_index = contestants_df['fitness'].idxmax()  # 'fitness' 열에서 최댓값을 가진 행의 인덱스를 찾음

        # 3-5. 우승자를 selected_parents에 추가
        selected_parents.append(df.loc[best_index, 'parent'])  # best_index에 해당하는 'parent' 열의 값을 selected_parents에 추가

        # 3-6. 선택된 인덱스를 remain_indices에서 제거
        remain_indices.remove(best_index)  # 선택된 인덱스는 더 이상 토너먼트에 참여하지 않도록 remain_indices에서 제거

    # 4. 선택된 부모를 적합도 기준으로 내림차순 정렬
    selected_df = df[df['parent'].isin(selected_parents)].copy()  # 선택된 부모만 포함하는 새로운 DataFrame 생성 (isin()으로 필터링)
    selected_df.sort_values('fitness', ascending=False, inplace=True)  # 'fitness' 열을 기준으로 내림차순 정렬 (inplace=True로 자체 변경)

    # 5. 정렬된 부모 배열열 반환
    return selected_df['parent'].tolist()  # 'parent' 열만 추출하여 반환


def crossover_elitsm(selected_parents, num_genes, pop_size, generation):
    """다중지점 교차를 통해 새로운 세대의 개체를 생성하는 함수."""
    global duplicate_info_df
    crossover = []
    
    elitism = selected_parents[:1]
    crossover.extend(elitism)
    # 나머지 자손 생성
    while len(crossover) < pop_size:
        # 부모 10명 선택
        parents = random.sample(selected_parents, 10)
        
        crossover_points = sorted(random.sample(range(1, num_genes), 9))

        child = np.concatenate([
            parents[0][:crossover_points[0]],
            parents[1][crossover_points[0]:crossover_points[1]],
            parents[2][crossover_points[1]:crossover_points[2]],
            parents[3][crossover_points[2]:crossover_points[3]],
            parents[4][crossover_points[3]:crossover_points[4]],
            parents[5][crossover_points[4]:crossover_points[5]],
            parents[6][crossover_points[5]:crossover_points[6]],
            parents[7][crossover_points[6]:crossover_points[7]],
            parents[8][crossover_points[7]:crossover_points[8]],
            parents[9][crossover_points[8]:]])
        crossover.append(child)
    
    # 총 중복 개체 수
    counter = Counter(tuple(ind) for ind in crossover)
    duplicates_count = sum((count - 1) for count in counter.values() if count > 1)

    # 중복 정보 저장 및 출력
    if duplicates_count > 0:
        print(f"[중복 개체 확인] 이번 세대에서 총 {duplicates_count}개의 중복 개체가 발견되었습니다.")
        
        duplicate_info = {}
        for idx, ind in enumerate(crossover):
            ind_tuple = tuple(ind)
            if counter[ind_tuple] > 1:
                if ind_tuple not in duplicate_info:
                    duplicate_info[ind_tuple] = []
                duplicate_info[ind_tuple].append(idx)

        for solution, indices in duplicate_info.items():
            # 중복 정보 출력
            print(f" - ID {indices[0]}와 동일한 개체가 {len(indices)}회 등장 (인덱스: {indices})")

            # DataFrame에 저장
            duplicate_info_df = pd.concat([duplicate_info_df, pd.DataFrame({
                'Generation': [generation + 1],
                'Solution': [str(solution)], 
                'Indices': [str(indices)],
                'Count': [len(indices)]
            })], ignore_index=True)

    else:
        print("[중복 개체 확인] 이번 세대에서는 중복된 개체가 없습니다.")
        duplicate_info_df = pd.concat([duplicate_info_df, pd.DataFrame({
            'Generation': [generation + 1],
            'Solution': ['No Duplicates'],
            'Indices': ['[]'],
            'Count': [0]
        })], ignore_index=True)

    return crossover[:pop_size]
    
def get_random_charger(max, adaptive_constant):    
    # 기본 확률 계산
    base_prob_0 = 0.5 - round((abs(GENERATIONS-adaptive_constant)/GENERATIONS), 4) * 0.3  # 20% ~ 50%   뒤로 갈수록 0 나올 확률 증가
    base_prob_1_max = 0.5 + round((abs(GENERATIONS-adaptive_constant)/GENERATIONS), 4) * 0.3  # 80% ~ 50% 초반 다양한 해 탐색, 후반 안정성
    
    prob_ratio_for_new_range = 0.1  # 10%를 새로운 범위에 할당
    prob_new_range = base_prob_1_max * prob_ratio_for_new_range  # 8% ~ 5%
    prob_1_max_adjusted = base_prob_1_max * (1 - prob_ratio_for_new_range)  # 72% ~ 45%
    
    mutated_charger_number = [
        (0, 0, base_prob_0),                    # 20% ~ 50%
        (1, max, prob_1_max_adjusted),          # 72% ~ 45%
        (max + 1, 2 * max, prob_new_range)     # 8% ~ 5%
    ]
    
    rand_val = random.random()
    cumulative_prob = 0
    for lower, upper, prob in mutated_charger_number:
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return random.randint(lower, upper)
                

def mutation(crossovered, pop_size, mutation_rate, num_candi, initial_chargers, adaptive_constant):
    """
    돌연변이 함수.

    - crossovered를 DataFrame으로 변환한 후, id 열을 추가합니다.
    - DataFrame을 사용하여 중복 개체를 관리합니다.
    - 중복된 개체가 존재하면 중복된 개체 중 원본을 제외한 나머지 개체들에 대해 유전자를 일부 변경합니다.
      이때, 중복 개체 수만큼 변경할 유전자 개수를 조정합니다.
    - 중복 개체 수가 num_candi의 절반을 초과하면, 유전자 변경을 num_candi의 절반까지만 수행합니다.
    - 중복이 없을 때는 무작위 개체에 대해 변이(전체 유전자 변경)를 수행합니다.
    - 중복이 있을 때는 중복된 개체 중 원본을 제외한 나머지 개체에 대해서만 변이(전체 유전자 변경)를 수행합니다.
    """

    # 1. crossovered를 DataFrame으로 변환하고 ID 열 추가
    df = pd.DataFrame(crossovered)
    df['id'] = range(len(df))  # 새로운 ID 열 추가

    # 2. 중복 개체 확인 및 ID 그룹화
    duplicates = df.groupby(list(range(num_candi)))['id'].apply(list)
    duplicates = duplicates[duplicates.str.len() > 1].to_dict()

    unique_indices = df.groupby(list(range(num_candi)))['id'].apply(list)
    unique_indices = unique_indices[unique_indices.str.len() == 1].to_dict()

    # 중복이 없는 개체의 인덱스 리스트를 생성합니다.
    unique_indices = [idx for indices in unique_indices.values() for idx in indices]

    # num_genes_to_change 초기화 
    num_genes_to_change = 0

    # 3. 중복 개체가 있는 경우
    for ind, indices in duplicates.items():
        num_duplicates = len(indices) - 1  # 자신(원본)을 제외한 중복 개체 수

        # 원본 개체의 최댓값 찾기
        original_genome = df.loc[df['id'] == indices[0], list(range(num_candi))].iloc[0].values
        max_gene = np.max(original_genome)

        # 3-1. 유전자 일부 변경 (중복 개체 수만큼 변경, 최대 num_candi 절반까지)
        # 중복된 해의 일부 유전자를 무작위로 변경 (여기서 num_duplicates 사용)
        num_genes_to_change = int(min(num_duplicates*MUTATION_GENES_MULTIPLE*(round((abs(GENERATIONS-adaptive_constant)/GENERATIONS)*0.9+0.1, 4)), num_candi // 2))

        for i in range(1, len(indices)):  # 원본(첫 번째 인덱스)을 제외하고 나머지 중복 개체에 대해 수행
            
            modify_index = indices[i]

            # 변경할 유전자 위치 무작위 선택
            indices_to_change = random.sample(range(num_candi), num_genes_to_change)
            for idx in indices_to_change:
                # 해당 위치의 유전자를 0부터 원본 개체 최댓값 이하의 무작위 값으로 변경
                original_value = df.loc[df['id'] == modify_index, idx].iloc[0]  # 원본 유전자 값
                new_value = original_value
                while new_value == original_value:
                    new_value = get_random_charger(max_gene+1, adaptive_constant)  # 0 ~ max_gene * MUTATIOM_RANGE 사이의 값으로 변경
                df.loc[df['id'] == modify_index, idx] = new_value

    # 4. 돌연변이 (전체 유전자 변경)

    # 중복이 없는 경우 무작위 개체 변이
    if not duplicates:
        for i in range(pop_size):  # range(pop_size) 사용
            if random.random() <= mutation_rate:
                # unique_indices에 있는지 확인 후, 있으면 해당 인덱스 사용
                if i in unique_indices:
                    df.loc[df['id'] == i, list(range(num_candi))] = np.random.multinomial(initial_chargers, [1/num_candi]*num_candi)
                    print(f"돌연변이 발생 (전체 유전자 변경): 인덱스 {i}")

    # 중복이 있는 경우 중복 개체 변이
    else:
        for indices in duplicates.values():
            for i in indices[1:]:  # 원본 제외
                if random.random() <= mutation_rate*5:
                    df.loc[df['id'] == i, list(range(num_candi))] = np.random.multinomial(initial_chargers, [1/num_candi]*num_candi)
                    print(f"돌연변이 발생 (전체 유전자 변경): 인덱스 {i} (중복)")

    # 5. 수정된 개체들을 다시 crossovered로 변환
    crossovered = df[list(range(num_candi))].to_numpy() # id 제외
    print(f"num_genes_to_change: {num_genes_to_change}, adaptive probablity: {(round((abs(GENERATIONS-adaptive_constant)/GENERATIONS)*0.9+0.1, 4)*100):.2f}%")
    print(f"돌연변이에서 0이 나올 확률 : {(0.2+round((abs(GENERATIONS-adaptive_constant)/GENERATIONS), 4)*0.3)*100:.4f}%")

    del df # df 명시적으로 삭제

    return crossovered


def immigration(population, num_candi, initial_chargers, generation):
    """이민자 연산을 통해 개체군 다양성을 유지하는 함수.
    하위 일부 개체를 완전히 새로운 무작위 개체로 교체한다."""
    
    replace_ratio = 0.4 - 0.3 * (generation / GENERATIONS)  # 10%~40% 범위로 조정
    num_to_replace = max(1, int(len(population) * replace_ratio))
    
    print(f"이민자 연산: {num_to_replace}개 개체 교체 (전체의 {replace_ratio*100:.1f}%)")
    
    for i in range(len(population) - num_to_replace, len(population)):
        #하위 개체에 대해서 완전히 새로운 해 생성
        population[i] = np.random.multinomial(initial_chargers, [1/num_candi] * num_candi)
            
    return population


def genetic_algorithm():
    no_improvement_count = 0
    best_fitness = float('-inf')
    fitness_history = []
    path_history = []
    immigration_count = 0
    MAX_IMMIGRATIONS = 5
    convergence_count = 0
    last_immigration_generation = -50 # 마지막 이민자 연산 로직 추적

    convergence_df = pd.DataFrame({
        'Generation': pd.Series(dtype='int'),
        'Fitness_Mean': pd.Series(dtype='float'),
        'Fitness_Mean_Change': pd.Series(dtype='float'),
        'Charger_Change': pd.Series(dtype='float'),
        'Best_Fitness': pd.Series(dtype='float'),
        'Best_Chargers': pd.Series(dtype='int')
    })

    path_for_result = r"/home/semlab/SEM/EVCS/화물차 충전소 배치 최적화/Data/Processed_Data/GA_result"
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d-%H-%M")
    result_folder_path = os.path.join(path_for_result, folder_name)
    os.makedirs(result_folder_path, exist_ok=True)

    # 각 세대의 전체 적합도 값 저장용 리스트
    all_fitness_history = []
    min_fitness_history = []
    max_fitness_history = []
    mean_fitness_history = []
    best_fitness_number_of_charger = []
    best_individual_history = []  # 세대별 최고 개체 유전자 저장

    station_file_path = path_for_station
    original_station_df = si.load_station_df(station_file_path)

    best_individual = None  # 역대 최고 개체 정보 저장 변수 초기화
    last_generation_individuals = None  # 마지막 세대 개체 정보 저장 변수 초기화

    # --- 주기적 저장을 위한 버퍼 초기화 ---
    current_file_batch_num = 0
    convergence_data_batch = []
    all_population_details_batch = []
    current_batch_best_individuals_genes = []
    current_batch_best_fitness_values = []
    current_batch_min_fitness = []
    current_batch_max_fitness = []
    current_batch_mean_fitness = []
    current_batch_generation_numbers = []

    population_data_for_current_generation = None

    # 이 폴더명들이 생성되고, 주기적 파일들이 이 안에 저장됩니다.
    folder_names_for_periodic_data = {
        "convergence_info": "convergence_info",
        "all_population": "all_population",
        "best_individuals": "best_individuals",
        "ga_results": "ga_results", 
        "best_individual": "best_individual"
    }

    # 데이터 유형별 폴더 미리 생성
    for key, periodic_folder_name in folder_names_for_periodic_data.items():
        os.makedirs(os.path.join(result_folder_path, periodic_folder_name), exist_ok=True)

    for generation in range(GENERATIONS): # 각 세대 루프 시작
        start_time = time.time()
        print(f"\n세대 {generation + 1}/{GENERATIONS} 진행 중...")

        # 매 세대마다 Pool 생성
        with Pool(processes=CORE_NUM,
                initializer=init_worker_station,
                initargs=(original_station_df,)) as pool: # original_station_df는 루프 밖에서 한 번만 로드
            
            if generation == 0:
                population = station_gene_initial(POPULATION_SIZE, NUM_CANDIDATES, INITIAL_CHARGERS)
            else:
                population = mutated # 이전 세대의 mutated 결과를 사용

            if generation == GENERATIONS: 
                print("지정된 세대의 연산이 종료되었으므로 계산 결과를 출력합니다")
                break # 루프를 빠져나감

            # 적합도 계산 (pool 객체를 인자로 전달)
            fitness_values, sorted_population = fitness_func(population, original_station_df, path_history, pool, generation)
            print('적합도 평가 완료')

            # 모든 개체 정보 저장(버퍼)
            for i in range(len(sorted_population)):
                individual_data_for_batch = {'Generation': generation + 1}
                for gene_idx, gene_val in enumerate(sorted_population[i]):
                    individual_data_for_batch[f'Gene_{gene_idx+1}'] = gene_val
                individual_data_for_batch['Fitness'] = fitness_values[i]
                all_population_details_batch.append(individual_data_for_batch)

            # 전체 적합도 값 저장
            all_fitness_history.append(fitness_values)
            print('적합도 값 저장 완료')

            # 세대별 최소, 최대, 평균 적합도 저장
            current_min = np.min(fitness_values)
            current_max = np.max(fitness_values)
            current_mean = np.mean(fitness_values)

            min_fitness_history.append(current_min)
            max_fitness_history.append(current_max)
            mean_fitness_history.append(current_mean)

            current_batch_min_fitness.append(current_min)
            current_batch_max_fitness.append(current_max)
            current_batch_mean_fitness.append(current_mean)
            current_batch_generation_numbers.append(generation + 1)

            current_best_fitness = current_max
            current_best_individual = sorted_population[0]
            
            best_individual_history.append(current_best_individual)
            fitness_history.append(current_best_fitness)

            current_batch_best_individuals_genes.append(current_best_individual)
            current_batch_best_fitness_values.append(current_best_fitness)
            
            current_best_chargers = int(np.sum(current_best_individual))

            current_best_individual_list = current_best_individual.tolist() # NumPy 배열을 리스트로 변환
            print(f"  - 현재 세대 최고 적합도 (OFV): {current_max:,.0f}")
            print(f"  - 현재 세대 최고 개체 유전자: ")
            columns_to_show = 50  # 한 줄에 보여줄 유전자 개수 (사용자 설정 가능)

            # 각 숫자를 일정한 너비로 맞추기 위해 최대 자릿수 계산 (선택 사항, 정렬에 도움)
            if current_best_individual_list: # 리스트가 비어있지 않은 경우에만 max 계산
                try:
                    # 모든 요소가 숫자인지 확인 후 max() 적용, 그렇지 않으면 기본 너비 사용
                    if all(isinstance(item, (int, float)) for item in current_best_individual_list):
                        max_val_in_list = max(current_best_individual_list) if current_best_individual_list else 0
                        max_width = len(str(max_val_in_list)) + 1  # 최대값 자릿수 + 공백 1칸
                    else:
                        max_width = 5 # 숫자가 아닌 경우 기본 너비
                except ValueError: # max() 함수가 비어있는 시퀀스에 대해 호출될 경우 등
                    max_width = 2 # 기본값
            else: # 리스트가 비었을 경우
                max_width = 2 # 기본값

            for index, item in enumerate(current_best_individual_list):
                # f-string을 사용하여 각 숫자를 max_width 만큼의 공간에 왼쪽 정렬하여 출력
                # 사용자 원본 코드에서는 왼쪽 정렬을 사용했습니다: f"{item:<{max_width}}"
                print(f"{str(item):<{max_width}}", end="") # item을 str()으로 변환하여 다양한 타입에 대응
                
                # (index + 1)이 columns_to_show의 배수이면 줄 바꿈
                if (index + 1) % columns_to_show == 0:
                    print() # 새 줄로 이동

            # 마지막 줄이 columns_to_show 개수만큼 채워지지 않았을 경우,
            # 마지막에 줄 바꿈을 한 번 더 해줘서 다음 출력이 이어지지 않도록 함
            if len(current_best_individual_list) % columns_to_show != 0 and len(current_best_individual_list) > 0:
                print()

            # 수렴 체크를 위한 변수 초기화 (세대 10 이전에는 0으로 설정)
            charger_change = 0
            fitness_mean_change = 0
            curr_10_mean = 0

            # 수렴 체크 (새로운 방식)
            if generation >= 10:  # 11개 세대(index 10) 이상의 데이터가 쌓여야 비교 가능
                # 1. 충전기 개수 변화량 체크
                prev_best_chargers = np.sum(best_individual_history[-2])  # 이전 세대 최고 개체 충전기 수
                charger_change = (current_best_chargers - prev_best_chargers) / abs(prev_best_chargers)

                # 2. 적합도 평균 변화량 체크
                prev_10_fitness = fitness_history[-11:-1]  # 이전 10개 세대 최고 적합도 (현재 세대 미미포함)
                curr_10_fitness = fitness_history[-10:]    # 현재 10개 세대 최고 적합도 (현재 세대 포함)

                prev_10_mean = np.mean(prev_10_fitness)
                curr_10_mean = np.mean(curr_10_fitness)

                if prev_10_mean != 0:
                    fitness_mean_change = (curr_10_mean - prev_10_mean) / abs(prev_10_mean)

                # no_improvement_count 증가/초기화 조건 추가
                if generation >= CONVERGENCE_CHECK_START_GENERATIONS:
                    if abs(charger_change) <= 0.01 and abs(fitness_mean_change) <= 0.01:
                        no_improvement_count += 1
                        print(f"충전기 수 변화율: {charger_change * 100:.2f}%, 적합도 평균 변화율: {fitness_mean_change * 100:.2f}%")
                        print(f"{no_improvement_count} 세대 동안 개선 없음")
                    else:
                        no_improvement_count = 0  # 조건 불만족 시 초기화
                        print(f"충전기 수 변화율: {charger_change * 100:.2f}%, 적합도 평균 변화율: {fitness_mean_change * 100:.2f}%")
                        print(f"{no_improvement_count} 세대 동안 개선 없음")
                else: # CONVERGENCE_CHECK_START_GENERATIONS 이전에는 카운트 변화 없음
                    print(f"충전기 수 변화율: {charger_change * 100:.2f}%, 적합도 평균 변화율: {fitness_mean_change * 100:.2f}%")
                    print(f"개선 여부 판단 유보 (세대 {generation+1}/{CONVERGENCE_CHECK_START_GENERATIONS})")


                if no_improvement_count >= MAX_NO_IMPROVEMENT and immigration_count >= MAX_IMMIGRATIONS:
                    # 연속 MAX_NO_IMPROVEMENT 세대 동안 개선이 없으며 이민자 연산을 전부 수행하면 전체 종료
                    print(f"최적해 변화가 {MAX_NO_IMPROVEMENT} 세대 연속 없어 알고리즘을 종료합니다.")
                    convergence_df = pd.concat([convergence_df, pd.DataFrame({
                        'Generation': [generation + 1],
                        'Fitness_Mean': [curr_10_mean],
                        'Fitness_Mean_Change': [fitness_mean_change],
                        'Charger_Change': [charger_change],
                        'Best_Fitness': [current_best_fitness],         # 현재 세대 최고 적합도
                        'Best_Chargers': [current_best_chargers]        # 최고 적합도 개체의 충전기 수
                    })], ignore_index=True)
                    break

            else: # 11개 세대가 쌓이지 않았을 경우
                no_improvement_count = 0
                print("변화량 측정을 위한 data 축적중")

            # convergence_df에 결과 저장
            convergence_df = pd.concat([convergence_df, pd.DataFrame({
                'Generation': [generation + 1],
                'Fitness_Mean': [curr_10_mean],
                'Fitness_Mean_Change': [fitness_mean_change],
                'Charger_Change': [charger_change],
                'Best_Fitness': [current_best_fitness],         # 현재 세대 최고 적합도
                'Best_Chargers': [current_best_chargers]        # 최고 적합도 개체의 충전기 수
            })], ignore_index=True)

            # 역대 최고 개체 갱신
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                print("역대 최고 개체 갱신")

            # 마지막 세대 개체 정보 저장
            if generation == GENERATIONS - 1:
                last_generation_individuals = sorted_population
                print("마지막 세대 개체 정보 저장")

            # 부모 선택
            parents = choice_gene_tournament_no_duplicate(sorted_population, TOURNAMENT_SIZE, PARENTS_SIZE, fitness_values)
            print('부모 선택 완료')
            best_fitness_number_of_charger.append(np.sum(parents[0]))

            if (no_improvement_count >= 5 and immigration_count <= MAX_IMMIGRATIONS and (generation - last_immigration_generation) >= 50):
                parents = immigration(parents, NUM_CANDIDATES, INITIAL_CHARGERS, generation)
                immigration_count += 1
                no_improvement_count = 0
                last_immigration_generation = generation
                print(f"이민자 연산 실행 ({immigration_count}/{MAX_IMMIGRATIONS})")  
                print("수렴 카운트 초기화")

            # 교차
            children = crossover_elitsm(parents, NUM_CANDIDATES, POPULATION_SIZE, generation)
            print('교차 연산 완료')

            # 변이
            mutated = mutation(children, POPULATION_SIZE, MUTATION_RATE, NUM_CANDIDATES, INITIAL_CHARGERS, generation)
            print('변이 연산 완료')

            # --- 주기적 파일 저장 로직 (폴더 구조 변경 적용) ---
            should_save_batch = False
            if (generation + 1) % SAVE_INTERVAL == 0:
                should_save_batch = True
                current_file_batch_num += 1
            elif (generation == GENERATIONS - 1): # 정상 종료되는 마지막 세대
                should_save_batch = True
                if not ((generation + 1) % SAVE_INTERVAL == 0) and current_batch_generation_numbers:
                     current_file_batch_num += 1
            
            if should_save_batch and current_batch_generation_numbers:
                batch_start_gen = (current_file_batch_num - 1) * SAVE_INTERVAL + 1
                batch_end_gen = generation + 1

                print(f"\n--- 주기적 저장 실행 (폴더 기반): 세대 {batch_start_gen}-{batch_end_gen} ---")
                
                file_suffix = f"g{batch_start_gen}-{batch_end_gen}.csv"

                # 1. Convergence Info
                if convergence_data_batch:
                    folder = os.path.join(result_folder_path, folder_names_for_periodic_data["convergence_info"])
                    pd.DataFrame(convergence_data_batch).to_csv(os.path.join(folder, file_suffix), index=False, mode='w')
                    print(f"Saved: {os.path.join(folder_names_for_periodic_data['convergence_info'], file_suffix)}")
                    convergence_data_batch.clear()

                # 2. All Population
                if all_population_details_batch:
                    folder = os.path.join(result_folder_path, folder_names_for_periodic_data["all_population"])
                    pd.DataFrame(all_population_details_batch).to_csv(os.path.join(folder, file_suffix), index=False, mode='w')
                    print(f"Saved: {os.path.join(folder_names_for_periodic_data['all_population'], file_suffix)}")
                    all_population_details_batch.clear()
                
                # 3. Best Individuals
                if current_batch_best_individuals_genes:
                    folder = os.path.join(result_folder_path, folder_names_for_periodic_data["best_individuals"])
                    df_to_save = pd.DataFrame(current_batch_best_individuals_genes, columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)]) # NUM_CANDIDATES 필요
                    df_to_save['Fitness'] = current_batch_best_fitness_values
                    df_to_save['Actual_Generation'] = current_batch_generation_numbers[-len(current_batch_best_individuals_genes):]
                    df_to_save.to_csv(os.path.join(folder, file_suffix), index=False, mode='w')
                    print(f"Saved: {os.path.join(folder_names_for_periodic_data['best_individuals'], file_suffix)}")
                    current_batch_best_individuals_genes.clear(); current_batch_best_fitness_values.clear()

                # 4. GA Results (Min/Max/Mean Fitness)
                if current_batch_min_fitness:
                    folder = os.path.join(result_folder_path, folder_names_for_periodic_data["ga_results"])
                    df_to_save = pd.DataFrame({
                        'Actual_Generation': current_batch_generation_numbers[-len(current_batch_min_fitness):],
                        'Min_Fitness': current_batch_min_fitness,
                        'Max_Fitness': current_batch_max_fitness,
                        'Mean_Fitness': current_batch_mean_fitness
                    })
                    df_to_save.to_csv(os.path.join(folder, file_suffix), index=False, mode='w')
                    print(f"Saved: {os.path.join(folder_names_for_periodic_data['ga_results'], file_suffix)}")
                    current_batch_min_fitness.clear(); current_batch_max_fitness.clear(); current_batch_mean_fitness.clear()
                
                # 5. Best Individual 
                if best_individual is not None:
                    folder = os.path.join(result_folder_path, folder_names_for_periodic_data["best_individual"])
                    snapshot_file_name = f"at_g{batch_end_gen}.csv"
                    df_to_save = pd.DataFrame([best_individual], columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)])
                    df_to_save['Fitness'] = best_fitness
                    df_to_save.to_csv(os.path.join(folder, snapshot_file_name), index=False, mode='w')
                    print(f"Saved: {os.path.join(folder_names_for_periodic_data['best_individual'], snapshot_file_name)}")
                
                current_batch_generation_numbers.clear()
                print(f"--- 주기적 저장 완료 (폴더 기반) ---")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n세대 {generation + 1}/{GENERATIONS} 연산 소요 시간 {elapsed_time:.2f}초")

# --- 루프 종료 후, 잔여 배치 데이터 저장 ---
    if current_batch_generation_numbers: # 저장할 데이터가 남아있다면 (조기 종료 등으로 인해)
        current_file_batch_num += 1
        batch_start_gen = (current_file_batch_num - 1) * SAVE_INTERVAL + 1
        batch_end_gen = generation + 1 # 루프가 종료된 실제 마지막 세대 + 1 (generation은 0부터 시작)

        print(f"\n--- 최종 잔여 데이터 저장 (폴더 기반): 세대 {batch_start_gen}-{batch_end_gen} ---")
        file_suffix = f"g{batch_start_gen}-{batch_end_gen}.csv"
        
        # (동일한 로직, 하지만 올바른 키 사용)
        if convergence_data_batch:
            folder = os.path.join(result_folder_path, folder_names_for_periodic_data["convergence_info"])
            pd.DataFrame(convergence_data_batch).to_csv(os.path.join(folder, file_suffix), index=False, mode='w'); print(f"Saved: {os.path.join(folder_names_for_periodic_data['convergence_info'], file_suffix)}")
        if all_population_details_batch:
            folder = os.path.join(result_folder_path, folder_names_for_periodic_data["all_population"])
            pd.DataFrame(all_population_details_batch).to_csv(os.path.join(folder, file_suffix), index=False, mode='w'); print(f"Saved: {os.path.join(folder_names_for_periodic_data['all_population'], file_suffix)}")
        if current_batch_best_individuals_genes:
            folder = os.path.join(result_folder_path, folder_names_for_periodic_data["best_individuals"])
            df_to_save = pd.DataFrame(current_batch_best_individuals_genes, columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)])
            df_to_save['Fitness'] = current_batch_best_fitness_values
            df_to_save['Actual_Generation'] = current_batch_generation_numbers[-len(current_batch_best_individuals_genes):]
            df_to_save.to_csv(os.path.join(folder, file_suffix), index=False, mode='w'); print(f"Saved: {os.path.join(folder_names_for_periodic_data['best_individuals'], file_suffix)}")
        if current_batch_min_fitness:
            folder = os.path.join(result_folder_path, folder_names_for_periodic_data["ga_results"])
            df_to_save = pd.DataFrame({'Actual_Generation': current_batch_generation_numbers[-len(current_batch_min_fitness):],
                                       'Min_Fitness': current_batch_min_fitness, 'Max_Fitness': current_batch_max_fitness,
                                       'Mean_Fitness': current_batch_mean_fitness})
            df_to_save.to_csv(os.path.join(folder, file_suffix), index=False, mode='w'); print(f"Saved: {os.path.join(folder_names_for_periodic_data['ga_results'], file_suffix)}")
        if best_individual is not None:
            folder = os.path.join(result_folder_path, folder_names_for_periodic_data["best_individual"])
            snapshot_file_name = f"at_g{batch_end_gen}.csv"
            # 버그 수정: df_to_save에 할당 후 저장해야 함
            df_to_save = pd.DataFrame([best_individual], columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)])
            df_to_save['Fitness'] = best_fitness
            df_to_save.to_csv(os.path.join(folder, snapshot_file_name), index=False, mode='w'); print(f"Saved: {os.path.join(folder_names_for_periodic_data['best_individual'], snapshot_file_name)}")
        print(f"--- 최종 잔여 데이터 저장 완료 (폴더 기반) ---")

        # 버퍼 비우기 (이미 위에서 수행됨, 명시적으로 한번 더)
        convergence_data_batch.clear(); all_population_details_batch.clear(); current_batch_best_individuals_genes.clear(); current_batch_best_fitness_values.clear(); current_batch_min_fitness.clear(); current_batch_max_fitness.clear(); current_batch_mean_fitness.clear(); current_batch_generation_numbers.clear()


    final_executed_generations = generation + 1

    if last_generation_individuals is None and 'sorted_population' in locals() and sorted_population is not None:
        last_generation_individuals = sorted_population
    if last_generation_individuals is not None:
        # 이 파일은 result_folder_path 최상위에 저장 (주기적 파일이 아님)
        last_gen_df = pd.DataFrame(last_generation_individuals, columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)])
        if all_fitness_history:
            num_individuals_in_last_gen = len(last_generation_individuals)
            last_fitness_values_for_df = all_fitness_history[-1][:num_individuals_in_last_gen]
            last_gen_df['Fitness'] = last_fitness_values_for_df
        last_gen_filename = os.path.join(result_folder_path, f"last_generation_g{final_executed_generations}.csv")
        last_gen_df.to_csv(last_gen_filename, index=False, mode='w'); print(f"Saved: {last_gen_filename}")
    
    if best_individual is not None:
        # 이 파일은 result_folder_path 최상위에 저장 (주기적 파일이 아님)
        final_best_df = pd.DataFrame([best_individual], columns=[f"Station_{i+1}" for i in range(NUM_CANDIDATES)])
        final_best_df['Fitness'] = best_fitness
        final_best_filename = os.path.join(result_folder_path, f"best_individual_overall_final_g{final_executed_generations}.csv")
        final_best_df.to_csv(final_best_filename, index=False, mode='w'); print(f"Saved: {final_best_filename}")

    print("\n유전 알고리즘 종료")
    print(f"최종 세대 수: {final_executed_generations}")
    print(f"최고 적합도: {best_fitness}")

   
    # 그래프 출력
    gens = np.arange(1, len(fitness_history) + 1)

    fig, ax = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    fig2, ax2 = plt.subplots(2, 1, figsize=(20, 14))  # 수정: 수렴 정보 그래프를 위한 figure
    fig3, ax3 = plt.subplots(1, 1, figsize=(20, 14))

    # 상단: 최고, 평균, 최소-최대 범위 표시
    ax[0].plot(gens, mean_fitness_history, label='Mean Fitness', color='blue', marker='o')
    ax[0].plot(gens, fitness_history, label='Best Fitness', color='red', marker='o')
    # ax[0].fill_between(gens, min_fitness_history, max_fitness_history, color='gray', alpha=0.3, label='Min-Max Range')
    ax[0].set_xticks(range(0, len(fitness_history), 100))
    ax[0].set_ylabel('Fitness')
    ax[0].set_xlabel('Generation')
    ax[0].legend()
    ax[0].grid(True)

    # 하단: 박스플랏으로 세대별 전체 분포 표시
    ax[1].boxplot(all_fitness_history, positions=gens)
    ax[1].set_xticks(range(0, len(fitness_history), 100))
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Fitness Distribution ()')
    ax[1].grid(False)
    fig.suptitle('Fitness Value of Each Generation', fontsize=16)  # fig 에 title 설정 (전체 figure title)

   # 수렴 정보 그래프 (수정)
    charger_changes = [0]  # 첫 세대는 변화 없음
    fitness_mean_changes = [0] # 첫 세대는 변화 없음

    for i in range(1, len(best_individual_history)):
        # 충전기 변화
        prev_chargers = np.sum(best_individual_history[i-1])
        curr_chargers = np.sum(best_individual_history[i])
        charger_changes.append(abs(curr_chargers - prev_chargers) / prev_chargers if prev_chargers !=0 else 0)

        # 적합도 평균 변화 (10개 세대 평균)
        if i >= 10:
            prev_mean = np.mean(fitness_history[i-10:i])
            curr_mean = np.mean(fitness_history[i-9:i+1])
            fitness_mean_changes.append(abs(curr_mean - prev_mean) / prev_mean if prev_mean != 0 else 0)

        else:
            fitness_mean_changes.append(0)


    ax2[0].plot(gens, charger_changes, label='Charger Change', color='blue', marker='o')
    ax2[0].set_xticks(range(0, len(gens), len(gens) // 5))
    ax2[0].set_ylabel('Charger Change Ratio')
    ax2[0].set_xlabel('Generation')
    ax2[0].legend()
    ax2[0].grid(True)

    ax2[1].plot(gens, fitness_mean_changes, label='Fitness Mean Change', color='red', marker='o')
    ax2[1].set_xticks(range(0, len(gens), len(gens) // 5))
    ax2[1].set_ylabel('Fitness Mean Change Ratio')
    ax2[1].set_xlabel('Generation')
    
    
    ax2[1].legend()
    ax2[1].grid(True)

    fig2.suptitle('Convergence Information', fontsize=16)


    # 최고 적합도 개체의 충전기 설치 개수 그래프
    ax3.plot(convergence_df['Generation'], convergence_df['Best_Chargers'], label='Number of Chargers', color='green', marker='o')
    ax3.set_xticks(range(0, len(convergence_df['Generation']), len(convergence_df['Generation']) // 5))  # convergence_df 기반
    ax3.set_ylabel('Number of Chargers')
    ax3.set_xlabel('Generation')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Number of Chargers in Best Fitness Solution')

    fig.tight_layout()  # fig 에 tight_layout 적용
    fig2.tight_layout()  # fig2 에 tight_layout 적용
    fig3.tight_layout()  # fig3 에 tight_layout 적용

    # 각 figure 별로 savefig 호출 및 파일명 변경
    fig.savefig(os.path.join(result_folder_path, 'fitness_history.png'))
    fig2.savefig(os.path.join(result_folder_path, 'convergence_info.png'))  # 수정: 수렴 정보 그래프 저장
    fig3.savefig(os.path.join(result_folder_path, 'charger_count.png'))

    plt.show()  # 마지막에 plt.show() 호출하여 그래프 화면 출력 (선택 사항)

def _extract_gen_numbers_from_filename(filename):
    """ Helper function to extract start and end generation numbers from filenames like 'g1-50.csv' or 'at_g50.csv' """
    match_range = re.match(r"g(\d+)-(\d+)\.csv", filename)
    if match_range:
        return int(match_range.group(1)), int(match_range.group(2))
    match_single = re.match(r"at_g(\d+)\.csv", filename)
    if match_single:
        num = int(match_single.group(1))
        return num, num # Treat as a range of one for sorting
    return None, None # Or raise an error/return a value indicating no match

def _merge_periodic_csv_files(base_result_folder, data_type_folder_names):
    """
    Merges periodically saved CSV files for each data type into a single CSV file.
    The merged file is saved in a 'merged_data' subfolder within each data type folder.
    """
    print("\n--- 주기별 파일 통합 시작 ---")
    for data_folder_name in data_type_folder_names:
        current_data_type_folder = os.path.join(base_result_folder, data_folder_name)
        
        if not os.path.isdir(current_data_type_folder):
            print(f"폴더를 찾을 수 없습니다 (통합 건너뛰기): {current_data_type_folder}")
            continue

        # 주기별 파일 패턴 (예: g1-50.csv, at_g50.csv)
        # glob을 사용하여 모든 csv 파일을 가져옴
        periodic_files_pattern = os.path.join(current_data_type_folder, "*.csv")
        periodic_files = glob.glob(periodic_files_pattern)

        if not periodic_files:
            print(f"통합할 CSV 파일이 없습니다: {current_data_type_folder}")
            continue
            
        # 파일명에서 세대 정보를 추출하여 정렬
        # (start_gen, end_gen, filepath) 튜플 리스트 생성
        files_with_gen_info = []
        for f_path in periodic_files:
            f_name = os.path.basename(f_path)
            start_g, end_g = _extract_gen_numbers_from_filename(f_name)
            if start_g is not None: # 유효한 파일명 패턴일 경우에만 추가
                files_with_gen_info.append((start_g, end_g, f_path))
        
        # 시작 세대 번호 기준으로 정렬
        files_with_gen_info.sort(key=lambda x: x[0])
        
        sorted_periodic_files = [f_path for start_g, end_g, f_path in files_with_gen_info]

        if not sorted_periodic_files:
            print(f"유효한 패턴의 통합할 CSV 파일이 없습니다 (정렬 후): {current_data_type_folder}")
            continue

        all_data_frames = []
        for f_path in sorted_periodic_files:
            try:
                df = pd.read_csv(f_path)
                all_data_frames.append(df)
            except Exception as e:
                print(f"파일 읽기 오류 (건너뛰기): {f_path} - {e}")
                continue
        
        if not all_data_frames:
            print(f"읽을 수 있는 데이터 프레임이 없습니다 (통합 실패): {data_folder_name}")
            continue
            
        merged_df = pd.concat(all_data_frames, ignore_index=True)
        
        merged_data_subfolder = os.path.join(current_data_type_folder, "merged_data")
        os.makedirs(merged_data_subfolder, exist_ok=True)
        
        merged_filename = f"all_{data_folder_name}.csv"
        merged_filepath = os.path.join(merged_data_subfolder, merged_filename)
        
        try:
            merged_df.to_csv(merged_filepath, index=False, mode='w')
            print(f"통합 파일 저장 완료: {os.path.join(data_folder_name, 'merged_data', merged_filename)}")
        except Exception as e:
            print(f"통합 파일 저장 오류: {merged_filepath} - {e}")
            
    print("--- 주기별 파일 통합 완료 ---")

if __name__ == '__main__':
    try:
        # 멀티프로세싱 시작 방식을 'forkserver'로 설정합니다.
        multiprocessing.set_start_method('forkserver', force=True)
        print("멀티프로세싱 시작 방식을 'forkserver'로 설정했습니다.")
    except RuntimeError as e:
        # 이미 시작 방식이 설정되어 변경할 수 없는 경우 등 RuntimeError가 발생할 수 있습니다.
        print(f"경고: 멀티프로세싱 시작 방식을 'forkserver'로 변경하지 못했습니다. ({e})")
        print("      이미 설정되었거나 다른 컨텍스트에서 호출되었을 수 있습니다.")
    except Exception as e_other:
        # 기타 예외 처리
        print(f"멀티프로세싱 시작 방식 설정 중 예기치 않은 오류 발생: {e_other}")
        
    genetic_algorithm()

