#화물차 궤적 데이터 기반 시뮬레이션 개발 및 전기차 충전소 위치 및 규모 최적화
import os
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import Simulator_for_day as si
from multiprocessing import Pool, cpu_count
import logging
from collections import Counter

path_for_car = r"C:\Users\yemoy\SEM_화물차충전소\drive-download-20241212T004036Z-001"
path_for_car_oneday =r"C:\Users\yemoy\SEM_화물차충전소\경로폴더"
path_for_station = r"C:\Users\yemoy\SEM_화물차충전소\Final_Candidates_Selected.csv"
#random.seed(42)
#np.random.seed(42)

#100개의 솔루션 -> 토너먼트로 25개 선정 -> 25개중 상위 4개는 엘리티즘 -> 1등을 제외한 24개에 대하여 교차를 통해 96개의 해 생성  -> 100개의 다음세대 솔루션 생성.
#해당 사항으로 우선 알고리즘이 개발되어 일단은 한 세대당 100개의 솔루션이 있음을 가정하고 시뮬레이터에 적용하길 바랍니다.
# 유전 알고리즘 파라미터 설정
POPULATION_SIZE = 100  # 개체군 크기
GENERATIONS = 10000  # 최대 세대 수 (필요 시 무시됨)
TOURNAMENT_SIZE = 4 # 토너먼트 크기
MUTATION_RATE = 0.015  # 변이 확률
MUTATION_GENES_MULTIPLE = 20  # 중복된 해에 들어간 유전자 정보의 변이 배수
NUM_CANDIDATES = 500 # 충전소 위치 후보지 개수
CONVERGENCE_CHECK_START_GENERATIONS = 1000  # 수렴 체크 시작 세대
MAX_NO_IMPROVEMENT = 15  # 개선 없는 최대 세대 수
INITIAL_CHARGERS =  2000 # 설치할 충전기의 대수 충전기 
TOTAL_CHARGERS = 10000 # 총 충전기 대수
PARENTS_SIZE = round(POPULATION_SIZE/2) # 부모의 수
# 전동화율이 5%임을 가정(원본이 10%임임)
ELECTRIFICATION_RATE = 1.0
TRUCK_NUMBERS = int(7062 * ELECTRIFICATION_RATE) # 전체 화물차 대수 / 7062대는 10%의 전동화율 기준 대수

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


def evaluate_individual(args):
    """
    단일 개체에 대한 적합도를 계산하는 함수입니다.
    """
    
    individual, index, car_paths_df, station_df = args
    #print(f"Evaluating individual {index}")
        
        #아래는 추후 df 길이 문제가 생기면 사용용
        #print(f"station_df 길이: {len(station_df)}")
        
        # 시뮬레이션 파라미터
    unit_minutes = 60
    simulating_hours = 30
    num_trucks = TRUCK_NUMBERS

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
    #print(f"fitness_value for individual {index}: {fitness_value}")
    return (index, fitness_value)


def fitness_func(population, station_df, path_history, pool):
    """시뮬레이션에서 리턴되는 값들을 통해 각 솔루션의 적합도를 평가하는 함수."""
    car_paths_folder = path_for_car  
    car_paths_df = si.load_car_path_df(car_paths_folder, TRUCK_NUMBERS)
    print("차량 경로 파일 길이", len(car_paths_df))
    path_history.append(len(car_paths_df))

    args_list = [
        (individual, idx, car_paths_df, station_df)
        for idx, individual in enumerate(population)
    ]

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 기존 pool 사용
            results = list(pool.imap(evaluate_individual, args_list))
            
            # 결과 정렬
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            sorted_indices = [x[0] for x in sorted_results]
            sorted_population = [population[i] for i in sorted_indices]

            print("\n모든 작업이 완료되었습니다.")

            # 적합도 값 추출
            fitness_values = [
                fitness if fitness is not None else -np.inf for _, fitness in sorted_results  
            ]
            return fitness_values, sorted_population
            
        except Exception as e:
            retry_count += 1
            logging.warning(
                f"An error occurred: {e}. Retry attempt {retry_count}/{max_retries}."
            )
            print(f"An error occurred: {e}. Retry attempt {retry_count}/{max_retries}.")
            time.sleep(5)

    logging.error("Max retries reached. Exiting.")
    return [0] * len(population)



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
        # 부모 4명 선택 (예: 랜덤 선택)
        parents = random.sample(selected_parents, 10)
        
        crossover_points =sorted(random.sample(range(1, num_genes), 9))

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
    
    #총 중복 개체 수 (동일한 해가 여러 번 나온 횟수의 합)
    counter = Counter(tuple(ind) for ind in crossover)
    duplicates_count = sum((count - 1) for count in counter.values() if count > 1)

    # 중복 정보 저장
    if duplicates_count > 0:
        print(f"[중복 개체 확인] 이번 세대에서 총 {duplicates_count}개의 중복 개체가 발견되었습니다.")
        
        duplicate_info = {}
        for idx, ind in enumerate(crossover):
            ind_tuple = tuple(ind)
            if counter[ind_tuple] > 1:
                if ind_tuple not in duplicate_info:
                    duplicate_info[ind_tuple] = []
                duplicate_info[ind_tuple].append(idx)

        for solution, count in counter.items():
            if count > 1:
                indices = [idx for idx, ind in enumerate(crossover) if tuple(ind) == solution]
                duplicate_info_df = pd.concat([duplicate_info_df, pd.DataFrame({
                    'Generation': [generation + 1],
                    'Solution': [str(solution)],
                    'Indices': [str(indices)],
                    'Count': [len(indices)]
                })], ignore_index=True)

        for solution, indices in duplicate_info.items():
            print(f"개체 {solution} 이(가) {len(indices)}회 등장 (인덱스: {indices})")

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
            
            mutated_charger_number = [
                (0, 0, 0.2+round((abs(GENERATIONS-adaptive_constant)/GENERATIONS), 4)*0.3),  # 20% + 세대 수에 따른 가변 확률(최대 30%)
                (1, max, 0.8-round((abs(GENERATIONS-adaptive_constant)/GENERATIONS), 4)*0.3)  # 80% - 세대 수에 따른 가변 확률(최대 30%)
                #총 세대 - 현재 세대 (adaptive constant -> 현재 세대 카운트트)
                 
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

    # 3. 중복 개체 처리
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
    convergence_df = pd.DataFrame({
        'Generation': pd.Series(dtype='int'),
        'Fitness_Mean': pd.Series(dtype='float'),
        'Fitness_Mean_Change': pd.Series(dtype='float'),
        'Charger_Change': pd.Series(dtype='float'),
        'Best_Fitness': pd.Series(dtype='float'),
        'Best_Chargers': pd.Series(dtype='int')
    })

    # 각 세대의 전체 적합도 값 저장용 리스트
    all_fitness_history = []
    min_fitness_history = []
    max_fitness_history = []
    mean_fitness_history = []
    best_fitness_number_of_charger = []
    best_individual_history = []  # 세대별 최고 개체 유전자 저장
    station_file_path = path_for_station
    station_df = si.load_station_df(station_file_path)

    best_individual = None  # 역대 최고 개체 정보 저장 변수 초기화
    last_generation_individuals = None  # 마지막 세대 개체 정보 저장 변수 초기화
    with Pool(processes=cpu_count()) as pool:
        for generation in range(GENERATIONS):
            print(f"\n세대 {generation + 1}/{GENERATIONS} 진행 중...")

            if generation == 0:
                population = station_gene_initial(POPULATION_SIZE, NUM_CANDIDATES, INITIAL_CHARGERS)
            else:
                population = mutated
            if generation == GENERATIONS:
                print("지정된 세대의 연산이 종료되었으므로 계산 결과를 출력합니다")
                break

            # 적합도 계산
            fitness_values, sorted_population = fitness_func(population, station_df, path_history, pool)
            print('적합도 평가 완료')

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

            # 현재 세대의 최고 적합도 및 해당 개체
            current_best_fitness = current_max
            current_best_individual = sorted_population[0]  # 현재 세대 최고 개체
            best_individual_history.append(current_best_individual)  # 최고 개체 저장
            fitness_history.append(current_best_fitness)  # 최고 적합도 저장
            current_best_chargers = np.sum(current_best_individual) # 현재 세대 최고 개체 충전기 수

            print(f"세대 {generation + 1}의 최고 적합도: {current_best_fitness}")


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
                    if abs(charger_change) <= 0.01 and abs(fitness_mean_change) <= 0.005:
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


                if no_improvement_count >= MAX_NO_IMPROVEMENT:  # 연속 MAX_NO_IMPROVEMENT 세대 동안 개선 없으면 종료
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

            if no_improvement_count >= 5 and immigration_count <= MAX_IMMIGRATIONS:
                parents = immigration(parents, NUM_CANDIDATES, INITIAL_CHARGERS, generation)
                immigration_count += 1
                no_improvement_count = 0
                print(f"이민자 연산 실행 ({immigration_count}/{MAX_IMMIGRATIONS})")  
                print("수렴 카운트 초기화")

            # 교차
            children = crossover_elitsm(parents, NUM_CANDIDATES, POPULATION_SIZE, generation)
            print('교차 연산 완료')

            # 변이
            mutated = mutation(children, POPULATION_SIZE, MUTATION_RATE, NUM_CANDIDATES, INITIAL_CHARGERS, generation)
            print('변이 연산 완료')

            

            best_fitness = max(max_fitness_history)  # 최고 적합도 갱신
            # 수렴 정보 저장
           # convergence_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\convergence_info_5%.csv", index=False, mode='w')

            # 중복 정보 저장
           # duplicate_info_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\duplicate_info_5%.csv", index=False, mode='w')
            print("세대별 중복 개체 정보를 duplicate_info.csv 파일로 저장")

            # 역대 최고 개체 및 마지막 세대 개체 정보 저장
            if best_individual is not None:
                best_individual_df = pd.DataFrame([best_individual],
                                                columns=[f"Station_{i + 1}" for i in range(NUM_CANDIDATES)])
                best_individual_df['Fitness'] = best_fitness
                #best_individual_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\best_individual_5%.csv", index=False, mode='w')
                print("역대 최고 개체 정보를 best_individual.csv 파일로 저장")

            if last_generation_individuals is not None:
                last_gen_df = pd.DataFrame(last_generation_individuals,
                                        columns=[f"Station_{i + 1}" for i in range(NUM_CANDIDATES)])
                last_gen_df['Fitness'] = all_fitness_history[-1]  # 마지막 세대의 적합도 리스트 추가
               # last_gen_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\last_generation_5%.csv", index=False, mode='w')
                print("마지막 세대 개체 정보를 last_generation.csv 파일로 저장")

            # 세대별 최고 개체 정보 저장
            best_individuals_df = pd.DataFrame(best_individual_history,
                                            columns=[f"Station_{i + 1}" for i in range(NUM_CANDIDATES)])
            best_individuals_df['Generation'] = np.arange(1, len(best_individual_history) + 1)
            #best_individuals_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\best_individuals_per_generation_5%.csv", index=False, mode='w')
            print("세대별 최고 개체 정보를 best_individuals_per_generation.csv 파일에 저장")

            result_dict = {
                'Generation': np.arange(1, len(min_fitness_history) + 1),
                'Min_fitness': min_fitness_history,
                'Max_fitness': max_fitness_history,
                'Mean_fitness': mean_fitness_history
            }
            result_df = pd.DataFrame(result_dict)
            #result_df.to_csv(r"C:\Users\user\Desktop\화물차 충전소\ga_results.csv", index=False, mode='w')
            print("각 세대별 fitness value를 csv파일로 저장")

    print("\n유전 알고리즘 종료")
    print(f"최종 세대 수: {generation + 1}")
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
    #fig.savefig(r'C:\Users\user\Desktop\화물차 충전소\fitness_history_5%.png')
    #fig2.savefig(r'C:\Users\user\Desktop\화물차 충전소\convergence_info_5%.png')  # 수정: 수렴 정보 그래프 저장
    #fig3.savefig(r'C:\Users\user\Desktop\화물차 충전소\charger_count_5%.png')

    plt.show()  # 마지막에 plt.show() 호출하여 그래프 화면 출력 (선택 사항)

if __name__ == '__main__':
    # 로그 파일 경로
    #log_file_path = os.path.join(r"C:\Users\user\Desktop\화물차 충전소", "genetic_algorithm.log")

    # 로깅 설정
    #logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        #format='%(asctime)s - %(levelname)s - %(message)s',filemode='w')
    genetic_algorithm()
