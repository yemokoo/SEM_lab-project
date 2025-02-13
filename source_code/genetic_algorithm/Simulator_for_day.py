# simulator.py
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict
import gc
from charger import Charger
from station import Station
from truck import Truck
import matplotlib.pyplot as plt
import seaborn as sns

#random.seed(42)
#np.random.seed(42)

class Simulator:
    """
    시뮬레이션 클래스

    Attributes:
        car_paths_df (DataFrame): 차량 경로 데이터 데이터프레임
        station_df (DataFrame): 충전소 정보 데이터프레임
        unit_minutes (int): 시뮬레이션 단위 시간 (분)
        simulating_hours (int): 시뮬레이션 시간 (시간)
        stations (list): 충전소 객체 리스트
        link_id_to_station (dict): 링크 ID를 키로, 충전소 객체를 값으로 갖는 딕셔너리
        trucks (list): 트럭 객체 리스트
        current_time (int): 현재 시뮬레이션 시간 (분)
        truck_results_df (DataFrame): 시뮬레이션 결과를 저장하는 데이터프레임
        number_of_trucks (int): 시뮬레이션에 사용할 트럭 수
        number_of_max_chargers (int): 시뮬레이션에 사용할 최대 충전기 수

    Methods:
        __init__: 시뮬레이터 객체 초기화
        prepare_simulation: 시뮬레이션 환경 설정
        run_simulation: 시뮬레이션 실행
        remove_truck: 시뮬레이터에서 트럭 객체 제거
        analyze_results: 결과 분석
        calculate_OPEX: 모든 충전소의 유지관리 비용과 총 전기 비용 계산
        calculate_CAPEX: 모든 충전소의 CAPEX 계산
        calculate_revenue: 모든 충전소의 총 수익 계산
        calculate_penalty: 배터리 부족으로 정지한 트럭들의 위약금 계산
        calculate_of: OF 값 계산
        load_stations: DataFrame에서 충전소 정보 로드
    """
    def __init__(self, car_paths_df, station_df, unit_minutes, simulating_hours, number_of_trucks, number_of_max_chargers):
        """
        시뮬레이터 객체를 초기화합니다.

        Args:
            car_paths_df (DataFrame): 차량 경로 데이터 데이터프레임
            station_df (DataFrame): 충전소 정보 데이터프레임
            unit_minutes (int): 시뮬레이션 단위 시간 (분)
            simulating_hours (int): 시뮬레이션 시간 (시간)
            number_of_trucks (int): 시뮬레이션에 사용할 트럭 수
            number_of_max_chargers (int): 시뮬레이션에 사용할 최대 충전기 수
        """
        self.car_paths_df = car_paths_df
        self.station_df = station_df
        self.number_of_max_chargers = number_of_max_chargers
        self.unit_minutes = unit_minutes
        self.simulating_hours = simulating_hours
        self.number_of_trucks = number_of_trucks

        # 시뮬레이션 결과 저장 변수 초기화
        self.stations = None
        self.link_id_to_station = None
        self.trucks = None
        self.current_time = None
        self.truck_results_df = pd.DataFrame()

    def prepare_simulation(self):
        """
        차량 경로 데이터를 전처리하고 시뮬레이션 환경을 설정합니다.
        """

        # 데이터 전처리
        self.stations = self.load_stations(self.station_df)
        self.link_id_to_station = {station.link_id: station for station in self.stations}
        station_link_ids = [station.link_id for station in self.stations]  # 충전소가 있는 링크 ID 리스트

        # car_paths_df에 EVCS 열 추가
        self.car_paths_df['EVCS'] = 0
        self.car_paths_df.loc[self.car_paths_df['LINK_ID'].isin(station_link_ids), 'EVCS'] = 1

        # 시뮬레이션 환경 설정
        self.trucks = [] # 트럭 리스트 초기화
        for _, group in self.car_paths_df.groupby('TRIP_ID'):  # self.car_paths_df 직접 사용
            truck = Truck(group, self.simulating_hours, self.link_id_to_station, self, 30)
            self.trucks.append(truck)

        self.current_time = 0
        gc.collect()


    def run_simulation(self):
        """
        시뮬레이션을 실행합니다. 각 시간 단위별로 트럭의 이동 및 충전 상태를 업데이트하고,
        충전소의 대기열 길이와 전체 충전 중인 차량 수를 기록합니다.
        """
        for _ in range(self.simulating_hours * (60 // self.unit_minutes)):
            # 충전소 상태 업데이트
            for station in self.stations:
                station.update_chargers(self.current_time)

            # 충전소 대기열 처리
            for station in self.stations:
                station.process_queue(self.current_time)

            # 트럭 이동
            for truck in self.trucks:
                truck.step(self.current_time)

            # 충전소 대기열 재처리
            for station in self.stations:
                station.process_queue(self.current_time)

            self.current_time += self.unit_minutes

    def remove_truck(self, truck):
        """
        시뮬레이터에서 트럭 객체를 제거합니다.

        Args:
            truck (Truck): 제거할 트럭 객체
        """
        if truck in self.trucks:
            self.trucks.remove(truck)

    def analyze_results(self):
        """
        시뮬레이션 결과를 분석합니다.
        OF 값을 계산하고, 충전소별 총이익, 순이익, 비용, 충전기 당 총이익 및 순이익을 계산하여 그래프로 시각화합니다.
        """

        # 충전소별 정보를 DataFrame으로 저장
        station_data = pd.DataFrame([{
            'station_id': station.station_id,
            'num_of_charger': station.num_of_chargers  # 충전기 개수 정보 추가
        } for station in self.stations])

        self.station_results_df = station_data

        # 배터리 부족으로 정지한 및 시뮬레이팅 시간 초과 트럭 데이터(시간 내에 목적지 도착 실패)만 남긴 DataFrame
        self.failed_trucks_df = self.truck_results_df[
            (self.truck_results_df['destination_reached'] == False) & (
                (self.truck_results_df['stopped_due_to_low_battery'] == True) | (self.truck_results_df['stopped_due_to_simulation_end'] == True)
            )
        ]

        # OF 값 계산 (기존 함수 활용)
        of = self.calculate_of()
        
        return of


    def calculate_OPEX(self, station_df):
        """
        모든 충전소의 유지관리 비용과 총 전기 비용을 계산하여 DataFrame으로 반환합니다.

        Args:
            station_df (DataFrame): 충전소 정보 DataFrame

        Returns:
            DataFrame: 충전소별 유지관리 비용, 전기 비용, 총 OPEX를 포함한 DataFrame
        """
        results = []  # 결과를 저장할 리스트

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환

            # 충전기별 충전량을 모두 더함
            total_charged_energy_station = sum(
                charger.total_charged_energy for charger in self.stations[station_id].chargers
            )

            total_power = sum(
                charger.power for charger in self.stations[station_id].chargers
            )  # 모든 충전기의 power 합산
            energy_price = ((total_power * (2580/30) + total_charged_energy_station * (101.7 + 9 + 5))*1.132
            )  # 전기요금계: KW 당 기본 요금 + 전력량요금 + 기후환경요금 + 연료비조정액
        # 총 전기 비용: 전기요금계 + 부가가치세 + 전력산업기반요금

            labor_cost = self.stations[station_id].num_of_chargers * (6250)  # 인건비: 충전기 당 인건비
            maintenance_cost = self.stations[station_id].num_of_chargers * (800)  # 유지관리 비용: 충전기 당 유지 관리 비용
            opex = (
                labor_cost + maintenance_cost + energy_price
            )  # 충전소의 유지관리 비용과 전기 비용

            # 결과를 딕셔너리로 저장
            results.append({
                'station_id': station_id,
                'labor_cost': labor_cost,
                'maintenance_cost': maintenance_cost,
                'energy_price': energy_price,
                'opex': opex
            })

        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame(results)
        return result_df  # 충전소별 유지관리 비용, 전기 비용, 총 OPEX를 포함한 DataFrame 반환


    def calculate_CAPEX(self, station_df):
        """
        모든 충전소의 CAPEX를 계산하여 DataFrame으로 반환합니다.

        Args:
            station_df (DataFrame): 충전소 정보 DataFrame

        Returns:
            DataFrame: 충전소별 CAPEX 상세 내역을 포함한 DataFrame (station_id, charger_cost, kepco_cost, construction_cost, capex)
        """
        results = []  # 결과를 저장할 리스트

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환
            num_chargers = self.stations[station_id].num_of_chargers  # 충전소의 충전기 개수

            # CAPEX 계산
            if num_chargers == 0:  # 충전기 개수가 0이면 해당 충전소 CAPEX는 0
                charger_cost = 0
                kepco_cost = 0
                construction_cost = 0
                station_capex = 0
            else:
                # CAPEX 계산 (충전기 개수가 0보다 클 때만 계산)
                charger_cost = ((80000000) * num_chargers) / (365*5)   # 충전기 비용
                kepco_cost = 50000 * num_chargers * 200 / (365*5)   # 한전 불입금
                construction_cost = 1868123 * 50 * num_chargers / (365*5)  # 충전소 건설 비용
                station_capex = (
                    charger_cost
                    + kepco_cost
                    + construction_cost
                )  # 충전소 1개의 CAPEX

            # 결과를 딕셔너리로 저장
            results.append({
                'station_id': station_id,
                'charger_cost': charger_cost,
                'kepco_cost': kepco_cost,
                'construction_cost': construction_cost,
                'capex': station_capex
            })

        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame(results)
        return result_df  # 충전소별 CAPEX 상세 내역을 포함한 DataFrame 반환

    def calculate_revenue(self, station_df):
        """
        모든 충전소의 충전 요금을 계산하여 DataFrame으로 반환합니다.

        Args:
            station_df (DataFrame): 충전소 정보 DataFrame

        Returns:
            DataFrame: 충전소별 수익을 포함한 DataFrame (station_id, revenue)
        """

        results = []  # 결과를 저장할 리스트

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환
            revenue = 0  # 수익 초기화
            for charger in self.stations[station_id].chargers:  # 충전기별 수익 계산
                revenue += charger.rate * charger.total_charged_energy  # 충전기별 rate와 total_charged_energy를 곱하여 수익 계산

            # 결과를 딕셔너리로 저장
            results.append({
                'station_id': station_id,
                'revenue': revenue
            })

        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame(results)

        return result_df  # 충전소별 수익을 포함한 DataFrame 반환

    def calculate_penalty(self, failed_trucks_df, station_df):
        """
        배터리 부족으로 정지한 트럭들의 위약금과 충전기 관련 위약금을 계산하여 DataFrame으로 반환합니다.

        Args:
            failed_trucks_df (DataFrame): 배터리 부족으로 정지한 트럭 정보 DataFrame

        Returns:
            DataFrame: 위약금 정보를 포함한 DataFrame (truck_penalty, charger_penalty, total_penalty) - 1개의 행
        """
        truck_penalty = 0  # 트럭 위약금 초기화
        charger_penalty = 0 # 충전기 패널티 초기화
        number_of_charges = 0 # number_of_charges 변수 추가 및 초기화

        for idx, row in failed_trucks_df.iterrows():  # DataFrame의 각 행을 순회
            distance = row['total_distance'] / 2  # 이동 거리의 절반을 위약금 계산에 사용

            # 랜덤으로 컨테이너 종류 선택
            if random.choice([True, False]):  # 50% 확률로 40FT 또는 20FT 선택
                penalty = 136395.90 + 3221.87 * distance - 2.72 * distance**2 #40ft
            else:  # 20FT 컨테이너
                penalty = 121628.18 + 2765.50 * distance - 2.00 * distance**2 #20ft

            truck_penalty += penalty  # 위약금을 총 위약금에 누적

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환
            number_of_charges += self.stations[station_id].num_of_chargers  # 충전소의 충전기 개수

        if number_of_charges > self.number_of_max_chargers:
            charger_penalty = 80000000 * (number_of_charges - self.number_of_max_chargers) # 초과 설치 페널티
        else:
            charger_penalty = 0

        total_penalty = truck_penalty + charger_penalty

        # 결과를 딕셔너리로 저장 (1개의 행)
        results = {
            'truck_penalty': truck_penalty,
            'charger_penalty': charger_penalty,
            'total_penalty': total_penalty
        }

        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame([results])  # 딕셔너리를 리스트로 감싸서 1개의 행으로 만듦

        return result_df  # 위약금 정보를 포함한 DataFrame 반환

    def calculate_of(self):
        """
        OF 값을 계산하고, 충전소별 CAPEX, OPEX, REVENUE 및 페널티를 시각화합니다.
        또한, 순이익 그래프를 추가합니다.
        """
        revenue_df = self.calculate_revenue(self.station_results_df)  # 모든 충전소의 총 수익
        opex_df = self.calculate_OPEX(self.station_results_df)  # 모든 충전소의 OPEX
        capex_df = self.calculate_CAPEX(self.station_results_df)  # 모든 충전소의 CAPEX
        penalty_df = self.calculate_penalty(self.failed_trucks_df, self.station_results_df)  # 트럭 정지 페널티 및 충전기 초과 설치 페널티

        # 각 DataFrame에서 필요한 열을 추출하고 station_id를 기준으로 병합합니다.
        revenue_df = revenue_df[['station_id', 'revenue']]
        opex_df = opex_df[['station_id', 'opex']]
        capex_df = capex_df[['station_id', 'capex']]

        # station_id를 기준으로 DataFrame 병합
        merged_df = pd.merge(revenue_df, opex_df, on='station_id', how='outer')
        merged_df = pd.merge(merged_df, capex_df, on='station_id', how='outer')
        merged_df.fillna(0, inplace=True)  # 결측값은 0으로 채움

        # 순이익 계산
        merged_df['net_profit'] = merged_df['revenue'] - merged_df['opex'] - merged_df['capex']

        # 전체 합계 계산
        revenue = merged_df['revenue'].sum()
        opex = merged_df['opex'].sum()
        capex = merged_df['capex'].sum()
        total_penalty = penalty_df['total_penalty'].sum()
        truck_penalty = penalty_df['truck_penalty'].sum()
        charger_penalty = penalty_df['charger_penalty'].sum()

        revenue = round(revenue)
        opex = round(opex)
        capex = round(capex)
        total_penalty = round(total_penalty) 
        truck_penalty = round(truck_penalty)
        charger_penalty = round(charger_penalty)

        of = revenue - opex - capex - total_penalty  # OF 값 계산

        of = round(of)

        print(f"Revenue: {revenue}, OPEX: {opex}, CAPEX: {capex}, Penalty: {total_penalty}, OF: {of}")
        
        return of

    def load_stations(self, df):
        """
        DataFrame에서 충전소 정보를 읽어와서 충전소 객체 리스트를 생성하는 함수입니다.

        Args:
            df (DataFrame): 충전소 정보 DataFrame (첫 번째 열: link_id, 두 번째 열: num_of_charger)

        Returns:
            list: 충전소 객체 리스트
        """

        stations = []  # 충전소 객체를 저장할 리스트
        for idx, row in df.iterrows():  # DataFrame의 각 행을 순회
            num_of_chargers = int(row['num_of_charger'])  # 충전기 개수

            charger_specs = []  # 충전기 사양을 저장할 리스트
            for _ in range(num_of_chargers):  # 충전기 개수만큼 반복
                charger_specs.append({'power': 200, 'rate': 560})  # 충전기 사양 추가 (200kW, 560원/kWh)
            station = Station(  # 충전소 객체 생성
                station_id=idx,  # 충전소 ID (행 인덱스 사용)
                link_id=row['link_id'],  # 충전소가 위치한 링크 ID
                num_of_chargers=num_of_chargers,  # 총 충전기 개수
                charger_specs=charger_specs  # 충전기 사양 리스트
            )
            stations.append(station)  # 생성된 충전소 객체를 리스트에 추가
        return stations  # 충전소 객체 리스트 반환
    
def run_simulation(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers):
    """
    시뮬레이션을 실행하고 실행 시간을 반환합니다.
    """

    # 시뮬레이션 객체 생성
    sim = Simulator(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers)

    # 시뮬레이션 준비
    sim.prepare_simulation()

    # 시뮬레이션 실행
    sim.run_simulation()

    # 시뮬레이션 적합도 계산
    of = sim.analyze_results()
    
    return of

def load_car_path_df(car_paths_folder, number_of_trucks): 
    """
    차량 경로 데이터를 로드하는 함수

    Args:
        car_paths_folder (str): 차량 경로 데이터가 있는 폴더 경로
        number_of_trucks (int): 시뮬레이션에 사용할 트럭 수
        station_link_ids (list): 충전소가 위치한 링크 ID 리스트

    Returns:
        DataFrame (car_paths_df): 전처리 및 필터링된 차량 경로 데이터 DataFrame
    """
    
    #random.seed(100)

    # 차량 경로 데이터 로드 및 전처리
    subfolders = [d for d in os.listdir(car_paths_folder) if os.path.isdir(os.path.join(car_paths_folder, d))]
    random_subfolder = random.choice(subfolders)
    car_paths_list = []

    date_str = random_subfolder
    date = date_str.split('=')[-1]
    print(f"Processing data for date: {date}")

    for file in os.listdir(os.path.join(car_paths_folder, random_subfolder)):
        if file.endswith(".csv"):
            file_path = os.path.join(car_paths_folder, random_subfolder, file)
            car_paths_list.append(pd.read_csv(file_path))

    car_paths_df = pd.concat(car_paths_list, ignore_index=True)

    car_paths_df['DATETIME'] = pd.to_datetime(car_paths_df['DATETIME'], format='%H:%M').dt.time

    car_paths_df['START_TIME_MINUTES'] = car_paths_df.groupby('TRIP_ID')['DATETIME'].transform('first').apply(
        lambda x: (x.hour * 60 + x.minute)
    )

    grouped_data = car_paths_df.groupby('TRIP_ID')
    grouped_paths = [group for _, group in grouped_data]

    random.shuffle(grouped_paths)
    selected_groups = grouped_paths[:min(number_of_trucks, len(grouped_paths))]

    # area_id 기반 샘플링
    area_paths = defaultdict(list)
    for group in grouped_paths:
        area_id = group['AREA_ID'].iloc[0]
        area_paths[area_id].append(group)

    for area_id, paths in area_paths.items():
        paths.sort(key=lambda x: x['CUMULATIVE_LINK_LENGTH'].iloc[-1], reverse=True)

    area_selected_groups = []
    for area_id, paths in area_paths.items():
        # map 함수를 사용하여 경로 복사 및 TRIP_ID 수정
        processed_paths = list(map(lambda path: path.copy().assign(TRIP_ID=path['TRIP_ID'].astype(str) + "_AREA"), 
                                   paths[:int(number_of_trucks * 0.01)]))
        area_selected_groups.extend(processed_paths)

    selected_groups.extend(area_selected_groups)

    # 최종 결과 DataFrame 생성
    filtered_car_paths_df = pd.concat(selected_groups, ignore_index=True)

    return filtered_car_paths_df

def load_station_df(station_file_path):
    # 충전소 데이터 로드 및 전처리
    station_df = pd.read_csv(station_file_path, sep=',')  # CSV 파일을 읽어와서 pandas DataFrame으로 저장

    # 열 이름의 앞뒤 공백 제거 및 소문자로 변환
    station_df.columns = station_df.columns.str.strip()
    station_df.columns = station_df.columns.str.lower()
    return station_df # 충전소 정보 df 반환


# 메인 함수 (스크립트 실행 시 호출)
if __name__ == '__main__':
    # 파일 경로 설정
    car_paths_folder = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\analyzed_paths_for_simulator(DAY)"
    station_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\candidate(debug).csv"
    
    simuating_hours = 30
    unit_time = 60
    number_of_trucks = 5863
    number_of_charges = 2000

    car_paths_df = load_car_path_df(car_paths_folder, number_of_trucks)
    station_df = load_station_df(station_file_path)

    gc.collect()
    
    run_simulation(car_paths_df, station_df, unit_time, simuating_hours, number_of_trucks, number_of_charges)
