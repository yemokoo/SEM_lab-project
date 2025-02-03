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

random.seed(42)
np.random.seed(42)
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

        # 배터리 부족으로 정지한 트럭 데이터만 남긴 DataFrame
        self.failed_trucks_df = self.truck_results_df[
            self.truck_results_df['stopped_due_to_low_battery'] == True
        ]

        # OF 값 계산 (기존 함수 활용)
        of = self.calculate_of()
        of = round(of/10000000, 6)  # OF 값을 소수점 아래 6자리까지 반올림

        '''
        print(f"OF: {of:.6f} (단위: 천만원)")  # OF 값 출력 (천만원 단위)

        # 충전소별 수익, OPEX, CAPEX 계산
        self.station_results_df['revenue'] = self.station_results_df.apply(
            lambda row: sum(charger.rate * charger.total_charged_energy for charger in self.stations[int(row['station_id'])].chargers), axis=1
        )
        self.station_results_df['opex'] = self.station_results_df.apply(
            lambda row: self.calculate_OPEX(pd.DataFrame([row])), axis=1 # calculate_OPEX 함수는 DataFrame을 인자로 받으므로, row를 DataFrame으로 변환하여 전달
        )
        self.station_results_df['capex'] = self.station_results_df.apply(
            lambda row: self.calculate_CAPEX(pd.DataFrame([row])), axis=1 # calculate_CAPEX 함수는 DataFrame을 인자로 받으므로, row를 DataFrame으로 변환하여 전달
        )

        # 충전소별 총이익, 순이익 계산
        self.station_results_df['total_profit'] = self.station_results_df['revenue']
        self.station_results_df['net_profit'] = self.station_results_df['revenue'] - self.station_results_df['opex'] - self.station_results_df['capex'] / (365 * 5)

        # 충전기 당 총이익, 순이익 계산
        self.station_results_df['profit_per_charger'] = self.station_results_df['total_profit'] / self.station_results_df['num_of_charger']
        self.station_results_df['net_profit_per_charger'] = self.station_results_df['net_profit'] / self.station_results_df['num_of_charger']

        self.station_results_df['station_id'] = self.station_results_df['station_id'].astype(int)

        # 그래프 그리기 및 저장
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        # 충전소별 총이익
        plt.figure(figsize=(12, 8))
        sns.barplot(x='station_id', y='total_profit', data=self.station_results_df)
        plt.title('Total Profit per Station')
        plt.xlabel('Station ID')
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.tight_layout()
        plt.savefig('total_profit_per_station.png')

        # 충전소별 순이익
        plt.figure(figsize=(12, 8))
        sns.barplot(x='station_id', y='net_profit', data=self.station_results_df)
        plt.title('Net Profit per Station')
        plt.xlabel('Station ID')
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.tight_layout()
        plt.savefig('net_profit_per_station.png')

        # 충전소별 비용 (OPEX + 일일 CAPEX)
        self.station_results_df['total_cost'] = self.station_results_df['opex'] + self.station_results_df['capex'] / (365 * 5)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='station_id', y='total_cost', data=self.station_results_df)
        plt.title('Total Cost per Station (OPEX + Daily CAPEX)')
        plt.xlabel('Station ID')
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.tight_layout()
        plt.savefig('total_cost_per_station.png')

        # 충전기 당 총이익
        plt.figure(figsize=(12, 8))
        sns.barplot(x='station_id', y='profit_per_charger', data=self.station_results_df)
        plt.title('Total Profit per Charger')
        plt.xlabel('Station ID')
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.tight_layout()
        plt.savefig('total_profit_per_charger.png')

        # 충전기 당 순이익
        plt.figure(figsize=(12, 8))
        sns.barplot(x='station_id', y='net_profit_per_charger', data=self.station_results_df)
        plt.title('Net Profit per Charger')
        plt.xlabel('Station ID')
        plt.xticks([0, 50, 100, 150, 200, 250])
        plt.tight_layout()
        plt.savefig('net_profit_per_charger.png')
        '''
        
        return of
    
    def calculate_OPEX(self, station_df):
        """
        모든 충전소의 유지관리 비용과 총 전기 비용을 계산합니다.

        Args:
            station_df (DataFrame): 충전소 정보 DataFrame

        Returns:
            float: 모든 충전소의 유지관리 비용과 총 전기 비용의 합
        """
        total_opex = 0  # 총 OPEX 초기화

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환

            # 충전기별 충전량을 모두 더함
            total_charged_energy_station = sum(
                charger.total_charged_energy for charger in self.stations[station_id].chargers
            )

            total_power = sum(
                charger.power for charger in self.stations[station_id].chargers
            )  # 모든 충전기의 power 합산
            basis_price = total_power * (7220 / 30) + total_charged_energy_station * (
                137.1 + 9 + 5
            )  # 전기요금계: KW 당 기본 요금 + 전력량요금 + 기후환경요금 + 연료비조정액
            total_energy_price = basis_price * (
                1.132
            )  # 총 전기 비용: 전기요금계 + 부가가치세 + 전력산업기반요금

            variable_cost = self.stations[
                station_id
            ].num_of_chargers * (
                33375 / 30
            )  # 가변 비용: 충전기 당 유지 관리 비용

            total_opex += (
                variable_cost + total_energy_price
            )  # 충전소의 유지관리 비용과 전기 비용을 더하여 총 OPEX에 누적

        return total_opex  # 모든 충전소의 유지관리 비용과 총 전기 비용의 합 반환

    def calculate_CAPEX(self, station_df):
        """
    
        모든 충전소의 CAPEX를 계산합니다.

         Args:
                station_df (DataFrame): 충전소 정보 DataFrame

         Returns:
            float: 모든 충전소의 CAPEX 합
        """    
        total_capex = 0  # 총 CAPEX 초기화

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환
            num_chargers = self.stations[station_id].num_of_chargers  # 충전소의 충전기 개수

            # CAPEX 계산
            if num_chargers == 0:  # 충전기 개수가 0이면 해당 충전소 CAPEX는 0
                station_capex = 0  # 해당 충전소의 CAPEX를 0으로 설정
            else:
                # CAPEX 계산 (충전기 개수가 0보다 클 때만 계산)
                charger_cost = 80000000 * num_chargers  # 충전기 비용
                kepco_cost = 10000000  # 한전 불입금
                construction_cost = 200000000  # 충전소 건설 비용
                installation_cost = 10000000 * num_chargers  # 충전기 설치 비용

                station_capex = (
                    charger_cost
                    + kepco_cost
                    + construction_cost
                    + installation_cost
                    )  # 충전소 1개의 CAPEX

            total_capex += station_capex  # 총 CAPEX에 누적

            return total_capex  # 모든 충전소의 CAPEX 합 반환

    def calculate_revenue(self, station_df):
        """
        모든 충전소의 충전 요금을 계산합니다.

        Args:
            station_df (DataFrame): 충전소 정보 DataFrame

        Returns:
            float: 모든 충전소의 총 수익
        """

        station_df['revenue'] = 0.0  # revenue 열 초기화

        for idx, row in station_df.iterrows():  # DataFrame의 각 행을 순회
            station_id = int(row['station_id'])  # station_id를 정수형으로 변환
            income = 0  # 수익 초기화
            for charger in self.stations[station_id].chargers:  # 충전기별 수익 계산
                income += charger.rate * charger.total_charged_energy  # 충전기별 rate와 total_charged_energy를 곱하여 수익 계산
            station_df.loc[idx, 'revenue'] = income  # 계산된 수익을 DataFrame에 저장

        total_revenue = station_df['revenue'].sum()  # 모든 충전소의 총 수익 반환

        return total_revenue

    def calculate_penalty(self, failed_trucks_df):
        """
        배터리 부족으로 정지한 트럭들의 위약금을 계산합니다.

        Args:
            failed_trucks_df (DataFrame): 배터리 부족으로 정지한 트럭 정보 DataFrame

        Returns:
            float: 모든 트럭의 위약금 합
        """
        total_penalty = 0  # 총 위약금 초기화

        for idx, row in failed_trucks_df.iterrows():  # DataFrame의 각 행을 순회
            distance = row['total_distance'] / 2  # 이동 거리의 절반을 위약금 계산에 사용

            # 랜덤으로 컨테이너 종류 선택
            if random.choice([True, False]):  # 50% 확률로 40FT 또는 20FT 선택
                penalty = 136395.90 + 3221.87 * distance - 2.72 * distance**2
            else:  # 20FT 컨테이너
                penalty = 121628.18 + 2765.50 * distance - 2.00 * distance**2

            total_penalty += penalty  # 위약금을 총 위약금에 누적

        return total_penalty  # 모든 트럭의 위약금 합 반환

    def calculate_of(self):
        """
        OF 값을 계산합니다.
        """
        total_revenue = self.calculate_revenue(self.station_results_df)  # 모든 충전소의 총 수익
        total_opex = self.calculate_OPEX(self.station_results_df)  # 모든 충전소의 OPEX
        total_capex = self.calculate_CAPEX(self.station_results_df)  # 모든 충전소의 CAPEX
        total_penalty = self.calculate_penalty(self.failed_trucks_df)  # 모든 트럭의 위약금 합

        daily_capex = total_capex / (365 * 5)  # 5년 일일 CAPEX

        # 예산 초과분 계산
        budget_excess = max(0, total_capex - 1500000000000)  # 1조 5천억 초과분
        daily_budget_excess = 2*budget_excess / (365 * 5)  # 5년 일일 예산 초과분

        of = total_revenue - total_opex - daily_capex - total_penalty - daily_budget_excess

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
                charger_specs.append({'power': 200, 'rate': 600})  # 충전기 사양 추가 (200kW, 600원/kWh)
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
    car_paths_folder = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\debug"
    station_file_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\data analysis\station_for_simulator(debug).csv"
    
    simuating_hours = 30
    unit_time = 60
    number_of_trucks = 1000
    number_of_charges = 2000

    car_paths_df = load_car_path_df(car_paths_folder, number_of_trucks)
    station_df = load_station_df(station_file_path)

    gc.collect()
    
    run_simulation(car_paths_df, station_df, unit_time, simuating_hours, number_of_trucks, number_of_charges)
