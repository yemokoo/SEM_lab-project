from datetime import datetime
import random
import warnings
from matplotlib import pyplot as plt, ticker
import pandas as pd
import numpy as np
import os
import gc
import time
from charger import Charger
from station import Station # Assuming these exist
from truck import Truck   
import pyarrow.parquet as pq
import pyarrow as pa

seed = 42
random.seed(seed)
np.random.seed(seed)

# 특정 FutureWarning 메시지를 기준으로 무시합니다.
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
    category=FutureWarning
)


class Simulator:
    """
    시뮬레이션 클래스 (최적화 및 로직 개선)
    """
    def __init__(self, car_paths_df, station_df, unit_minutes, simulating_hours, number_of_trucks, number_of_max_chargers):
        """
        시뮬레이터 객체를 초기화합니다.
        입력 데이터는 유효하다고 가정합니다.
        """
        self.car_paths_df = car_paths_df
        self.station_df = station_df
        self.number_of_max_chargers = number_of_max_chargers
        self.unit_minutes = unit_minutes
        self.simulating_hours = simulating_hours
        self.number_of_trucks = number_of_trucks # 초기 목표 트럭 수

        self.stations = []
        self.link_id_to_station = {}
        self.trucks = [] # 활성 트럭 리스트
        self.current_time = 0
        # 결과 저장을 위한 DataFrame 초기화 (컬럼 정의)
        self.truck_results_df = pd.DataFrame(columns=[
            'truck_id', 'final_SOC', 'destination_reached',
            'stopped_due_to_low_battery', 'stopped_due_to_simulation_end',
            'total_distance_planned', 'traveled_distance_at_last_stop85'
        ])
        self.station_results_df = None
        self.failed_trucks_df = None


    def prepare_simulation(self):
        """
        시뮬레이션 환경을 설정합니다.
        """
        self.stations = self.load_stations(self.station_df)
        self.link_id_to_station = {station.link_id: station for station in self.stations}
        station_link_ids = set(self.link_id_to_station.keys())

        # EVCS 컬럼 추가 또는 업데이트
        if 'EVCS' not in self.car_paths_df.columns:
             self.car_paths_df['EVCS'] = 0
        self.car_paths_df['EVCS'] = np.where(self.car_paths_df['LINK_ID'].isin(station_link_ids), 1, 0)

        # 트럭 객체 생성
        self.trucks = []
        truck_creation_start_time = time.time()
        for obu_id, group in self.car_paths_df.groupby('OBU_ID'):
            required_cols = ['OBU_ID', 'LINK_ID', 'START_TIME_MINUTES', 'EVCS',
                             'CUMULATIVE_LINK_LENGTH', 'CUMULATIVE_DRIVING_TIME_MINUTES',
                             'STOPPING_TIME']
            missing_cols = [col for col in required_cols if col not in group.columns]
            if missing_cols:
                continue # 필수 컬럼 없으면 건너뛰기
            # Truck 객체 생성 시 오류 없다고 가정
            truck = Truck(group, self.simulating_hours, self.link_id_to_station, self, 10)
            self.trucks.append(truck)


        self.current_time = 0
        gc.collect()


    def run_simulation(self):
        """
        시뮬레이션을 실행합니다.
        """
        total_steps = self.simulating_hours * (60 // self.unit_minutes)

        for step_num in range(total_steps):
            step_start_time = time.time()

            # 1. 충전소 상태 업데이트
            for station in self.stations:
                station.update_chargers(self.current_time)

            # 2. 충전소 대기열 처리
            for station in self.stations:
                station.process_queue(self.current_time)

            # 3. 트럭 행동 결정 및 상태 업데이트
            # 리스트 복사본 사용 (반복 중 제거 대비)
            current_trucks_in_step = list(self.trucks) # 매 스텝마다 현재 트럭 리스트의 복사본 생성
            
            active_truck_count_this_step = 0
            
            for truck in current_trucks_in_step:
                # current_trucks_in_step로 순회 중 self.trucks에서 제거되었을 수 있으므로,
                # 실제 self.trucks에 아직 존재하는지, 그리고 상태가 stopped가 아닌지 확인
                if truck in self.trucks and truck.status != 'stopped':
                    if self.current_time >= truck.next_activation_time:
                        active_truck_count_this_step += 1
                        truck.step(self.current_time)

            # 시간 증가
            self.current_time += self.unit_minutes

        # --- 최종 정리 단계 ---
        # self.trucks 리스트는 Truck.stop()에 의해 변경되므로 복사본 사용
        final_cleanup_trucks = list(self.trucks)
        cleaned_up_count = 0
        if final_cleanup_trucks:
            for truck_to_cleanup in final_cleanup_trucks:
                # truck.stop() 내부에서 self.status를 'stopped'로 먼저 바꾸므로 중복 호출은 방지됨.
                if truck_to_cleanup.status != 'stopped':
                    truck_to_cleanup.stop() 
                    cleaned_up_count +=1



    def remove_truck(self, truck):
        """
        시뮬레이터의 활성 트럭 리스트에서 특정 트럭 객체를 제거합니다.
        """
        if truck in self.trucks:
            self.trucks.remove(truck)


    def analyze_results(self):
        """
        시뮬레이션 결과를 분석하고 최종 OF(Objective Function) 값을 계산합니다.
        """
        # 각 충전소의 운영 결과를 집계하여 station_results_df를 생성합니다.
        station_data = [
            {'station_id': station.station_id, 
             'link_id': station.link_id,
             'num_of_charger': station.num_of_chargers,
             'total_charged_energy_kWh': sum(c.total_charged_energy for c in station.chargers),
             'total_charging_events': sum(c.charging_events_count for c in station.chargers),
             'avg_queue_length': np.mean(station.queue_history) if station.queue_history else 0,
             'max_queue_length': np.max(station.queue_history) if station.queue_history else 0,
             'avg_waiting_time_min': np.mean(station.waiting_times) if station.waiting_times else 0,
            }
            for station in self.stations
        ]
        self.station_results_df = pd.DataFrame(station_data)

        # 시뮬레이션 결과가 담긴 truck_results_df를 성공과 실패 그룹으로 분리합니다.
        if self.truck_results_df is None or self.truck_results_df.empty:
            # 결과가 없는 경우, 분석을 위해 빈 데이터프레임을 생성합니다.
            self.successful_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else [])
            self.failed_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else [])
        else:
            # 성공 트럭: destination_reached 플래그가 True인 경우
            self.successful_trucks_df = self.truck_results_df[
                self.truck_results_df['destination_reached'] == True
            ].copy()

            # 실패 트럭: destination_reached가 False이고, 배터리 부족 또는 시뮬레이션 시간 초과로 중단된 경우
            self.failed_trucks_df = self.truck_results_df[
                (self.truck_results_df['destination_reached'] == False) &
                (
                    (self.truck_results_df['stopped_due_to_low_battery'] == True) |
                    (self.truck_results_df['stopped_due_to_simulation_end'] == True)
                )
            ].copy()
        
        # 재무 분석 및 OF 계산을 수행합니다.
        of = self.calculate_of()
        return of


    def calculate_OPEX(self, station_df):
        """
        모든 충전소의 OPEX(운영 비용)를 계산합니다.
        """
        opex_results = []
        base_rate_per_kw = 2580 / 30
        energy_rate_per_kwh = 101.7 + 9 + 5
        vat_and_fund_multiplier = 1.132
        labor_cost_per_charger = 6250
        maint_cost_per_charger = 800

        for station in self.stations:
            total_charged_energy_station = sum(c.total_charged_energy for c in station.chargers)
            total_power = sum(c.power for c in station.chargers) 
            
            energy_price = ((total_power * base_rate_per_kw) + (total_charged_energy_station * energy_rate_per_kwh)) * vat_and_fund_multiplier
            labor_cost = station.num_of_chargers * labor_cost_per_charger
            maintenance_cost = station.num_of_chargers * maint_cost_per_charger
            opex = labor_cost + maintenance_cost + energy_price
            
            opex_results.append({
                'station_id': station.station_id, 
                'labor_cost': labor_cost, 
                'maintenance_cost': maintenance_cost, 
                'energy_price': energy_price, 
                'opex': opex
            })
        result_df = pd.DataFrame(opex_results)
        return result_df


    def calculate_CAPEX(self, station_df):
        """
        모든 충전소의 CAPEX(자본 비용)를 계산합니다. (일일 비용 기준)
        """
        capex_results = []
        LIFESPAN_YEARS = 5
        DAYS_PER_YEAR = 365
        daily_divider = LIFESPAN_YEARS * DAYS_PER_YEAR
        charger_cost_per_unit = 80000000
        kepco_cost_per_kw = 50000
        charger_power_kw = 200 
        construction_cost_multiplier = 1868123 * 50 

        for station in self.stations:
            num_chargers = station.num_of_chargers
            if num_chargers == 0:
                charger_cost, kepco_cost, construction_cost, station_capex = 0, 0, 0, 0
            else:
                charger_cost = (charger_cost_per_unit * num_chargers) / daily_divider
                kepco_cost = (kepco_cost_per_kw * num_chargers * charger_power_kw) / daily_divider
                construction_cost = (construction_cost_multiplier * num_chargers) / daily_divider
                station_capex = charger_cost + kepco_cost + construction_cost
            
            capex_results.append({
                'station_id': station.station_id, 
                'charger_cost': charger_cost, 
                'kepco_cost': kepco_cost, 
                'construction_cost': construction_cost, 
                'capex': station_capex
            })
        result_df = pd.DataFrame(capex_results)
        return result_df


    def calculate_revenue(self, station_df):
        """
        모든 충전소의 수익을 계산합니다.
        """
        revenue_results = []
        for station in self.stations:
            revenue = sum(charger.rate * charger.total_charged_energy 
                          for charger in station.chargers)
            revenue_results.append({'station_id': station.station_id, 'revenue': revenue})
        result_df = pd.DataFrame(revenue_results)
        return result_df


    def calculate_penalty(self, successful_trucks_df, failed_trucks_df, station_df):
        """
        시뮬레이션 결과에 기반하여 다양한 유형의 페널티를 계산합니다.
        - 미도착 트럭 페널티, 도착 트럭 지연 페널티, 충전기 수 초과 페널티, 대기 시간 페널티
        """
        # --- 1. 트럭 관련 페널티 계산 ---
        
        # 1-1. 목적지 미도착 트럭에 대한 페널티
        failed_truck_penalty = 0.0
        if failed_trucks_df is not None and not failed_trucks_df.empty:
            # 미도착 지점까지의 남은 거리에 비례하여 페널티를 계산합니다.
            planned_dist = failed_trucks_df['total_distance_planned']
            last_stop_dist = failed_trucks_df['traveled_distance_at_last_stop85'].fillna(0)
            distance_for_penalty = np.where(
                last_stop_dist <= 0,
                planned_dist / 2,
                np.maximum(0, planned_dist - last_stop_dist) / 2
            )
            choice = np.random.choice([True, False], size=len(failed_trucks_df))
            penalty = np.where(
                choice,
                136395.90 + 3221.87 * distance_for_penalty - 2.72 * distance_for_penalty**2,
                121628.18 + 2765.50 * distance_for_penalty - 2.00 * distance_for_penalty**2
            )
            failed_truck_penalty = np.maximum(0, penalty).sum()

        # 1-2. 목적지에 도착했으나 지연된 트럭에 대한 페널티
        late_truck_penalty = 0.0
        if successful_trucks_df is not None and not successful_trucks_df.empty:
            required_cols = ['starting_time', 'reaching_time', 'actual_reached_time', 'total_distance_planned']
            if all(col in successful_trucks_df.columns for col in required_cols):
                # 운행 거리(km)에 비례한 충전 마진(분)을 계산합니다.
                # (100km당 36분의 충전 시간을 기준으로 함)
                charging_margin = (successful_trucks_df['total_distance_planned'] / 100.0) * 36.0
                
                # 마진을 포함한 허용 도착 시간과 실제 도착 시간의 차이를 통해 지연 시간(분)을 계산합니다.
                allowed_arrival_time = successful_trucks_df['reaching_time'] + charging_margin
                delay_minutes = successful_trucks_df['actual_reached_time'] - allowed_arrival_time

                # 페널티의 기준이 되는 '베이스 위약금'을 전체 운행 거리를 기반으로 산정합니다.
                distance_for_penalty = successful_trucks_df['total_distance_planned']
                choice = np.random.choice([True, False], size=len(successful_trucks_df))
                base_penalty_per_truck = np.where(
                    choice,
                    136395.90 + 3221.87 * distance_for_penalty - 2.72 * distance_for_penalty**2,
                    121628.18 + 2765.50 * distance_for_penalty - 2.00 * distance_for_penalty**2
                )
                base_penalty_per_truck = np.maximum(0, base_penalty_per_truck)

                # 지연 시간에 따라 차등적인 페널티 비율(10%, 20%)을 적용합니다.
                conditions = [
                    delay_minutes > 120,                          # 2시간 초과 지연
                    (delay_minutes > 60) & (delay_minutes <= 120) # 1시간 초과 2시간 이하 지연
                ]
                penalty_rates = [0.20, 0.10]
                actual_penalty_per_truck = np.select(conditions, 
                                                     [base_penalty_per_truck * rate for rate in penalty_rates], 
                                                     default=0)
                late_truck_penalty = actual_penalty_per_truck.sum()

        total_truck_penalty = failed_truck_penalty + late_truck_penalty

        # --- 2. 최대 충전기 설치 가능 대수 초과에 대한 페널티 ---
        charger_penalty = 0.0
        number_of_total_chargers = sum(station.num_of_chargers for station in self.stations)
        if number_of_total_chargers > self.number_of_max_chargers:
            charger_cost_per_unit = 80000000
            charger_penalty = float(charger_cost_per_unit * (number_of_total_chargers - self.number_of_max_chargers))

        # --- 3. 충전소에서 발생한 총 대기 시간에 대한 페널티 (기회비용) ---
        HOURLY_REVENUE_VALUE = 11000000 / (10.9 * 22.4)
        station_waiting_penalties = {}
        for station in self.stations:
            station_penalty = 0.0
            if station.waiting_times:
                for wait_time in station.waiting_times:
                    penalty_hours = wait_time / 60.0
                    station_penalty += penalty_hours * HOURLY_REVENUE_VALUE
            station_waiting_penalties[station.station_id] = station_penalty
        total_waiting_penalty = sum(station_waiting_penalties.values())

        # --- 4. 모든 페널티 항목을 합산하고 결과를 DataFrame 형식으로 반환합니다. ---
        total_penalty = total_truck_penalty + charger_penalty + total_waiting_penalty
        summary_results = {
            'failed_truck_penalty': failed_truck_penalty,
            'late_truck_penalty': late_truck_penalty,
            'truck_penalty': total_truck_penalty,
            'charger_penalty': charger_penalty,
            'waiting_penalty': total_waiting_penalty,
            'total_penalty': total_penalty
        }
        summary_df = pd.DataFrame([summary_results])
        station_penalty_df = pd.DataFrame(list(station_waiting_penalties.items()), columns=['station_id', 'waiting_penalty'])
        
        return summary_df, station_penalty_df

    def calculate_of(self):
        """
        OF(Objective Function) 값을 계산하고 재무/운영 요약을 저장합니다.
        OF = 총수익 - 총운영비용 - 총자본비용 - 총페널티
        """
        # 수익, 운영비용(OPEX), 자본비용(CAPEX)을 각각 계산합니다.
        revenue_df = self.calculate_revenue(self.station_results_df)
        opex_df = self.calculate_OPEX(self.station_results_df)
        capex_df = self.calculate_CAPEX(self.station_results_df)
        
        # 분리된 성공/실패 트럭 데이터를 기반으로 페널티를 계산합니다.
        penalty_summary_df, station_penalty_df = self.calculate_penalty(
            self.successful_trucks_df, self.failed_trucks_df, self.station_results_df
        )

        # 재무 및 페널티 관련 데이터프레임을 하나로 병합합니다.
        merged_df = pd.merge(revenue_df, opex_df, on='station_id', how='outer')
        merged_df = pd.merge(merged_df, capex_df, on='station_id', how='outer')
        merged_df = pd.merge(merged_df, station_penalty_df, on='station_id', how='outer')
        merged_df.fillna(0, inplace=True)
        
        if 'station_id' in merged_df.columns:
            merged_df['station_id'] = merged_df['station_id'].astype(int)
        
        # 페널티 차감 전 순이익을 계산합니다.
        merged_df['net_profit_before_penalty'] = merged_df['revenue'] - merged_df['opex'] - merged_df['capex']

        # 전체 네트워크의 재무 지표를 합산합니다.
        total_revenue = merged_df['revenue'].sum()
        total_opex = merged_df['opex'].sum()
        total_capex = merged_df['capex'].sum()
        total_penalty = penalty_summary_df['total_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        
        # 최종 목적 함수(OF) 값을 계산합니다.
        of_value = round(total_revenue - total_opex - total_capex - total_penalty)

        return of_value
    
    def load_stations(self, df):
        """
        DataFrame에서 충전소 정보를 읽어 Station 객체 리스트를 생성합니다.
        """
        stations = []
        required_cols = ['link_id', 'num_of_charger']
        if not all(col in df.columns for col in required_cols):
            return [] 

        stations = [
            Station(
                station_id=idx, 
                link_id=int(row['link_id']), 
                num_of_chargers=int(row['num_of_charger']), 
                charger_specs=[{'power': 200, 'rate': 560}] * int(row['num_of_charger']) 
            )
            for idx, row in df.iterrows() 
        ]
        return stations


# --- 전역 함수 ---

def run_simulation(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers):
    """
    시뮬레이션을 준비, 실행하고 최종 OF 값을 반환합니다. 실행 시간도 측정합니다.
    """

    sim = Simulator(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers)
    
    sim.prepare_simulation()

    sim.run_simulation()

    of = sim.analyze_results()

    return of

def load_car_path_df(car_paths_folder, number_of_trucks, estimated_areas=33):
    """
    차량 경로 데이터(Parquet 형식)를 로드하고 샘플링하는 함수.
    파일/데이터 관련 오류가 발생하지 않는다고 가정합니다.
    """
    load_start_time = time.time()

    # 1. 날짜 폴더 선택
    subfolders = [d for d in os.listdir(car_paths_folder)
                  if os.path.isdir(os.path.join(car_paths_folder, d)) and len(d) == 10 and d[4] == '-' and d[7] == '-']
    if not subfolders:
         raise FileNotFoundError(f"{car_paths_folder} 에서 'YYYY-MM-DD' 형식의 하위 폴더를 찾을 수 없습니다.")
    random_subfolder = random.choice(subfolders)
    selected_folder_path = os.path.join(car_paths_folder, random_subfolder)

    # 2. 전체 OBU_ID 및 AREA_ID 정보 수집
    parquet_files = []
    all_obu_data = []
    files_in_folder = [f for f in os.listdir(selected_folder_path) if f.endswith(".parquet")]
    if not files_in_folder:
         raise FileNotFoundError(f"{selected_folder_path} 에서 .parquet 파일을 찾을 수 없습니다.")
    
    for file in files_in_folder:
        file_path = os.path.join(selected_folder_path, file)
        parquet_files.append(file_path)
        table = pq.read_table(file_path, columns=['OBU_ID', 'AREA_ID', 'CUMULATIVE_LINK_LENGTH'])
        df_partial = table.to_pandas()
        last_entries = df_partial.loc[df_partial.groupby('OBU_ID')['CUMULATIVE_LINK_LENGTH'].idxmax()]
        all_obu_data.extend(last_entries[['OBU_ID', 'AREA_ID', 'CUMULATIVE_LINK_LENGTH']].values.tolist())
    
    if not all_obu_data:
         raise ValueError("어떤 파일에서도 OBU 정보를 추출할 수 없었습니다.")

    all_obu_df = pd.DataFrame(all_obu_data, columns=['OBU_ID', 'AREA_ID', 'MAX_CUMUL_DIST']).drop_duplicates(subset=['OBU_ID'])
    all_obu_ids = set(all_obu_df['OBU_ID'])
    unique_area_ids = all_obu_df['AREA_ID'].unique()
    actual_num_areas = len(unique_area_ids) 

    # 3. OBU_ID 샘플링
    selected_obu_ids = set()
    num_area_select = min(actual_num_areas, estimated_areas)
    area_sampled_ids = set()
    if num_area_select > 0:
        all_obu_df_sorted = all_obu_df.sort_values('MAX_CUMUL_DIST', ascending=False)
        area_groups = all_obu_df_sorted.groupby('AREA_ID')
        for area_id, group in area_groups:
            if len(area_sampled_ids) < num_area_select:
                area_sampled_ids.add(group['OBU_ID'].iloc[0])
            else: break
        selected_obu_ids.update(area_sampled_ids)

    remaining_needed = number_of_trucks - len(selected_obu_ids)
    if remaining_needed > 0:
        available_random_ids = list(all_obu_ids - selected_obu_ids)
        num_to_sample_randomly = min(remaining_needed, len(available_random_ids))
        if num_to_sample_randomly > 0:
            random_sampled_ids = random.sample(available_random_ids, num_to_sample_randomly)
            selected_obu_ids.update(random_sampled_ids)

    if len(selected_obu_ids) > number_of_trucks:
        excess = len(selected_obu_ids) - number_of_trucks
        ids_to_remove_from = list(selected_obu_ids - area_sampled_ids) 
        if len(ids_to_remove_from) >= excess:
            ids_to_remove = random.sample(ids_to_remove_from, excess)
        else: 
            ids_to_remove = ids_to_remove_from + random.sample(list(area_sampled_ids), excess - len(ids_to_remove_from))
        selected_obu_ids -= set(ids_to_remove)

    selected_obu_ids_set = selected_obu_ids 

    # 4. 선택된 OBU_ID 데이터 로드
    car_paths_list = []
    for file_path in parquet_files:
        filters = [('OBU_ID', 'in', list(selected_obu_ids_set))] 
        df_filtered = pd.read_parquet(file_path, engine='pyarrow', filters=filters) 
        
        if not df_filtered.empty:
            car_paths_list.append(df_filtered)

    if not car_paths_list:
        raise ValueError("선택된 OBU_ID에 대한 데이터를 로드하지 못했습니다.")

    # 5. 데이터 병합
    car_paths_df = pd.concat(car_paths_list, ignore_index=True)
    del car_paths_list 
    gc.collect()

    # 6. 데이터 전처리
    preprocess_start_time = time.time()
    car_paths_df['DATETIME'] = pd.to_datetime(car_paths_df['DATETIME'], format='%H:%M', errors='coerce').dt.time
    first_times = car_paths_df.groupby('OBU_ID')['DATETIME'].transform('first')
    car_paths_df['START_TIME_MINUTES'] = first_times.apply(lambda x: x.hour * 60 + x.minute if pd.notnull(x) else np.nan)
    
    original_truck_count = car_paths_df['OBU_ID'].nunique()
    car_paths_df.dropna(subset=['START_TIME_MINUTES'], inplace=True)
    final_truck_count = car_paths_df['OBU_ID'].nunique()

    return car_paths_df


def load_station_df(station_file_path):
    """충전소 데이터를 로드하고 전처리합니다. 파일/데이터 관련 오류 없다고 가정."""
    station_df = pd.read_csv(station_file_path, sep=',')
    station_df.columns = station_df.columns.str.strip().str.lower()
    if 'link_id' not in station_df.columns or 'num_of_charger' not in station_df.columns:
         raise ValueError("Station file missing required columns: 'link_id', 'num_of_charger'")
    station_df['link_id'] = station_df['link_id'].astype(int)
    station_df['num_of_charger'] = station_df['num_of_charger'].astype(int)
    return station_df


# 메인 함수 (스크립트 실행 시 호출)
if __name__ == '__main__':
    car_paths_folder = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_90km)"
    station_file_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Final_Candidates_Selected.csv"

    simulating_hours = 36
    unit_time = 20 
    number_of_trucks = 5946
    number_of_max_chargers = 2000 

    data_load_start = time.time()
    car_paths_df = load_car_path_df(car_paths_folder, number_of_trucks, estimated_areas=33)
    station_df = load_station_df(station_file_path)
    data_load_end = time.time()

    # 데이터 로딩 성공 여부 확인 (데이터가 비어있지 않은지)
    if car_paths_df is not None and not car_paths_df.empty and station_df is not None and not station_df.empty:
        gc.collect() 
        run_simulation(car_paths_df, station_df, unit_time, simulating_hours, number_of_trucks, number_of_max_chargers)

