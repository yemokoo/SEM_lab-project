from datetime import datetime, timedelta
import calendar
import random
import warnings
from matplotlib import pyplot as plt, ticker, colors
import pandas as pd
import numpy as np
import os
import gc
import time
from charger import Charger
from station import Station
from truck_for_month import Truck
import pyarrow.parquet as pq
import pyarrow as pa
import re
import seaborn as sns
import multiprocessing
import bisect

# 시드 설정을 통해 재현성 확보
seed = 42
random.seed(seed)
np.random.seed(seed)

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
    category=FutureWarning
)


class Simulator:
    """
    시뮬레이션 클래스 (최적화 및 로직 개선)
    """
    def __init__(self, car_paths_df, station_df, unit_minutes, simulating_hours, number_of_trucks, number_of_max_chargers, truck_step_frequency, num_days_in_month):
        self.car_paths_df = car_paths_df
        self.station_df = station_df
        self.number_of_max_chargers = number_of_max_chargers
        self.unit_minutes = unit_minutes
        self.simulating_hours = simulating_hours
        self.number_of_trucks_target = number_of_trucks
        self.truck_step_frequency = truck_step_frequency
        self.num_days_in_month = num_days_in_month

        self.daily_respawn_count = 0
        self.current_day_tracker = 1 

        self.stations = []
        self.link_id_to_station = {}
        self.trucks = []
        self.pending_trucks = []
        self.current_time = 0
        self.number_of_trucks_actual = 0
        self.truck_results_df = pd.DataFrame(columns=[
            'truck_id', 'final_SOC', 'destination_reached',
            'stopped_due_to_low_battery', 'stopped_due_to_simulation_end',
            'total_distance_planned', 'traveled_distance_at_last_stop85'
        ])
        self.station_results_df = None
        self.failed_trucks_df = None


    def prepare_simulation(self):
        """
        시뮬레이션 환경을 설정합니다. (Just-In-Time 트럭 생성 로직 적용)
        """
        self.stations = self.load_stations(self.station_df)
        self.link_id_to_station = {station.link_id: station for station in self.stations}

        operational_station_link_ids = {s.link_id for s in self.stations if s.num_of_chargers > 0}
        if 'EVCS' not in self.car_paths_df.columns:
            self.car_paths_df['EVCS'] = 0
        self.car_paths_df['EVCS'] = np.where(self.car_paths_df['LINK_ID'].isin(operational_station_link_ids), 1, 0)

        print("트럭 경로 데이터 그룹화 및 대기 명단 준비 중...")
        prep_start_time = time.time()

        for obu_id, group in self.car_paths_df.groupby('OBU_ID'):
            if not group.empty:
                start_time = group['START_TIME_MINUTES'].iloc[0]
                self.pending_trucks.append({'obu_id': obu_id, 'start_time': start_time, 'group': group})

        self.pending_trucks.sort(key=lambda x: x['start_time'])

        self.number_of_trucks_actual = len(self.pending_trucks)
        print(f"총 {self.number_of_trucks_actual}대의 트럭이 대기 명단에 등록되었습니다.")
        prep_end_time = time.time()
        print(f"  대기 명단 준비 완료 ({prep_end_time - prep_start_time:.2f}초 소요).")

        self.current_time = 0
        gc.collect()


    def run_simulation(self):
        """
        시뮬레이션을 실행합니다. (매일 00시에만 리스폰 요약 정보 출력)
        """
        total_steps = self.simulating_hours * (60 // self.unit_minutes)
        run_start_time = time.time()
        print(f"\n--- 시뮬레이션 시작 (총 {total_steps} 스텝, 단위 시간: {self.unit_minutes}분) ---")
        print(f"시뮬레이션 총 시간: {self.simulating_hours}시간 ({self.simulating_hours * 60}분)")

        last_printed_hour = -1

        for step_num in range(total_steps):
            current_total_hours = self.current_time / 60
            if int(current_total_hours) > last_printed_hour:
                day = int(current_total_hours // 24) + 1
                hour_of_day = int(current_total_hours % 24)
                elapsed_seconds = time.time() - run_start_time

                # --- [수정] 매일 00시에만 리스폰 정보를 출력하고, 이후 카운터를 리셋 ---
                if hour_of_day == 0:
                    # 00시에는 '이전 날'의 총 리스폰 대수를 출력
                    # self.daily_respawn_count는 직전 23시까지의 값이 그대로 유지된 상태
                    print(f"--- Day {day}, {hour_of_day:02d}:00 (활성: {len(self.trucks)}, 대기: {len(self.pending_trucks)}, 전일 리스폰: {self.daily_respawn_count}, 실행 시간: {elapsed_seconds:.1f}s) ---")
                    
                    # 출력이 끝난 후, '오늘'의 카운트를 위해 0으로 초기화
                    self.daily_respawn_count = 0
                else:
                    # 나머지 시간에는 리스폰 정보를 숨김
                    print(f"--- Day {day}, {hour_of_day:02d}:00 (활성: {len(self.trucks)}, 대기: {len(self.pending_trucks)}, 실행 시간: {elapsed_seconds:.1f}s) ---")
                
                last_printed_hour = int(current_total_hours)
                # --- 수정 완료 ---

            trucks_to_activate = []
            while self.pending_trucks and self.pending_trucks[0]['start_time'] <= self.current_time:
                trucks_to_activate.append(self.pending_trucks.pop(0))

            if trucks_to_activate:
                for truck_data in trucks_to_activate:
                    group = truck_data['group']
                    initial_soc = truck_data.get('initial_soc', None) 
                    
                    new_truck = Truck(
                        path_df=group, 
                        simulating_hours=self.simulating_hours, 
                        link_id_to_station=self.link_id_to_station, 
                        model=self, 
                        links_to_move=10,
                        initial_soc=initial_soc
                    )
                    self.trucks.append(new_truck)

            self.trucks[:] = [truck for truck in self.trucks if truck.status != 'stopped']
            if not self.trucks and not self.pending_trucks:
                print(f"모든 트럭이 종료되어 {step_num} 스텝에서 시뮬레이션을 조기 종료합니다.")
                break

            for station in self.stations:
                station.update_chargers(self.current_time)
                station.process_queue(self.current_time)

            if step_num % self.truck_step_frequency == 0:
                for truck in list(self.trucks):
                    if truck in self.trucks and self.current_time >= truck.next_activation_time:
                        truck.step(self.current_time)

            self.current_time += self.unit_minutes

        loop_end_time = time.time()
        print(f"--- 시뮬레이션 주 루프 종료 ({loop_end_time - run_start_time:.2f}초 소요) ---")

        print(f"\n--- 시뮬레이션 최종 정리 시작 ---")
        for station in self.stations:
            station.finalize_unprocessed_trucks(self.current_time)

        cleaned_up_count = 0
        for truck_to_cleanup in list(self.trucks):
            if truck_to_cleanup.status != 'stopped':
                truck_to_cleanup.stop()
                cleaned_up_count +=1

        print(f"--- 최종 정리 완료 ({cleaned_up_count}대 트럭 강제 종료) ---")

    def remove_truck(self, truck):
        """ 시뮬레이터의 활성 트럭 리스트에서 특정 트럭 객체를 제거합니다. """
        try:
            self.trucks.remove(truck)
        except ValueError:
            pass

    def handle_truck_respawn(self, dead_truck):
        """
        방전된 트럭을 경로상의 '다음 충전소'에서 낮은 SOC로 리스폰 시킵니다.
        """
        # 1. 방전 트럭의 정보 가져오기
        original_path_df = dead_truck.path_df
        if original_path_df is None or original_path_df.empty: return

        original_id = dead_truck.unique_id
        
        # --- [수정] OBU_ID가 bytes 타입일 경우 str으로 변환 ---
        if isinstance(original_id, bytes):
            original_id = original_id.decode('utf-8')
        # --- 수정 완료 ---

        death_index = dead_truck.current_path_index

        # 2. 방전 위치 '이후'의 경로에서 가장 가까운 충전소(EVCS)를 찾습니다.
        future_path = original_path_df.iloc[death_index:]
        evcs_in_future = future_path[future_path['EVCS'] == 1]

        if evcs_in_future.empty:
            #print(f"--- [RESPAWN FAILED] Truck {original_id} ran out of battery, but no EVCS found on the remaining path. ---")
            return

        respawn_station_original_index = evcs_in_future.index[0]
        respawn_link_id = original_path_df.loc[respawn_station_original_index]['LINK_ID']

        # 3. 리스폰할 경로와 ID, 시간, 상태를 준비합니다.
        new_path_df = original_path_df.iloc[respawn_station_original_index:].copy().reset_index(drop=True)
        if new_path_df.empty: return

        match = re.search(r'_(\d+)$', original_id)
        new_obu_id = f"{original_id.rsplit('_', 1)[0]}_{int(match.group(1)) + 1}" if match else f"{original_id}_1"
        
        respawn_time_str = new_path_df['DATETIME'].iloc[0]
        respawn_dt_object = pd.to_datetime(respawn_time_str)
        time_of_day_in_minutes = respawn_dt_object.hour * 60 + respawn_dt_object.minute
        
        FIXED_DOWNTIME_MINUTES = 2 * 60 
        new_start_time = self.current_time + FIXED_DOWNTIME_MINUTES

        # 4. 새 경로 데이터프레임 값들을 리셋
        new_path_df['OBU_ID'] = new_obu_id
        new_path_df['START_TIME_MINUTES'] = new_start_time
        new_path_df['CUMULATIVE_DRIVING_TIME_MINUTES'] -= new_path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[0]
        new_path_df['CUMULATIVE_LINK_LENGTH'] -= new_path_df['CUMULATIVE_LINK_LENGTH'].iloc[0]

        # 5. 리스폰 트럭을 '낮은 초기 SOC' 정보와 함께 대기열에 추가
        respawn_truck_data = {
            'obu_id': new_obu_id, 
            'start_time': new_start_time, 
            'group': new_path_df,
            'initial_soc': 5.0
        }
        
        bisect.insort(self.pending_trucks, respawn_truck_data, key=lambda x: x['start_time'])
        
        self.daily_respawn_count += 1
        
        # print(f"--- [RESPAWN] Truck {original_id} FAILED. Will respawn as {new_obu_id} at next EVCS (Link {respawn_link_id}) with 5% SOC. ---")

    def analyze_results(self):
        """
        시뮬레이션 결과를 분석하고, 계산을 조율한 뒤, 시각화/리포트 생성 함수를 호출합니다.
        """
        analysis_start_time = time.time()
        print("\n--- 결과 분석 및 계산 시작 ---")

        # 1. 충전소 운영 결과 집계
        station_data = []
        for station in self.stations:
            total_charged_energy_station = sum(c.total_charged_energy for c in station.chargers)
            total_charging_events_station = sum(c.charging_events_count for c in station.chargers)
            total_available_charger_minutes = self.simulating_hours * 60 * station.num_of_chargers
            total_charger_occupied_minutes = sum(c.total_charging_duration_minutes for c in station.chargers)
            utilization_percentage = (total_charger_occupied_minutes / total_available_charger_minutes * 100) if total_available_charger_minutes > 0 else 0

            station_data.append({
                'station_id': station.station_id, 'link_id': station.link_id, 'num_of_charger': station.num_of_chargers,
                'total_charged_energy_kWh': total_charged_energy_station, 'total_charging_events': total_charging_events_station,
                'avg_queue_length': np.mean(station.queue_history) if station.queue_history else 0,
                'max_queue_length': np.max(station.queue_history) if station.queue_history else 0,
                'avg_waiting_time_min': np.mean(station.waiting_times) if station.waiting_times else 0,
                'utilization_percentage': round(utilization_percentage, 2),
                'queue_history_raw': station.queue_history, 'charging_history_raw': station.charging_history,
                'cumulative_arrivals_history': station.cumulative_arrivals_history, 'cumulative_departures_history': station.cumulative_departures_history,
            })
        self.station_results_df = pd.DataFrame(station_data)

        # 2. 실패 트럭 집계
        if self.truck_results_df is None or self.truck_results_df.empty:
            self.failed_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else [])
        else:
            self.failed_trucks_df = self.truck_results_df[
                (self.truck_results_df['destination_reached'] == False) & (self.truck_results_df['stopped_due_to_low_battery'] == True)             
            ].copy()
        print(f"  실패 트럭 수: {len(self.failed_trucks_df)}대")
        
        # 3. 재무 및 페널티 계산 (계산만 수행)
        of_value, merged_df, penalty_summary_df, station_penalty_df, timestamped_folder_path = self.calculate_of()
        
        analysis_end_time = time.time()
        print(f"--- 결과 분석 및 계산 완료 ({analysis_end_time - analysis_start_time:.2f}초 소요) ---")

        # 4. 시각화 및 리포트 생성 함수 호출
        self.generate_visualizations_and_reports(merged_df, penalty_summary_df, station_penalty_df, timestamped_folder_path)
        
        return of_value


    def calculate_of(self):
        """
        OF(Objective Function) 값과 관련 재무 데이터프레임을 계산하여 반환합니다.
        """
        if self.station_results_df is None or self.station_results_df.empty:
            print("Warning: station_results_df가 비어있어 OF 계산을 중단합니다.")
            return 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""

        revenue_df = self.calculate_revenue(self.station_results_df)
        opex_df = self.calculate_OPEX(self.station_results_df)
        capex_df = self.calculate_CAPEX(self.station_results_df)
        penalty_summary_df, station_penalty_df = self.calculate_penalty(
            self.failed_trucks_df, self.station_results_df
        )

        merged_df = pd.merge(revenue_df, opex_df, on='station_id', how='outer')
        merged_df = pd.merge(merged_df, capex_df, on='station_id', how='outer')
        merged_df = pd.merge(merged_df, station_penalty_df, on='station_id', how='outer')
        
        merged_df.fillna(0, inplace=True)
        if 'station_id' in merged_df.columns:
            merged_df['station_id'] = merged_df['station_id'].astype(int)
        
        merged_df['net_profit_before_penalty'] = merged_df['revenue'] - merged_df['opex'] - merged_df['capex']

        total_revenue = merged_df['revenue'].sum()
        total_opex = merged_df['opex'].sum()
        total_capex = merged_df['capex'].sum()
        total_penalty = penalty_summary_df['total_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        
        of_value = round(total_revenue - total_opex - total_capex - total_penalty)
        
        base_save_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Result"
        current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M")
        timestamped_folder_path = os.path.join(base_save_path, current_timestamp_str)
        os.makedirs(timestamped_folder_path, exist_ok=True)

        return of_value, merged_df, penalty_summary_df, station_penalty_df, timestamped_folder_path


    def calculate_OPEX(self, station_df):
        """ 모든 충전소의 OPEX(운영 비용)를 계산합니다. (월 단위 비용) """
        opex_results = []
        base_rate_per_kw = 2580
        energy_rate_per_kwh = 101.7 + 9 + 5
        vat_and_fund_multiplier = 1.132
        labor_cost_per_charger_daily = 6250
        maint_cost_per_charger_daily = 800

        for station in self.stations:
            total_charged_energy_station = sum(c.total_charged_energy for c in station.chargers)
            total_power = sum(c.power for c in station.chargers)
            energy_price = ((total_power * base_rate_per_kw) + (total_charged_energy_station * energy_rate_per_kwh)) * vat_and_fund_multiplier
            labor_cost = station.num_of_chargers * labor_cost_per_charger_daily * self.num_days_in_month
            maintenance_cost = station.num_of_chargers * maint_cost_per_charger_daily * self.num_days_in_month
            opex = labor_cost + maintenance_cost + energy_price
            opex_results.append({
                'station_id': station.station_id, 'labor_cost': labor_cost,
                'maintenance_cost': maintenance_cost, 'energy_price': energy_price, 'opex': opex
            })
        return pd.DataFrame(opex_results)


    def calculate_CAPEX(self, station_df):
        """ 모든 충전소의 CAPEX(자본 비용)를 계산합니다. (월 단위 비용 기준) """
        capex_results = []
        monthly_divider = 5 * 12
        charger_installation_cost_per_unit = 96000000

        for station in self.stations:
            num_chargers = station.num_of_chargers
            charger_cost = (charger_installation_cost_per_unit * num_chargers) / monthly_divider if num_chargers > 0 else 0
            capex_results.append({
                'station_id': station.station_id, 'charger_cost': charger_cost, 'capex': charger_cost
            })
        return pd.DataFrame(capex_results)


    def calculate_revenue(self, station_df):
        """ 모든 충전소의 수익을 계산합니다. """
        revenue_results = [{'station_id': s.station_id, 'revenue': sum(c.rate * c.total_charged_energy for c in s.chargers)} for s in self.stations]
        return pd.DataFrame(revenue_results)


    def calculate_penalty(self, failed_trucks_df, station_df):
        """ 시뮬레이션 결과에 기반하여 다양한 유형의 페널티를 계산합니다. """
        failed_truck_penalty = 0.0
        if failed_trucks_df is not None and not failed_trucks_df.empty:
            planned_dist = failed_trucks_df['total_distance_planned']
            last_stop_dist = failed_trucks_df['traveled_distance_at_last_stop85'].fillna(0)
            distance_for_penalty = np.where(last_stop_dist <= 0, planned_dist / 2, np.maximum(0, planned_dist - last_stop_dist) / 2)
            choice = np.random.choice([True, False], size=len(failed_trucks_df))
            penalty = np.where(choice, 136395.90 + 3221.87 * distance_for_penalty - 2.72 * distance_for_penalty**2, 121628.18 + 2765.50 * distance_for_penalty - 2.00 * distance_for_penalty**2)
            failed_truck_penalty = np.maximum(0, penalty).sum()
        
        charger_penalty = 0.0
        number_of_total_chargers = sum(s.num_of_chargers for s in self.stations)
        if number_of_total_chargers > self.number_of_max_chargers:
            charger_penalty = float(96000000 * (number_of_total_chargers - self.number_of_max_chargers))

        HOURLY_REVENUE_VALUE = 11000000 / (10.9 * 22.4)
        station_waiting_penalties = {s.station_id: sum(wt / 60.0 * HOURLY_REVENUE_VALUE for wt in s.waiting_times) for s in self.stations}
        total_waiting_penalty = sum(station_waiting_penalties.values())
        
        summary_results = {
            'failed_truck_penalty': failed_truck_penalty, 'truck_penalty': failed_truck_penalty,
            'charger_penalty': charger_penalty, 'waiting_penalty': total_waiting_penalty,
            'total_penalty': failed_truck_penalty + charger_penalty + total_waiting_penalty
        }
        summary_df = pd.DataFrame([summary_results])
        station_penalty_df = pd.DataFrame(list(station_waiting_penalties.items()), columns=['station_id', 'waiting_penalty'])
        return summary_df, station_penalty_df


    def generate_visualizations_and_reports(self, merged_df, penalty_summary_df, station_penalty_df, timestamped_folder_path):
        """
        모든 시각화 자료와 리포트(CSV, 그래프)를 생성하고 저장합니다.
        """
        print("\n--- 결과 시각화 및 리포트 생성 시작 ---")
        print(f"결과가 다음 경로에 저장됩니다: {timestamped_folder_path}")

        # --- 1. CSV 파일 저장 ---
        self.station_results_df.to_csv(os.path.join(timestamped_folder_path, "station_operational_summary.csv"), index=False, encoding='utf-8-sig')
        merged_df.to_csv(os.path.join(timestamped_folder_path, "financial_summary_by_station.csv"), index=False, encoding='utf-8-sig')
        print("CSV 리포트 (운영 요약, 재무 요약) 저장 완료.")
        
        # --- 2. 재무 요약 출력 ---
        total_revenue = merged_df['revenue'].sum()
        total_opex = merged_df['opex'].sum()
        total_capex = merged_df['capex'].sum()
        total_penalty = penalty_summary_df['total_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        truck_p = penalty_summary_df['truck_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        failed_truck_p = penalty_summary_df['failed_truck_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        charger_p = penalty_summary_df['charger_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        waiting_p = penalty_summary_df['waiting_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        of_value = round(total_revenue - total_opex - total_capex - total_penalty)
        
        print(f"\n--- Financial Summary (Monthly) ---")
        print(f"Total Revenue                  : {total_revenue:,.0f}")
        print(f"Total OPEX                     : {total_opex:,.0f}")
        print(f"Total CAPEX                    : {total_capex:,.0f}")
        print(f"Total Penalty                  : {total_penalty:,.0f}")
        print(f"  ├─ Truck Penalty (Total)      : {truck_p:,.0f}")
        print(f"  │  └─ Failed Truck Penalty   : {failed_truck_p:,.0f}")
        print(f"  ├─ Charger Penalty            : {charger_p:,.0f}")
        print(f"  └─ Waiting Penalty            : {waiting_p:,.0f}")
        print(f"------------------------------------")
        print(f"Objective Function (OF) Value    : {of_value:,.0f}")
        print(f"------------------------------------")

        if 'station_id' not in merged_df.columns or merged_df['station_id'].isnull().all():
            print("오류: 'station_id'가 없어 그래프 생성을 중단합니다.")
            return

        # --- 3. 그래프 생성을 위한 데이터 준비 ---
        analysis_df = pd.merge(self.station_results_df, self.station_df, on=['link_id', 'num_of_charger'], how='left')
        analysis_df.fillna(0, inplace=True)
        analysis_df['station_id'] = analysis_df['station_id'].astype(int)

        plot_data_scatter = analysis_df[analysis_df['num_of_charger'] > 0].copy()

        # --- 4. 후보지 특징별 Scatter Plot ---
        if not plot_data_scatter.empty:
            print("\n--- 후보지 특징별 대기시간 Scatter Plot 생성 시작 ---")
            scored_features = ['point', 'traffic', 'od']
            binary_features = ['rest_area', 'infra', 'interval']
            all_scatter_features = scored_features + binary_features

            scored_color_list = ['blue', 'green', 'yellow', 'orange', 'red', 'purple']
            scored_cmap = colors.ListedColormap(scored_color_list)
            scored_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            scored_norm = colors.BoundaryNorm(scored_bounds, scored_cmap.N)

            binary_color_list = ['black', 'red']
            binary_cmap = colors.ListedColormap(binary_color_list)
            binary_bounds = [-0.5, 0.5, 1.5]
            binary_norm = colors.BoundaryNorm(binary_bounds, binary_cmap.N)

            for feature in all_scatter_features:
                if feature not in plot_data_scatter.columns:
                    print(f" - 경고: '{feature}' 컬럼이 없어 해당 그래프는 건너뜁니다.")
                    continue
                
                if feature in scored_features:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    scatter = ax.scatter(plot_data_scatter['num_of_charger'], plot_data_scatter['avg_waiting_time_min'], c=plot_data_scatter[feature], cmap=scored_cmap, norm=scored_norm, alpha=0.8, s=90, edgecolors='w', linewidth=0.5)
                    ax.set_xlabel('Number of Chargers per Station', fontsize=12); ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12); ax.set_title(f'Chargers vs. Waiting Time (Colored by {feature.capitalize()})', fontsize=16)
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    legend = ax.legend(*scatter.legend_elements(prop="colors", num="auto"), title=f'{feature.capitalize()} Score', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
                    ax.add_artist(legend)
                    plt.tight_layout(rect=[0, 0, 0.9, 1]); plt.savefig(os.path.join(timestamped_folder_path, f"station_chargers_vs_wait_time_by_{feature}.png"), dpi=300); plt.close(fig)

                elif feature in binary_features:
                    plot_data_yes_only = plot_data_scatter[plot_data_scatter[feature] == 1]
                    if not plot_data_yes_only.empty:
                        fig, ax = plt.subplots(figsize=(16, 10))
                        ax.scatter(plot_data_yes_only['num_of_charger'], plot_data_yes_only['avg_waiting_time_min'], c='red', alpha=0.8, s=90, edgecolors='w', linewidth=0.5, label=f'{feature.capitalize()} = YES')
                        ax.set_xlabel('Number of Chargers per Station', fontsize=12); ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12); ax.set_title(f'Chargers vs. Waiting Time for Stations with "{feature.capitalize()}" (YES only)', fontsize=16)
                        ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)); ax.legend()
                        plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, f"station_chargers_vs_wait_time_by_{feature}_YES_ONLY.png"), dpi=300); plt.close(fig)
                    
                    fig, ax = plt.subplots(figsize=(16, 10))
                    scatter = ax.scatter(plot_data_scatter['num_of_charger'], plot_data_scatter['avg_waiting_time_min'], c=plot_data_scatter[feature], cmap=binary_cmap, norm=binary_norm, alpha=0.8, s=90, edgecolors='w', linewidth=0.5)
                    ax.set_xlabel('Number of Chargers per Station', fontsize=12); ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12); ax.set_title(f'Chargers vs. Waiting Time (Colored by {feature.capitalize()})', fontsize=16)
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    legend = ax.legend(*scatter.legend_elements(prop="colors", num="auto"), title=f'{feature.capitalize()} (1:Yes, 0:No)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
                    ax.add_artist(legend)
                    plt.tight_layout(rect=[0, 0, 0.9, 1]); plt.savefig(os.path.join(timestamped_folder_path, f"station_chargers_vs_wait_time_by_{feature}_BOTH.png"), dpi=300); plt.close(fig)
            print("후보지 특징별 대기시간 Scatter Plot 저장 완료.")
        
        # --- 5. Boxplot (개별 & 통합) ---
        plot_data_box = pd.merge(analysis_df, merged_df, on='station_id', how='left')
        plot_data_box = plot_data_box[plot_data_box['num_of_charger'] > 0]
        if not plot_data_box.empty:
            print("\n--- 개별/통합 지표 Boxplot 생성 시작 ---")
            boxplot_configs = [
                {'col': 'avg_waiting_time_min', 'title': 'Distribution of Average Waiting Time', 'ylabel': 'Average Waiting Time (min)'},
                {'col': 'num_of_charger', 'title': 'Distribution of Installed Chargers', 'ylabel': 'Number of Chargers'},
                {'col': 'revenue', 'title': 'Distribution of Monthly Revenue', 'ylabel': 'Revenue (KRW)'},
                {'col': 'opex', 'title': 'Distribution of Monthly OPEX', 'ylabel': 'OPEX (KRW)'},
                {'col': 'capex', 'title': 'Distribution of Monthly CAPEX', 'ylabel': 'CAPEX (KRW)'},
                {'col': 'waiting_penalty', 'title': 'Distribution of Monthly Waiting Penalty', 'ylabel': 'Waiting Penalty (KRW)'},
            ]
            for config in boxplot_configs:
                fig, ax = plt.subplots(figsize=(8, 10))
                sns.boxplot(y=plot_data_box[config['col']], ax=ax, palette="pastel", width=0.5)
                ax.set_title(f"{config['title']} (Chargers > 0)", fontsize=16, weight='bold')
                ax.set_ylabel(config['ylabel'], fontsize=12); ax.set_xlabel('')
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                if 'KRW' in config['ylabel']:
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                else:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, f"station_{config['col']}_boxplot.png"), dpi=300); plt.close(fig)
            
            financial_cols = ['revenue', 'opex', 'capex', 'waiting_penalty']
            melted_df = plot_data_box.melt(id_vars=['station_id'], value_vars=financial_cols, var_name='Financial Component', value_name='Amount (KRW)')
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.boxplot(x='Financial Component', y='Amount (KRW)', data=melted_df, ax=ax, palette='viridis', width=0.6)
            ax.set_title('Comparison of Monthly Financial Components per Station', fontsize=18, weight='bold')
            ax.set_xlabel('Financial Component', fontsize=14); ax.set_ylabel('Amount (KRW)', fontsize=14)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.xticks(rotation=10, ha='right', fontsize=12)
            plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "financial_components_comparison_boxplot.png"), dpi=300); plt.close(fig)
            print("개별/통합 지표 Boxplot 저장 완료.")
            
        # --- 6. 기타 분포 및 상관관계 그래프 ---
        print("\n--- 분포, 상관관계 및 관계도 그래프 생성 시작 ---")
        # 충전기 대수 분포 Histogram
        plot_data_hist = analysis_df[analysis_df['num_of_charger'] > 0]
        if not plot_data_hist.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.histplot(data=plot_data_hist, x='num_of_charger', discrete=True, shrink=0.8, ax=ax)
            ax.set_title('Distribution of Installed Chargers per Station (where chargers > 0)', fontsize=16); ax.set_xlabel('Number of Chargers', fontsize=12); ax.set_ylabel('Number of Stations', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7); ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "station_charger_count_histogram.png"), dpi=300); plt.close(fig)

        # [수정] 상관관계 Heatmap: 분석 대상 변수 변경 및 레이블 수정
        correlation_df = pd.merge(analysis_df, merged_df, on='station_id', how='left')
        
        # 분석에 사용할 컬럼 (소문자 snake_case)
        correlation_cols_internal = [
            'point', 'od', 'rest_area', 'traffic', 'infra', 'interval', 
            'revenue', 'opex', 'capex', 'waiting_penalty'
        ]
        
        # 그래프에 표시될 예쁜 이름
        correlation_cols_display = [
            'Point', 'OD', 'Rest Area', 'Traffic', 'Infra', 'Interval', 
            'Revenue', 'OPEX', 'CAPEX', 'Waiting Time Penalty'
        ]
        
        # 실제 데이터프레임에 있는 컬럼만 필터링
        existing_cols_for_corr = [col for col in correlation_cols_internal if col in correlation_df.columns]
        
        if len(existing_cols_for_corr) > 1:
            corr_matrix = correlation_df[existing_cols_for_corr].corr()
            
            # Heatmap에 표시하기 전에 컬럼 이름 변경
            rename_dict = {internal: display for internal, display in zip(correlation_cols_internal, correlation_cols_display)}
            corr_matrix.rename(columns=rename_dict, index=rename_dict, inplace=True)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax)
            ax.set_title('Correlation Matrix: Candidate Criteria vs. Financial Results', fontsize=16, weight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "correlation_heatmap_criteria_vs_financials.png"), dpi=300)
            plt.close(fig)
        
        # 충전기 수 vs 대기시간 Scatter
        op_df_scatter = pd.merge(self.station_results_df, station_penalty_df, on='station_id', how='left').fillna(0)
        plot_data_scatter_wait = op_df_scatter[op_df_scatter['num_of_charger'] > 0].copy()
        if not plot_data_scatter_wait.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_data_scatter_wait['num_of_charger_jitter'] = plot_data_scatter_wait['num_of_charger'] + np.random.normal(0, 0.1, size=len(plot_data_scatter_wait))
            ax.scatter(plot_data_scatter_wait['num_of_charger_jitter'], plot_data_scatter_wait['avg_waiting_time_min'], alpha=0.6, s=50, label='Stations')
            slope, intercept = np.polyfit(plot_data_scatter_wait['num_of_charger'], plot_data_scatter_wait['avg_waiting_time_min'], 1)
            x_trend = np.array(sorted(plot_data_scatter_wait['num_of_charger'].unique()))
            ax.plot(x_trend, slope * x_trend + intercept, color='red', linestyle='--', label=f'Trend (y={slope:.2f}x + {intercept:.2f})')
            avg_wait_time = plot_data_scatter_wait['avg_waiting_time_min'].mean()
            ax.axhline(y=avg_wait_time, color='green', linestyle=':', linewidth=2, label=f'Average Waiting Time: {avg_wait_time:.2f} min')
            ax.set_title('Relationship between Number of Chargers and Average Waiting Time', fontsize=16)
            ax.set_xlabel('Number of Chargers per Station', fontsize=12); ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)); ax.set_ylim(bottom=0); ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.legend()
            fig.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "station_chargers_vs_wait_time_scatter_original.png"), dpi=300); plt.close(fig)
        print("분포, 상관관계 및 관계도 그래프 저장 완료.")

        # --- 7. 충전소별 상세 시계열 그래프 (Bar, Line 등) ---
        print("\n--- 충전소별 상세 시계열 그래프 생성 시작 ---")
        merged_df.sort_values('station_id', inplace=True)
        financial_station_ids_int = merged_df['station_id']
        financial_x_labels_str = financial_station_ids_int.astype(str)
        
        def set_xticks_by_50(ax, station_ids_int):
            if station_ids_int.empty: return
            unique_sorted_ids = station_ids_int.unique(); min_id, max_id = unique_sorted_ids[0], unique_sorted_ids[-1]
            ticks_to_show = [sid for sid in unique_sorted_ids if sid % 50 == 0]
            if not ticks_to_show:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
            else:
                if min_id not in ticks_to_show: ticks_to_show.insert(0, min_id)
                if max_id not in ticks_to_show and max_id != min_id: ticks_to_show.append(max_id)
                ax.set_xticks(sorted(list(set(ticks_to_show)))); ax.set_xticklabels([str(t) for t in sorted(list(set(ticks_to_show)))], rotation=90, ha='right')

        # 재무 요소 Bar Chart
        fig1, ax1 = plt.subplots(figsize=(18, 9))
        ax1.bar(financial_x_labels_str, merged_df['revenue'], label='Revenue', color='green')
        neg_opex, neg_capex, neg_waiting_penalty = -merged_df['opex'], -merged_df['capex'], -merged_df['waiting_penalty']
        ax1.bar(financial_x_labels_str, neg_opex, label='OPEX', color='orangered')
        ax1.bar(financial_x_labels_str, neg_capex, bottom=neg_opex, label='CAPEX', color='darkred')
        ax1.bar(financial_x_labels_str, neg_waiting_penalty, bottom=neg_opex + neg_capex, label='Waiting Penalty', color='gold')
        penalty_text = f"Truck Penalty: {round(truck_p):,.0f}\nCharger Penalty: {round(charger_p):,.0f}\nWaiting Penalty: {round(waiting_p):,.0f}"
        ax1.text(0.98, 0.98, penalty_text, ha='right', va='top', transform=ax1.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
        ax1.set_xlabel('Station ID'); ax1.set_ylabel('Amount (KRW)'); ax1.set_title('Financial Components by Station'); ax1.legend(loc='best'); set_xticks_by_50(ax1, financial_station_ids_int); ax1.axhline(0, color='black', linewidth=0.8); ax1.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "station_financial_components.png")); plt.close(fig1)

        # 순이익 Bar Chart
        fig2, ax2 = plt.subplots(figsize=(18, 9))
        net_profit_colors = ['mediumseagreen' if x >= 0 else 'tomato' for x in merged_df['net_profit_before_penalty']]
        ax2.bar(financial_x_labels_str, merged_df['net_profit_before_penalty'], label='Net Profit (Before Penalty)', color=net_profit_colors)
        ax2.set_xlabel('Station ID'); ax2.set_ylabel('Net Profit'); ax2.set_title('Net Profit by Station (Before Penalty)'); ax2.legend(loc='best'); set_xticks_by_50(ax2, financial_station_ids_int); ax2.axhline(0, color='black', linewidth=0.8); ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "station_net_profit_before_penalty.png")); plt.close(fig2)

        # 운영지표 Bar Charts
        op_df = pd.merge(self.station_results_df, station_penalty_df, on='station_id', how='left').fillna(0)
        op_df['station_id'] = op_df['station_id'].astype(int); op_df.sort_values('station_id', inplace=True)
        op_station_ids_int, op_x_labels_str = op_df['station_id'], op_df['station_id'].astype(str)
        graph_configs = [
            {'y_col': 'total_charged_energy_kWh', 'title': 'Total Charged Energy per Station (Monthly)', 'ylabel': 'Total Charged Energy (kWh)', 'color': 'dodgerblue', 'avg_color': 'red'},
            {'y_col': 'total_charging_events', 'title': 'Total Charging Events per Station (Monthly)', 'ylabel': 'Total Charging Events', 'color': 'mediumpurple', 'avg_color': 'darkmagenta'},
            {'y_col': 'avg_waiting_time_min', 'title': 'Average Waiting Time per Station (Monthly)', 'ylabel': 'Average Waiting Time (minutes)', 'color': 'teal', 'avg_color': 'darkcyan'},
            {'y_col': 'num_of_charger', 'title': 'Number of Chargers per Station', 'ylabel': 'Number of Chargers', 'color': 'goldenrod', 'avg_color': 'darkgoldenrod'},
            {'y_col': 'waiting_penalty', 'title': 'Waiting Time Penalty per Station (Monthly)', 'ylabel': 'Waiting Time Penalty (KRW)', 'color': 'lightcoral', 'avg_color': 'darkred'},
            {'y_col': 'utilization_percentage', 'title': 'Charger Utilization Percentage per Station (Monthly)', 'ylabel': 'Utilization (%)', 'color': 'darkgreen', 'avg_color': 'limegreen'}
        ]
        for config in graph_configs:
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.bar(op_x_labels_str, op_df[config['y_col']], label=config['ylabel'], color=config['color'])
            avg_val = op_df[op_df['num_of_charger'] > 0][config['y_col']].mean() if config['y_col'] in ['avg_waiting_time_min', 'utilization_percentage'] and not op_df[op_df['num_of_charger'] > 0].empty else op_df[config['y_col']].mean()
            ax.axhline(y=avg_val, color=config['avg_color'], linestyle='--', linewidth=1.5, label=f'Average: {avg_val:.2f}')
            ax.set_xlabel('Station ID'); ax.set_ylabel(config['ylabel']); ax.set_title(config['title']); ax.legend(loc='best'); set_xticks_by_50(ax, op_station_ids_int); ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, f"station_{config['y_col']}.png")); plt.close(fig)

        # 대기열 길이 Bar Chart
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.bar(op_x_labels_str, op_df['max_queue_length'], label='Max Queue Length', color='#1f77b4')
        ax.bar(op_x_labels_str, op_df['avg_queue_length'], label='Average Queue Length', color='#ff7f0e')
        avg_queue = op_df['avg_queue_length'].mean()
        ax.axhline(y=avg_queue, color='red', linestyle='--', linewidth=1.5, label=f'Overall Avg Queue Length: {avg_queue:.2f}')
        ax.set_xlabel('Station ID'); ax.set_ylabel('Queue Length (Number of Trucks)'); ax.set_title('Average and Max Queue Length per Station'); ax.legend(loc='best'); set_xticks_by_50(ax, op_station_ids_int); ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "station_queue_lengths.png")); plt.close(fig)
        
        # 점유/대기열, 누적 다이어그램
        if 'queue_history_raw' in self.station_results_df.columns and 'charging_history_raw' in self.station_results_df.columns:
            graph_folder = os.path.join(timestamped_folder_path, "station_occupancy_graphs")
            os.makedirs(graph_folder, exist_ok=True)
            stations_with_activity = self.station_results_df[self.station_results_df.apply(lambda row: (len(row['queue_history_raw']) > 0 and pd.Series(row['queue_history_raw']).max() > 0) or (len(row['charging_history_raw']) > 0 and pd.Series(row['charging_history_raw']).max() > 0), axis=1)]
            for index, row in stations_with_activity.iterrows():
                station_id, queue_history, charging_history, num_chargers = int(row['station_id']), row['queue_history_raw'], row['charging_history_raw'], int(row['num_of_charger'])
                fig, ax = plt.subplots(figsize=(15, 7))
                time_steps = np.arange(len(queue_history)) * self.unit_minutes
                ax.bar(time_steps, charging_history, width=self.unit_minutes, color='skyblue', alpha=0.8, label=f'Charging Trucks')
                ax.plot(time_steps, queue_history, marker='o', color='orangered', linestyle='-', markersize=4, label='Queued Trucks')
                ax.axhline(y=num_chargers, color='dodgerblue', linestyle='--', linewidth=1.5, label=f'Capacity ({num_chargers} Chargers)')
                ax.set_title(f'Station {station_id}: Occupancy & Queue History (Monthly)', fontsize=16, weight='bold'); ax.set_xlabel('Simulation Time (minutes)', fontsize=12); ax.set_ylabel('Number of Trucks', fontsize=12)
                ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5); ax.legend(loc='upper left'); ax.set_ylim(bottom=0); ax.set_xlim(left=0); ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.tight_layout(); plt.savefig(os.path.join(graph_folder, f"station_{station_id}_occupancy_queue.png"), dpi=150); plt.close(fig)

        if 'cumulative_arrivals_history' in self.station_results_df.columns:
            graph_folder = os.path.join(timestamped_folder_path, "cumulative_queue_graphs")
            os.makedirs(graph_folder, exist_ok=True)
            stations_with_activity = self.station_results_df[self.station_results_df['cumulative_arrivals_history'].apply(lambda x: len(x) > 1 and pd.Series(x).max() > 0)]
            for index, row in stations_with_activity.iterrows():
                station_id, arrivals, departures = int(row['station_id']), row['cumulative_arrivals_history'], row['cumulative_departures_history']
                fig, ax = plt.subplots(figsize=(15, 7))
                time_steps = np.arange(len(arrivals)) * self.unit_minutes
                ax.plot(time_steps, arrivals, drawstyle='steps-post', color='blue', label='Cumulative Arrivals')
                ax.plot(time_steps, departures, drawstyle='steps-post', color='green', label='Cumulative Departures (Charging Start)')
                ax.fill_between(time_steps, arrivals, departures, step='post', color='gray', alpha=0.3, label='waiting time')
                ax.set_title(f'Cumulative Queuing Diagram for Station {station_id} (Monthly)', fontsize=16, weight='bold'); ax.set_xlabel('Simulation Time (minutes)', fontsize=12); ax.set_ylabel('Cumulative Number of Trucks', fontsize=12)
                ax.grid(True, which='major', linestyle='--', linewidth=0.5); ax.legend(loc='upper left'); ax.set_ylim(bottom=0); ax.set_xlim(left=0, right=time_steps[-1])
                plt.tight_layout(); plt.savefig(os.path.join(graph_folder, f"station_{station_id}_cumulative_diagram.png"), dpi=150); plt.close(fig)
        print("충전소별 상세 시계열 그래프 저장 완료.")

        # --- 8. 월별 전체 요약 통계 및 그래프 (기존 함수 로직 통합) ---
        print("\n--- 월별 전체 요약 및 트래픽 추이 생성 시작 ---")
        total_active_trucks = self.number_of_trucks_actual
        actual_stopped_trucks = self.truck_results_df[self.truck_results_df['stopped_due_to_low_battery'] | self.truck_results_df['stopped_due_to_simulation_end'] | self.truck_results_df['destination_reached']]
        monthly_summary = {
            'Total Active Trucks Simulated': total_active_trucks,
            'Successful Trips (Destination Reached)': len(self.truck_results_df[self.truck_results_df['destination_reached'] == True]),
            'Failed Trips (Low Battery)': len(self.truck_results_df[self.truck_results_df['stopped_due_to_low_battery'] == True]),
            'Failed Trips (Simulation End)': len(self.truck_results_df[self.truck_results_df['stopped_due_to_simulation_end'] == True]),
            'Average Final SOC of Stopped Trucks (%)': round(actual_stopped_trucks['final_SOC'].mean() if not actual_stopped_trucks.empty else 0, 2),
            'Total Traveled Distance (km) (all loaded data)': round(self.car_paths_df['CUMULATIVE_LINK_LENGTH'].sum(), 2),
            'Total Charged Energy (kWh)': round(self.station_results_df['total_charged_energy_kWh'].sum() if self.station_results_df is not None else 0, 2),
            'Total Waiting Time at Stations (minutes)': round(sum(sum(s.waiting_times) for s in self.stations), 2),
            'Total Chargers Installed': self.station_df['num_of_charger'].sum(),
            'Average Charger Utilization (%)': self.station_results_df[self.station_results_df['num_of_charger'] > 0]['utilization_percentage'].mean() if self.station_results_df is not None and not self.station_results_df[self.station_results_df['num_of_charger'] > 0].empty else 0
        }
        summary_df = pd.DataFrame([monthly_summary]).T.rename(columns={0: 'Value'}); summary_df.index.name = 'Metric'
        summary_df.to_csv(os.path.join(timestamped_folder_path, "monthly_overall_summary.csv"), encoding='utf-8-sig')
        
        all_queue_histories = [s.queue_history for s in self.stations if s.queue_history]
        all_charging_histories = [s.charging_history for s in self.stations if s.charging_history]
        if all_queue_histories or all_charging_histories:
            max_len = max([len(h) for h_list in [all_queue_histories, all_charging_histories] if h_list for h in h_list])
            total_queued = np.zeros(max_len); total_charging = np.zeros(max_len)
            for h in all_queue_histories: total_queued[:len(h)] += h
            for h in all_charging_histories: total_charging[:len(h)] += h
            time_steps = np.arange(len(total_queued)) * self.unit_minutes
            if len(time_steps) > 1:
                fig, ax = plt.subplots(figsize=(18, 9))
                ax.plot(time_steps, total_queued, label='Total Queued Trucks', color='orange')
                ax.plot(time_steps, total_charging, label='Total Charging Trucks', color='blue')
                ax.set_xlabel('Simulation Time (minutes)'); ax.set_ylabel('Number of Trucks'); ax.set_title('Overall Monthly Traffic and Charging Activity')
                ax.legend(); ax.grid(True)
                plt.tight_layout(); plt.savefig(os.path.join(timestamped_folder_path, "overall_monthly_traffic_activity.png")); plt.close(fig)
        print("월별 전체 요약 및 트래픽 추이 저장 완료.")
        print("\n--- 모든 시각화 및 리포트 생성 완료 ---")

    def load_stations(self, df):
        stations = []
        required_cols = ['link_id', 'num_of_charger']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Station file missing required columns: {required_cols}. Returning empty list.")
            return []

        stations = [
            Station(
                station_id=idx,
                link_id=int(row['link_id']),
                num_of_chargers=int(row['num_of_charger']),
                charger_specs=[{'power': 200, 'rate': 560}] * int(row['num_of_charger']),
                unit_minutes=self.unit_minutes
            )
            for idx, row in df.iterrows()
        ]
        return stations

# --- 전역 함수 ---

def run_simulation(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers, truck_step_freqency, num_days_in_month):
    overall_start_time = time.time()
    print("\n=== 시뮬레이션 시작 ===")
    sim = Simulator(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers, truck_step_freqency, num_days_in_month)
    sim.prepare_simulation()
    sim.run_simulation()
    of = sim.analyze_results()
    total_duration = time.time() - overall_start_time
    print(f"\n=== 총 실행 시간: {total_duration:.2f}초 ({total_duration/60:.2f}분) ===")
    return of

def load_car_path_df(car_paths_folder, number_of_trucks, target_year=2020, target_month_selection=-1, estimated_areas=33):
    load_start_time = time.time()
    print(f"--- 차량 경로 데이터 로딩 시작 (목표 트럭 수: {number_of_trucks}, 지역 대표 수: {estimated_areas}) ---")

    if target_month_selection == -1:
        subfolders = [d for d in os.listdir(car_paths_folder) if os.path.isdir(os.path.join(car_paths_folder, d)) and re.match(r"\d{4}-\d{2}", d)]
        available_2020_months = [f for f in subfolders if f.startswith(str(target_year))]
        if not available_2020_months: raise FileNotFoundError(f"{car_paths_folder} 내에서 {target_year}년도 월별 하위 폴더를 찾을 수 없습니다.")
        target_month_str = random.choice(available_2020_months)
    else:
        target_month_str = f"{target_year}-{target_month_selection:02d}"

    selected_folder_path = os.path.join(car_paths_folder, target_month_str)
    print(f"  데이터 로딩 경로: {selected_folder_path}")

    parquet_files = [os.path.join(selected_folder_path, f) for f in os.listdir(selected_folder_path) if f.endswith(".parquet")]
    if not parquet_files: raise FileNotFoundError(f"Parquet 파일을 찾을 수 없습니다: {selected_folder_path}")

    print(f"  OBU/AREA 메타데이터 수집 중...")
    all_obu_data = []
    for file_path in parquet_files:
        table = pq.read_table(file_path, columns=['OBU_ID', 'AREA_ID', 'CUMULATIVE_LINK_LENGTH'])
        df_partial = table.to_pandas()
        last_entries = df_partial.loc[df_partial.groupby('OBU_ID')['CUMULATIVE_LINK_LENGTH'].idxmax()]
        all_obu_data.extend(last_entries[['OBU_ID', 'AREA_ID', 'CUMULATIVE_LINK_LENGTH']].values.tolist())
    
    if not all_obu_data:
        raise ValueError("어떤 파일에서도 OBU 정보를 추출할 수 없었습니다.")

    all_obu_df = pd.DataFrame(all_obu_data, columns=['OBU_ID', 'AREA_ID', 'MAX_CUMUL_DIST']).drop_duplicates(subset=['OBU_ID'])
    all_obu_ids = set(all_obu_df['OBU_ID'])
    unique_area_ids = all_obu_df['AREA_ID'].unique()
    print(f"  총 {len(all_obu_ids)}개의 고유 OBU_ID와 {len(unique_area_ids)}개의 고유 AREA_ID 발견.")

    selected_obu_ids = set()
    num_area_select = min(len(unique_area_ids), estimated_areas)
    area_sampled_ids = set()
    if num_area_select > 0:
        all_obu_df_sorted = all_obu_df.sort_values('MAX_CUMUL_DIST', ascending=False)
        area_groups = all_obu_df_sorted.groupby('AREA_ID')
        for area_id, group in area_groups:
            if len(area_sampled_ids) < num_area_select:
                area_sampled_ids.add(group['OBU_ID'].iloc[0])
            else:
                break
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
        ids_to_remove = random.sample(ids_to_remove_from, min(excess, len(ids_to_remove_from)))
        selected_obu_ids -= set(ids_to_remove)

    selected_obu_ids_set = selected_obu_ids
    print(f"  샘플링 완료. 로드할 최종 OBU_ID {len(selected_obu_ids_set)}개 선택.")

    car_paths_list = []
    print(f"  선택된 OBU_ID 데이터 로딩 중...")
    for file_path in parquet_files:
        try:
            df_filtered = pd.read_parquet(file_path, filters=[('OBU_ID', 'in', selected_obu_ids_set)])
            if not df_filtered.empty:
                car_paths_list.append(df_filtered)
        except Exception as e:
            print(f"Warning: {file_path} 파일 처리 중 오류 발생: {e}")

    if not car_paths_list:
        raise ValueError("선택된 OBU_ID에 대한 데이터를 로드하지 못했습니다.")

    car_paths_df = pd.concat(car_paths_list, ignore_index=True)
    del car_paths_list, all_obu_df, all_obu_data
    gc.collect()

    if 'OBU_ID' in car_paths_df.columns:
        car_paths_df['OBU_ID'] = car_paths_df['OBU_ID'].astype(str)

    print(f"--- 차량 경로 데이터 로딩 및 샘플링 완료 ({time.time() - load_start_time:.2f}초 소요), {car_paths_df['OBU_ID'].nunique()}대 트럭 데이터 반환. ---")
    return car_paths_df

def load_station_df(station_solution_path, station_features_path):
    solution_df = pd.read_csv(station_solution_path, sep=',')
    solution_df.columns = solution_df.columns.str.strip().str.lower()
    features_df = pd.read_csv(station_features_path, sep=',')
    features_df.columns = features_df.columns.str.strip().str.lower()
    station_df = pd.merge(solution_df[['link_id', 'num_of_charger']], features_df.drop(columns=['num_of_charger'], errors='ignore'), on='link_id', how='left')
    station_df['link_id'] = station_df['link_id'].astype(int)
    station_df['num_of_charger'] = station_df['num_of_charger'].astype(int)
    print("충전소 데이터 로딩 및 병합 완료.")
    return station_df

if __name__ == '__main__':
    # --- 파일 경로 설정 ---
    car_paths_folder_monthly_full = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(MONTH_90KM)"
    station_solution_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Final_Candidates_Selected.csv"
    station_features_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\candidate\Final_Candidates\Final_Candidates_Selected.csv"

    # --- 시뮬레이션 파라미터 설정 ---
    unit_time = 5
    truck_step_frequency = 3
    number_of_max_chargers = 10000

    # [사용자 설정] 시뮬레이션에 사용할 트럭 대수 및 지역 대표 수 지정
    number_of_trucks_to_run = 5946
    estimated_areas_to_represent = 33

    # --- 시뮬레이션 기간 설정 ---
    target_year = 2020
    target_month = 9

    num_days_in_month = calendar.monthrange(target_year, target_month)[1]
    simulating_hours = int(num_days_in_month * 1.25 * 24) 
    print(f"시뮬레이션 대상 월: {target_year}-{target_month:02d}, 총 일수: {num_days_in_month}일, 총 시간: {simulating_hours}h")

    # --- 데이터 로딩 ---
    print("\n--- 데이터 로딩 시작 ---")
    data_load_start = time.time()
    car_paths_df = load_car_path_df(
        car_paths_folder_monthly_full,
        number_of_trucks=number_of_trucks_to_run,
        target_year=target_year,
        target_month_selection=target_month,
        estimated_areas=estimated_areas_to_represent
    )
    station_df = load_station_df(station_solution_path, station_features_path)
    print(f"--- 데이터 로딩 완료 ({time.time() - data_load_start:.2f}초 소요) ---")

    # --- 시뮬레이션 실행 ---
    if car_paths_df is not None and not car_paths_df.empty and station_df is not None and not station_df.empty:
        gc.collect()
        run_simulation(
            car_paths_df,
            station_df,
            unit_time,
            simulating_hours,
            number_of_trucks_to_run,
            number_of_max_chargers,
            truck_step_frequency,
            num_days_in_month
        )
    else:
        print("\n--- 데이터 로딩 실패 또는 유효한 데이터 없음으로 시뮬레이션 중단 ---")