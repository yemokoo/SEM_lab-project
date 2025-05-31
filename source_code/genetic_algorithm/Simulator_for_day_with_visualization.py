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

        truck_creation_end_time = time.time()
        print(f"  {len(self.trucks)}개의 트럭 에이전트 생성 완료 ({truck_creation_end_time - truck_creation_start_time:.2f}초 소요).")

        self.current_time = 0
        gc.collect()


    def run_simulation(self):
        """
        시뮬레이션을 실행합니다.
        """
        total_steps = self.simulating_hours * (60 // self.unit_minutes)
        run_start_time = time.time()
        print(f"\n--- 시뮬레이션 시작 (총 {total_steps} 스텝, 단위 시간: {self.unit_minutes}분) ---")
        print(f"시뮬레이션 총 시간: {self.simulating_hours}시간 ({self.simulating_hours * 60}분)")

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
                        try:
                            truck.step(self.current_time)
                        except Exception as e:
                            print(f"ERROR: Truck {truck.unique_id} step failed at time {self.current_time}: {e}")
                            # 오류 발생 시 해당 트럭을 강제 종료하거나 다른 오류 처리 로직 추가 가능
                            # truck.stop() # 예: 오류 발생 시 강제 종료

            # 시간 증가
            self.current_time += self.unit_minutes
            step_end_time = time.time()

            # 스텝별 정보 출력 (너무 자주 출력되면 성능에 영향 줄 수 있음)
            #print(f"  스텝 {step_num + 1}/{total_steps} 완료. 현재 시뮬레이션 시간: {self.current_time - self.unit_minutes:.0f}분. 활성 트럭: {len(self.trucks)}. 스텝 소요 시간: {step_end_time - step_start_time:.3f}초.")

        loop_end_time = time.time()
        print(f"--- 시뮬레이션 주 루프 종료 ({loop_end_time - run_start_time:.2f}초 소요) ---")
        # 루프 종료 직후의 시간은 self.current_time 이지만, 실제 시뮬레이션 상 마지막으로 "처리된" 시간은 그 이전임
        print(f"최종 시뮬레이션 시간 도달 (처리 완료된 시간): {self.current_time - self.unit_minutes:.0f}분")


        # --- 최종 정리 단계 ---
        print(f"\n--- 시뮬레이션 최종 정리 시작 ---")
        # self.trucks 리스트는 Truck.stop()에 의해 변경되므로 복사본 사용
        final_cleanup_trucks = list(self.trucks)
        cleaned_up_count = 0
        if not final_cleanup_trucks:
            print("  정리할 트럭 없음 (모든 트럭이 이미 stopped 상태이거나 제거됨).")
        else:
            print(f"  정리 대상 트럭 수 (루프 종료 후): {len(final_cleanup_trucks)}")
            for truck_to_cleanup in final_cleanup_trucks:
                # truck.stop() 내부에서 self.status를 'stopped'로 먼저 바꾸므로 중복 호출은 방지됨.
                if truck_to_cleanup.status != 'stopped':
                    print(f"  최종 정리: 트럭 {truck_to_cleanup.unique_id} (상태: {truck_to_cleanup.status}, 현재 SOC: {truck_to_cleanup.SOC:.2f}%, 최종 활성화 예정 시간: {truck_to_cleanup.next_activation_time:.2f}분) 강제 종료 중...")
                    try:
                        # truck.stop()은 Truck 내부의 is_time_over와 유사한 역할을 여기서 수행
                        # Truck.stop()은 내부적으로 self.model.remove_truck(self)를 호출
                        truck_to_cleanup.stop() 
                        cleaned_up_count +=1
                    except Exception as e:
                        print(f"ERROR: Truck {truck_to_cleanup.unique_id} final stop failed: {e}")
                # else: # 이미 stopped 상태인 경우 (정상 종료 또는 이전 스텝에서 stop됨)
                    # print(f"  정보: 트럭 {truck_to_cleanup.unique_id}는 이미 'stopped' 상태입니다.")


        if cleaned_up_count > 0:
            print(f"--- 최종 정리 완료 ({cleaned_up_count}대 트럭 강제 종료) ---")
        else:
            print(f"--- 최종 정리 완료 (추가로 강제 종료된 트럭 없음) ---")
        
        print(f"시뮬레이션 종료 후 최종 활성 트럭 수: {len(self.trucks)}") # 최종 확인

        run_end_time = time.time() # 실제 run_simulation 종료 시간
        print(f"--- 시뮬레이션 전체 로직 종료 ({run_end_time - run_start_time:.2f}초 소요) ---")



    def remove_truck(self, truck):
        """
        시뮬레이터의 활성 트럭 리스트에서 특정 트럭 객체를 제거합니다.
        """
        if truck in self.trucks:
            self.trucks.remove(truck)


    def analyze_results(self):
        """
        시뮬레이션 결과를 분석하고 최종 OF 값을 계산합니다.
        """
        analysis_start_time = time.time()

        station_data = [
            {'station_id': station.station_id, 
             'link_id': station.link_id, # station_id 외에 link_id도 추가하면 유용할 수 있음
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

        # truck_results_df가 None이거나 비어있을 경우 처리
        if self.truck_results_df is None or self.truck_results_df.empty:
             self.failed_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else []) # 빈 DF 생성
        else:
            self.failed_trucks_df = self.truck_results_df[
                (self.truck_results_df['destination_reached'] == False) &
                (
                    (self.truck_results_df['stopped_due_to_low_battery'] == True) |
                    (self.truck_results_df['stopped_due_to_simulation_end'] == True)
                )
            ].copy()

        of = self.calculate_of()
        analysis_end_time = time.time()
        print(f"--- 결과 분석 완료 ({analysis_end_time - analysis_start_time:.2f}초 소요) ---")
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


    def calculate_penalty(self, failed_trucks_df, station_df):
        """
        위약금을 계산합니다.
        """
        truck_penalty = 0
        charger_penalty = 0
        
        # failed_trucks_df가 None이거나 비어있을 경우 처리
        if failed_trucks_df is not None and not failed_trucks_df.empty:
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
            
            truck_penalty = np.maximum(0, penalty).sum()

        number_of_total_chargers = sum(station.num_of_chargers for station in self.stations)

        if number_of_total_chargers > self.number_of_max_chargers:
            charger_cost_per_unit = 80000000
            charger_penalty = charger_cost_per_unit * (number_of_total_chargers - self.number_of_max_chargers)
        else:
            charger_penalty = 0
            
        total_penalty = truck_penalty + charger_penalty

        results = {'truck_penalty': truck_penalty, 'charger_penalty': charger_penalty, 'total_penalty': total_penalty}
        return pd.DataFrame([results])


    def calculate_of(self):
        """
        Calculates the Objective Function (OF) value.
        (Revenue - OPEX - CAPEX - Penalty)
        Saves financial summaries and operational graphs.
        All text in graphs will be in English.
        X-axis for station IDs will show ticks at multiples of 50.
        Operational graphs will include a horizontal line for the average value.
        """
        if self.station_results_df is None or self.station_results_df.empty:
            print("Warning: station_results_df is empty or not generated. OF calculation aborted, returning 0.")
            return 0

        revenue_df = self.calculate_revenue(self.station_results_df)
        opex_df = self.calculate_OPEX(self.station_results_df)
        capex_df = self.calculate_CAPEX(self.station_results_df)
        penalty_df = self.calculate_penalty(self.failed_trucks_df, self.station_results_df)

        merged_df = pd.merge(revenue_df[['station_id', 'revenue']],
                             opex_df[['station_id', 'opex']],
                             on='station_id', how='outer')
        merged_df = pd.merge(merged_df,
                             capex_df[['station_id', 'capex']],
                             on='station_id', how='outer')
        merged_df.fillna(0, inplace=True)
        
        # Ensure 'station_id' is integer for proper tick calculation
        if 'station_id' in merged_df.columns:
            try:
                merged_df['station_id'] = merged_df['station_id'].astype(int)
            except ValueError:
                print("Warning: Could not convert 'station_id' in merged_df to integer. X-axis ticks might not be as expected.")
        
        merged_df['net_profit_before_penalty'] = merged_df['revenue'] - merged_df['opex'] - merged_df['capex']

        total_revenue = merged_df['revenue'].sum()
        total_opex = merged_df['opex'].sum()
        total_capex = merged_df['capex'].sum()
        total_penalty = penalty_df['total_penalty'].iloc[0] if not penalty_df.empty and 'total_penalty' in penalty_df.columns else 0
        
        of_value = total_revenue - total_opex - total_capex - total_penalty
        of_value = round(of_value)

        truck_penalty_for_print = penalty_df['truck_penalty'].iloc[0] if not penalty_df.empty and 'truck_penalty' in penalty_df.columns else 0
        charger_penalty_for_print = penalty_df['charger_penalty'].iloc[0] if not penalty_df.empty and 'charger_penalty' in penalty_df.columns else 0
        
        print(f"\n--- Financial Summary ---")
        print(f"Total Revenue: {round(total_revenue)}")
        print(f"Total OPEX: {round(total_opex)}")
        print(f"Total CAPEX: {round(total_capex)}")
        print(f"Total Penalty: {round(total_penalty)} (Truck: {round(truck_penalty_for_print)}, Charger: {round(charger_penalty_for_print)})")
        print(f"Objective Function (OF) Value: {of_value}")
        print(f"-------------------")

        base_save_path = r"C:\Users\ADMIN\Desktop\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Result"
        current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M")
        timestamped_folder_path = os.path.join(base_save_path, current_timestamp_str)
        os.makedirs(timestamped_folder_path, exist_ok=True)
        print(f"Results will be saved in: {timestamped_folder_path}")

        station_results_filename = "station_operational_summary.csv"
        station_results_filepath = os.path.join(timestamped_folder_path, station_results_filename)
        if self.station_results_df is not None and not self.station_results_df.empty:
            self.station_results_df.to_csv(station_results_filepath, index=False, encoding='utf-8-sig')
            print(f"Station operational summary saved to: {station_results_filepath}")

        merged_df_filename = "financial_summary_by_station.csv"
        merged_df_filepath = os.path.join(timestamped_folder_path, merged_df_filename)
        merged_df.to_csv(merged_df_filepath, index=False, encoding='utf-8-sig')
        print(f"Financial summary by station saved to: {merged_df_filepath}")

        # --- Graphing ---
        # Prepare x-axis labels and ticks
        # Assuming station_id are integers and can be sorted for consistent ticking
        if 'station_id' not in merged_df.columns or merged_df['station_id'].isnull().all():
            print("Error: 'station_id' is missing or all null in merged_df. Cannot generate graphs.")
            return of_value
        
        # Convert station_id to string for categorical plotting, but use integer values for tick logic
        financial_station_ids_int = merged_df['station_id'].astype(int)
        financial_x_labels_str = financial_station_ids_int.astype(str) # For direct use in plt.bar if not setting custom ticks

        # Function to set x-axis ticks at multiples of 50
        def set_xticks_by_50(ax, station_ids_int):
            unique_sorted_ids = np.sort(station_ids_int.unique())
            if len(unique_sorted_ids) == 0:
                return

            min_id, max_id = unique_sorted_ids[0], unique_sorted_ids[-1]
            
            ticks_to_show_values = [sid for sid in unique_sorted_ids if sid % 50 == 0]
            
            # Ensure first and last are shown if not multiple of 50 and list is not empty
            if not ticks_to_show_values and len(unique_sorted_ids) > 0 : # If no multiples of 50, show some ticks
                 ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True)) # Show up to 10 ticks
            elif len(unique_sorted_ids) > 0:
                if min_id not in ticks_to_show_values:
                    ticks_to_show_values.insert(0, min_id)
                if max_id not in ticks_to_show_values and max_id != min_id:
                     # Check if max_id is already covered by a multiple of 50 close to it
                    if not any(abs(max_id - t) < 50 for t in ticks_to_show_values if t % 50 == 0) or max_id % 50 != 0 :
                        ticks_to_show_values.append(max_id)
                
                ticks_to_show_values = sorted(list(set(ticks_to_show_values))) # Remove duplicates and sort
                ax.set_xticks(ticks_to_show_values)
                ax.set_xticklabels([str(t) for t in ticks_to_show_values], rotation=90, ha='right')
            else: # Fallback if no data
                ax.set_xticks([])


        # Graph 1: Financial Components by Station
        fig1, ax1 = plt.subplots(figsize=(18, 9))
        ax1.bar(financial_x_labels_str, merged_df['revenue'], label='Revenue', color='green')
        ax1.bar(financial_x_labels_str, -merged_df['opex'], label='OPEX', color='orangered') # Changed color
        ax1.bar(financial_x_labels_str, -merged_df['capex'], label='CAPEX', color='darkred') # Changed color
        
        penalty_text_info = f"Total Truck Penalty: {round(truck_penalty_for_print)}\nTotal Charger Penalty: {round(charger_penalty_for_print)}"
        ax1.text(0.98, 0.98, penalty_text_info, ha='right', va='top', transform=ax1.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.7))
        ax1.set_xlabel('Station ID')
        ax1.set_ylabel('Amount')
        ax1.set_title('Financial Components by Station')
        ax1.legend(loc='best')
        set_xticks_by_50(ax1, financial_station_ids_int) # Apply custom ticks
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        graph_filename1 = "station_financial_components.png"
        file_path1 = os.path.join(timestamped_folder_path, graph_filename1)
        plt.savefig(file_path1)
        print(f"Graph 'Station Financial Components' saved to: {file_path1}")
        plt.close(fig1)

        # Graph 2: Net Profit (Before Penalty) by Station
        fig2, ax2 = plt.subplots(figsize=(18, 9))
        net_profit_colors = ['mediumseagreen' if x >= 0 else 'tomato' for x in merged_df['net_profit_before_penalty']]
        ax2.bar(financial_x_labels_str, merged_df['net_profit_before_penalty'], label='Net Profit (Before Penalty)', color=net_profit_colors)
        ax2.set_xlabel('Station ID')
        ax2.set_ylabel('Net Profit')
        ax2.set_title('Net Profit by Station (Before Penalty)')
        ax2.legend(loc='best')
        set_xticks_by_50(ax2, financial_station_ids_int) # Apply custom ticks
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        graph_filename2 = "station_net_profit_before_penalty.png"
        file_path2 = os.path.join(timestamped_folder_path, graph_filename2)
        plt.savefig(file_path2)
        print(f"Graph 'Net Profit by Station' saved to: {file_path2}")
        plt.close(fig2)

        # --- New Graphs from station_results_df ---
        if self.station_results_df is not None and not self.station_results_df.empty and 'station_id' in self.station_results_df.columns:
            try:
                operational_station_ids_int = self.station_results_df['station_id'].astype(int)
                operational_x_labels_str = operational_station_ids_int.astype(str)
            except ValueError:
                print("Warning: Could not convert 'station_id' in station_results_df to integer for operational graphs.")
                operational_x_labels_str = self.station_results_df['station_id'].astype(str) # Fallback to string
                operational_station_ids_int = pd.Series(range(len(self.station_results_df))) # Fallback for tick logic

            # Graph 3: Total Charged Energy per Station
            fig3, ax3 = plt.subplots(figsize=(18, 9))
            ax3.bar(operational_x_labels_str, self.station_results_df['total_charged_energy_kWh'], label='Total Charged Energy (kWh)', color='dodgerblue')
            avg_energy = self.station_results_df['total_charged_energy_kWh'].mean()
            ax3.axhline(y=avg_energy, color='red', linestyle='--', linewidth=1.5, label=f'Average Energy: {avg_energy:.2f} kWh')
            ax3.set_xlabel('Station ID')
            ax3.set_ylabel('Total Charged Energy (kWh)')
            ax3.set_title('Total Charged Energy per Station')
            ax3.legend(loc='best')
            set_xticks_by_50(ax3, operational_station_ids_int)
            ax3.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            graph_filename3 = "station_total_charged_energy.png"
            file_path3 = os.path.join(timestamped_folder_path, graph_filename3)
            plt.savefig(file_path3)
            print(f"Graph 'Total Charged Energy per Station' saved to: {file_path3}")
            plt.close(fig3)

            # Graph 4: Total Charging Events per Station
            fig4, ax4 = plt.subplots(figsize=(18, 9))
            ax4.bar(operational_x_labels_str, self.station_results_df['total_charging_events'], label='Total Charging Events', color='mediumpurple')
            avg_events = self.station_results_df['total_charging_events'].mean()
            ax4.axhline(y=avg_events, color='darkmagenta', linestyle='--', linewidth=1.5, label=f'Average Events: {avg_events:.2f}')
            ax4.set_xlabel('Station ID')
            ax4.set_ylabel('Total Charging Events')
            ax4.set_title('Total Charging Events per Station')
            ax4.legend(loc='best')
            set_xticks_by_50(ax4, operational_station_ids_int)
            ax4.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            graph_filename4 = "station_total_charging_events.png"
            file_path4 = os.path.join(timestamped_folder_path, graph_filename4)
            plt.savefig(file_path4)
            print(f"Graph 'Total Charging Events per Station' saved to: {file_path4}")
            plt.close(fig4)

            fig5, ax5 = plt.subplots(figsize=(18, 9))

            overall_max_queue_ever = self.station_results_df['max_queue_length'].max()
            overall_avg_queue_length = self.station_results_df['avg_queue_length'].mean() # Calculate once

            # 텍스트 정보 구성
            queue_summary_text = (f"Overall Max Queue: {overall_max_queue_ever:.2f}\n"
                                f"Overall Avg Queue (Line): {overall_avg_queue_length:.2f}")

            ax5.text(0.98, 0.98, # x, y 좌표 (0.98, 0.98은 오른쪽 상단 근처)
                    queue_summary_text,
                    ha='right', va='top', # 수평 오른쪽 정렬, 수직 위쪽 정렬
                    transform=ax5.transAxes, # ax5 축 기준 상대 좌표 사용
                    fontsize=9, # 폰트 크기 조절 가능
                    bbox=dict(boxstyle='round,pad=0.5', # 테두리 상자 스타일 (둥근 모서리, 내부 여백)
                            fc='lightyellow', # 배경색 (facecolor)
                            alpha=0.75))     # alpha는 bbox 딕셔너리 내부에 위치

            # 막대 그래프 앞뒤 배치 (Max Queue가 배경, Avg Queue가 전경)

            ax5.bar(operational_x_labels_str, self.station_results_df['max_queue_length'],
                     label='Max Queue Length', color='#1f77b4', zorder=1)
            ax5.bar(operational_x_labels_str, self.station_results_df['avg_queue_length'],
                     label='Average Queue Length', color='#ff7f0e', zorder=2)

            # axhline에 이미 계산된 overall_avg_queue_length 사용
            ax5.axhline(y=overall_avg_queue_length, color='red', linestyle='--', linewidth=1.5, label=f'Overall Avg Queue Length: {overall_avg_queue_length:.2f}')

            ax5.set_xlabel('Station ID')
            ax5.set_ylabel('Queue Length (Number of Trucks)')
            ax5.set_title('Average and Max Queue Length per Station')
            ax5.legend(loc='best')
            set_xticks_by_50(ax5, operational_station_ids_int)
            ax5.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_queue_lengths.png"))
            print(f"Graph 'Queue Lengths per Station' saved.")
            plt.close(fig5)

            # Graph 6: Average Waiting Time per Station
            fig6, ax6 = plt.subplots(figsize=(18, 9))
            ax6.bar(operational_x_labels_str, self.station_results_df['avg_waiting_time_min'], label='Average Waiting Time (min)', color='teal')
            avg_wait_time = self.station_results_df['avg_waiting_time_min'].mean()
            ax6.axhline(y=avg_wait_time, color='darkcyan', linestyle='--', linewidth=1.5, label=f'Overall Avg Wait Time: {avg_wait_time:.2f} min')
            ax6.set_xlabel('Station ID')
            ax6.set_ylabel('Average Waiting Time (minutes)')
            ax6.set_title('Average Waiting Time per Station')
            ax6.legend(loc='best')
            set_xticks_by_50(ax6, operational_station_ids_int)
            ax6.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            graph_filename6 = "station_avg_waiting_time.png"
            file_path6 = os.path.join(timestamped_folder_path, graph_filename6)
            plt.savefig(file_path6)
            print(f"Graph 'Average Waiting Time per Station' saved to: {file_path6}")
            plt.close(fig6)
        else:
            print("Info: station_results_df is empty or 'station_id' column is missing. Skipping operational graphs.")

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
    overall_start_time = time.time()
    print("\n=== 시뮬레이션 시작 ===") 
    sim = Simulator(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers)
    
    prepare_start = time.time()
    sim.prepare_simulation()
    prepare_end = time.time()
    print(f"--- 시뮬레이션 준비 완료 ({prepare_end - prepare_start:.2f}초 소요) ---") 

    run_start = time.time()
    sim.run_simulation()
    run_end = time.time()

    analyze_start = time.time()
    of = sim.analyze_results()
    analyze_end = time.time()

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"\n=== 총 실행 시간: {total_duration:.2f}초 ({total_duration/60:.2f}분) ===") 
    return of

def load_car_path_df(car_paths_folder, number_of_trucks, estimated_areas=33):
    """
    차량 경로 데이터(Parquet 형식)를 로드하고 샘플링하는 함수.
    파일/데이터 관련 오류가 발생하지 않는다고 가정합니다.
    """
    load_start_time = time.time()
    print(f"--- 차량 경로 데이터 로딩 시작 (목표 트럭 수: {number_of_trucks}) ---") 

    # 1. 날짜 폴더 선택
    subfolders = [d for d in os.listdir(car_paths_folder)
                  if os.path.isdir(os.path.join(car_paths_folder, d)) and len(d) == 10 and d[4] == '-' and d[7] == '-']
    if not subfolders:
         raise FileNotFoundError(f"{car_paths_folder} 에서 'YYYY-MM-DD' 형식의 하위 폴더를 찾을 수 없습니다.")
    random_subfolder = random.choice(subfolders)
    selected_folder_path = os.path.join(car_paths_folder, random_subfolder)
    print(f"  선택된 날짜 폴더: {random_subfolder}") 

    # 2. 전체 OBU_ID 및 AREA_ID 정보 수집
    parquet_files = []
    all_obu_data = []
    files_in_folder = [f for f in os.listdir(selected_folder_path) if f.endswith(".parquet")]
    if not files_in_folder:
         raise FileNotFoundError(f"{selected_folder_path} 에서 .parquet 파일을 찾을 수 없습니다.")
    
    print(f"  OBU/AREA 정보 수집 중...") 
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
    print(f"  총 {len(all_obu_ids)}개의 고유 OBU_ID와 {actual_num_areas}개의 고유 AREA_ID 발견.") 

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
    print(f"  로드할 OBU_ID {len(selected_obu_ids_set)}개 선택 완료.") 

    # 4. 선택된 OBU_ID 데이터 로드
    car_paths_list = []
    read_start_time = time.time()
    print(f"  선택된 OBU_ID 데이터 로딩 중...") 
    for file_path in parquet_files:
        filters = [('OBU_ID', 'in', list(selected_obu_ids_set))] 
        df_filtered = pd.read_parquet(file_path, engine='pyarrow', filters=filters) 
        
        if not df_filtered.empty:
            car_paths_list.append(df_filtered)
    read_end_time = time.time()
    print(f"  데이터 로딩 완료 ({read_end_time - read_start_time:.2f}초 소요).") 

    if not car_paths_list:
        raise ValueError("선택된 OBU_ID에 대한 데이터를 로드하지 못했습니다.")

    # 5. 데이터 병합
    concat_start_time = time.time()
    car_paths_df = pd.concat(car_paths_list, ignore_index=True)
    del car_paths_list 
    gc.collect()
    concat_end_time = time.time()
    print(f"  데이터 병합 완료 ({concat_end_time - concat_start_time:.2f}초 소요).") 

    # 6. 데이터 전처리
    preprocess_start_time = time.time()
    car_paths_df['DATETIME'] = pd.to_datetime(car_paths_df['DATETIME'], format='%H:%M', errors='coerce').dt.time
    first_times = car_paths_df.groupby('OBU_ID')['DATETIME'].transform('first')
    car_paths_df['START_TIME_MINUTES'] = first_times.apply(lambda x: x.hour * 60 + x.minute if pd.notnull(x) else np.nan)
    
    original_truck_count = car_paths_df['OBU_ID'].nunique()
    car_paths_df.dropna(subset=['START_TIME_MINUTES'], inplace=True)
    final_truck_count = car_paths_df['OBU_ID'].nunique()
    if original_truck_count != final_truck_count:
        print(f"  [정보] 유효하지 않은 시작 시간으로 인해 {original_truck_count - final_truck_count}개 트럭 경로 제거됨.") 
    
    preprocess_end_time = time.time()
    print(f"  데이터 전처리 완료 ({preprocess_end_time - preprocess_start_time:.2f}초 소요).") 

    # 7. 최종 반환
    load_end_time = time.time()
    print(f"--- 차량 경로 데이터 로딩 완료 (총 {load_end_time - load_start_time:.2f}초 소요) ---") 
    print(f"  최종 {final_truck_count}개 트럭 경로 데이터 반환.") 
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
    car_paths_folder = r"C:\Users\ADMIN\Desktop\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_stop_added)"
    station_file_path = r"C:\Users\ADMIN\Desktop\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Final_Candidates_Selected.csv"

    simulating_hours = 36
    unit_time = 20 
    number_of_trucks = 7262
    number_of_max_chargers = 2000 

    print("--- 데이터 로딩 시작 ---") 
    data_load_start = time.time()
    car_paths_df = load_car_path_df(car_paths_folder, number_of_trucks, estimated_areas=33)
    station_df = load_station_df(station_file_path)
    data_load_end = time.time()
    print(f"--- 데이터 로딩 완료 ({data_load_end - data_load_start:.2f}초 소요) ---") 

    # 데이터 로딩 성공 여부 확인 (데이터가 비어있지 않은지)
    if car_paths_df is not None and not car_paths_df.empty and station_df is not None and not station_df.empty:
        gc.collect() 
        run_simulation(car_paths_df, station_df, unit_time, simulating_hours, number_of_trucks, number_of_max_chargers)
    else:
        print("\n--- 데이터 로딩 실패 또는 유효한 데이터 없음으로 시뮬레이션 중단 ---") 
