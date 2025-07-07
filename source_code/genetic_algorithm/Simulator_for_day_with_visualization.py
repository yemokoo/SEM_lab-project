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

seed = time.time_ns() % (2**31 - 1)  # 현재 시간을 기반으로 시드 생성
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
    def __init__(self, car_paths_df, station_df, unit_minutes, simulating_hours, number_of_trucks, number_of_max_chargers, truck_step_frequency):
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
        self.truck_step_frequency = truck_step_frequency # 트럭 행동 결정 빈도 (스텝 단위)

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
        operational_station_link_ids = {s.link_id for s in self.stations if s.num_of_chargers > 0}
        if 'EVCS' not in self.car_paths_df.columns:
            self.car_paths_df['EVCS'] = 0
        self.car_paths_df['EVCS'] = np.where(self.car_paths_df['LINK_ID'].isin(operational_station_link_ids), 1, 0)

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
            if step_num % self.truck_step_frequency == 0:
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

        for station in self.stations:
            station.finalize_unprocessed_trucks(self.current_time)
            
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
             'queue_history_raw': station.queue_history,
             'charging_history_raw': station.charging_history, # 충전기별 충전 상태 기록 추가
             'cumulative_arrivals_history': station.cumulative_arrivals_history,
             'cumulative_departures_history': station.cumulative_departures_history,
            }
            for station in self.stations
        ]
        self.station_results_df = pd.DataFrame(station_data)

        # 시뮬레이션 결과가 담긴 truck_results_df 실패 그룹으로 분리합니다.
        if self.truck_results_df is None or self.truck_results_df.empty:
            # 결과가 없는 경우, 분석을 위해 빈 데이터프레임을 생성합니다.
            self.failed_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else [])
        else:
            # 실패 트럭: destination_reached가 False이고, 배터리 부족 또는 시뮬레이션 시간 초과로 중단된 경우
            self.failed_trucks_df = self.truck_results_df[
                (self.truck_results_df['destination_reached'] == False) & (self.truck_results_df['stopped_due_to_low_battery'] == True)             
            ].copy()
        print(f"  실패 트럭 수: {len(self.failed_trucks_df)}대")

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
        charger_installation_cost_per_unit = 96000000

        for station in self.stations:
            num_chargers = station.num_of_chargers
            if num_chargers == 0:
                charger_cost, station_capex = 0, 0
            else:
                charger_cost = (charger_installation_cost_per_unit * num_chargers) / daily_divider
                station_capex = charger_cost
            
            capex_results.append({
                'station_id': station.station_id, 
                'charger_cost': charger_cost, 
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
        시뮬레이션 결과에 기반하여 다양한 유형의 페널티를 계산합니다.
        - 미도착 트럭 페널티, 도착 트럭 지연 페널티, 충전기 수 초과 페널티, 대기 시간 페널티
        """
        # --- 1. 트럭 관련 페널티 계산 ---
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

            total_truck_penalty = failed_truck_penalty

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
            'truck_penalty': total_truck_penalty,
            'charger_penalty': charger_penalty,
            'waiting_penalty': total_waiting_penalty,
            'total_penalty': total_penalty
        }
        summary_df = pd.DataFrame([summary_results])
        station_penalty_df = pd.DataFrame(list(station_waiting_penalties.items()), columns=['station_id', 'waiting_penalty'])
        
        return summary_df, station_penalty_df
    

    def calculate_of(self):
        if self.station_results_df is None or self.station_results_df.empty:
            print("Warning: station_results_df가 비어있어 OF 계산을 중단합니다.")
            return 0

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

        truck_p = penalty_summary_df['truck_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        failed_truck_p = penalty_summary_df['failed_truck_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        charger_p = penalty_summary_df['charger_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        waiting_p = penalty_summary_df['waiting_penalty'].iloc[0] if not penalty_summary_df.empty else 0
        
        # 최종 재무 요약을 출력합니다.
        print(f"\n--- Financial Summary ---")
        print(f"Total Revenue                : {total_revenue:,.0f}")
        print(f"Total OPEX                   : {total_opex:,.0f}")
        print(f"Total CAPEX                  : {total_capex:,.0f}")
        print(f"Total Penalty                : {total_penalty:,.0f}")
        print(f"  ├─ Truck Penalty (Total)  : {truck_p:,.0f}")
        print(f"  │  └─ Failed Truck Penalty : {failed_truck_p:,.0f}")
        print(f"  ├─ Charger Penalty          : {charger_p:,.0f}")
        print(f"  └─ Waiting Penalty          : {waiting_p:,.0f}")
        print(f"------------------------------------")
        print(f"Objective Function (OF) Value: {of_value:,.0f}")
        print(f"------------------------------------")

        base_save_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Result"
        current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M")
        timestamped_folder_path = os.path.join(base_save_path, current_timestamp_str)
        os.makedirs(timestamped_folder_path, exist_ok=True)
        print(f"결과가 다음 경로에 저장됩니다: {timestamped_folder_path}")

        self.station_results_df.to_csv(os.path.join(timestamped_folder_path, "station_operational_summary.csv"), index=False, encoding='utf-8-sig')
        merged_df.to_csv(os.path.join(timestamped_folder_path, "financial_summary_by_station.csv"), index=False, encoding='utf-8-sig')
        print(f"결과 파일 저장 완료.")

        if 'station_id' not in merged_df.columns or merged_df['station_id'].isnull().all():
            print("오류: 'station_id'가 없어 그래프를 생성할 수 없습니다.")
            return of_value
        
        merged_df.sort_values('station_id', inplace=True)
        financial_station_ids_int = merged_df['station_id']
        financial_x_labels_str = financial_station_ids_int.astype(str)

        def set_xticks_by_50(ax, station_ids_int):
            if station_ids_int.empty: return
            unique_sorted_ids = station_ids_int.unique()
            min_id, max_id = unique_sorted_ids[0], unique_sorted_ids[-1]
            ticks_to_show = [sid for sid in unique_sorted_ids if sid % 50 == 0]
            if not ticks_to_show:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
            else:
                if min_id not in ticks_to_show: ticks_to_show.insert(0, min_id)
                if max_id not in ticks_to_show and max_id != min_id: ticks_to_show.append(max_id)
                ax.set_xticks(sorted(list(set(ticks_to_show))))
                ax.set_xticklabels([str(t) for t in sorted(list(set(ticks_to_show)))], rotation=90, ha='right')

        fig1, ax1 = plt.subplots(figsize=(18, 9))
        ax1.bar(financial_x_labels_str, merged_df['revenue'], label='Revenue', color='green')
        neg_opex = -merged_df['opex']
        neg_capex = -merged_df['capex']
        neg_waiting_penalty = -merged_df['waiting_penalty']
        ax1.bar(financial_x_labels_str, neg_opex, label='OPEX', color='orangered')
        ax1.bar(financial_x_labels_str, neg_capex, bottom=neg_opex, label='CAPEX', color='darkred')
        ax1.bar(financial_x_labels_str, neg_waiting_penalty, bottom=neg_opex + neg_capex, label='Waiting Penalty', color='gold')
        penalty_text = f"Truck Penalty: {round(truck_p)}\nCharger Penalty: {round(charger_p)}\nWaiting Penalty: {round(waiting_p)}"
        ax1.text(0.98, 0.98, penalty_text, ha='right', va='top', transform=ax1.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
        ax1.set_xlabel('Station ID'); ax1.set_ylabel('Amount (KRW)'); ax1.set_title('Financial Components by Station')
        ax1.legend(loc='best'); set_xticks_by_50(ax1, financial_station_ids_int)
        ax1.axhline(0, color='black', linewidth=0.8); ax1.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(timestamped_folder_path, "station_financial_components.png"))
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(18, 9))
        net_profit_colors = ['mediumseagreen' if x >= 0 else 'tomato' for x in merged_df['net_profit_before_penalty']]
        ax2.bar(financial_x_labels_str, merged_df['net_profit_before_penalty'], label='Net Profit (Before Penalty)', color=net_profit_colors)
        ax2.set_xlabel('Station ID'); ax2.set_ylabel('Net Profit'); ax2.set_title('Net Profit by Station (Before Penalty)')
        ax2.legend(loc='best'); set_xticks_by_50(ax2, financial_station_ids_int)
        ax2.axhline(0, color='black', linewidth=0.8); ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(timestamped_folder_path, "station_net_profit_before_penalty.png"))
        plt.close(fig2)

        if self.station_results_df is not None and not self.station_results_df.empty:
            op_df = pd.merge(self.station_results_df, station_penalty_df, on='station_id', how='left').fillna(0)
            op_df['station_id'] = op_df['station_id'].astype(int)
            op_df.sort_values('station_id', inplace=True)
            op_station_ids_int = op_df['station_id']
            op_x_labels_str = op_station_ids_int.astype(str)

            graph_configs = [
                {'y_col': 'total_charged_energy_kWh', 'title': 'Total Charged Energy per Station', 'ylabel': 'Total Charged Energy (kWh)', 'color': 'dodgerblue', 'avg_color': 'red'},
                {'y_col': 'total_charging_events', 'title': 'Total Charging Events per Station', 'ylabel': 'Total Charging Events', 'color': 'mediumpurple', 'avg_color': 'darkmagenta'},
                {'y_col': 'avg_waiting_time_min', 'title': 'Average Waiting Time per Station', 'ylabel': 'Average Waiting Time (minutes)', 'color': 'teal', 'avg_color': 'darkcyan'},
                {'y_col': 'num_of_charger', 'title': 'Number of Chargers per Station', 'ylabel': 'Number of Chargers', 'color': 'goldenrod', 'avg_color': 'darkgoldenrod'},
                {'y_col': 'waiting_penalty', 'title': 'Waiting Time Penalty per Station', 'ylabel': 'Waiting Time Penalty (KRW)', 'color': 'lightcoral', 'avg_color': 'darkred'}
            ]

            for config in graph_configs:
                fig, ax = plt.subplots(figsize=(18, 9))
                ax.bar(op_x_labels_str, op_df[config['y_col']], label=config['ylabel'], color=config['color'])
                
                if config['y_col'] == 'avg_waiting_time_min':
                    filtered_df = op_df[op_df['num_of_charger'] > 0]
                    avg_val = filtered_df[config['y_col']].mean() if not filtered_df.empty else 0
                else:
                    avg_val = op_df[config['y_col']].mean()

                ax.axhline(y=avg_val, color=config['avg_color'], linestyle='--', linewidth=1.5, label=f'Average: {avg_val:.2f}')
                ax.set_xlabel('Station ID'); ax.set_ylabel(config['ylabel']); ax.set_title(config['title'])
                ax.legend(loc='best'); set_xticks_by_50(ax, op_station_ids_int); ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(timestamped_folder_path, f"station_{config['y_col']}.png"))
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(18, 9))
            ax.bar(op_x_labels_str, op_df['max_queue_length'], label='Max Queue Length', color='#1f77b4')
            ax.bar(op_x_labels_str, op_df['avg_queue_length'], label='Average Queue Length', color='#ff7f0e')
            avg_queue = op_df['avg_queue_length'].mean()
            ax.axhline(y=avg_queue, color='red', linestyle='--', linewidth=1.5, label=f'Overall Avg Queue Length: {avg_queue:.2f}')
            ax.set_xlabel('Station ID'); ax.set_ylabel('Queue Length (Number of Trucks)'); ax.set_title('Average and Max Queue Length per Station')
            ax.legend(loc='best'); set_xticks_by_50(ax, op_station_ids_int); ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_queue_lengths.png"))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 8))
            plot_data = op_df[op_df['num_of_charger'] > 0].copy()
            plot_data['num_of_charger_jitter'] = plot_data['num_of_charger'] + np.random.normal(0, 0.1, size=len(plot_data))
            
            ax.scatter(
                plot_data['num_of_charger_jitter'], 
                plot_data['avg_waiting_time_min'], 
                alpha=0.6,
                s=50,
                label='Stations'
            )

            if not plot_data.empty:
                x_data = plot_data['num_of_charger']
                y_data = plot_data['avg_waiting_time_min']
                
                slope, intercept = np.polyfit(x_data, y_data, 1)
                
                x_trend = np.array(sorted(x_data.unique()))
                ax.plot(x_trend, slope * x_trend + intercept, color='red', linestyle='--', 
                        label=f'Trend (y={slope:.2f}x + {intercept:.2f})')

                # --- 추가된 부분 시작 ---
                avg_wait_time = plot_data['avg_waiting_time_min'].mean()
                ax.axhline(y=avg_wait_time, color='green', linestyle=':', linewidth=2, 
                        label=f'Average Waiting Time: {avg_wait_time:.2f} min')
                # --- 추가된 부분 끝 ---

            ax.set_xlabel('Number of Chargers per Station', fontsize=12)
            ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12)
            ax.set_title('Relationship between Number of Chargers and Average Waiting Time', fontsize=16)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
            
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()

            fig.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_chargers_vs_wait_time_scatter.png"), dpi=300)
            plt.close(fig)

        # --- 개별 충전소 대기열/점유율 그래프 생성 ---
        if 'queue_history_raw' in self.station_results_df.columns and 'charging_history_raw' in self.station_results_df.columns:
            # 1. 그래프 저장을 위한 전용 폴더 생성
            graph_folder = os.path.join(timestamped_folder_path, "station_occupancy_graphs")
            os.makedirs(graph_folder, exist_ok=True)
            print(f"충전소별 점유/대기열 그래프가 다음 경로에 저장됩니다: {graph_folder}")
            
            # 2. 활동이 있었던 충전소만 필터링 (대기 또는 충전이 한 번이라도 발생)
            stations_with_activity = self.station_results_df[
                self.station_results_df.apply(lambda row: (len(row['queue_history_raw']) > 0 and pd.Series(row['queue_history_raw']).max() > 0) or \
                                                         (len(row['charging_history_raw']) > 0 and pd.Series(row['charging_history_raw']).max() > 0), axis=1)
            ]
            
            # 3. 각 충전소별로 그래프 생성 루프
            for index, row in stations_with_activity.iterrows():
                station_id = int(row['station_id'])
                queue_history = row['queue_history_raw']
                charging_history = row['charging_history_raw']
                num_chargers = int(row['num_of_charger'])
                
                fig, ax = plt.subplots(figsize=(15, 7))
                time_steps = np.arange(len(queue_history)) * self.unit_minutes

                # 📊 그래프 그리기
                # (1) 충전 중인 트럭 수를 막대 그래프로 표시
                ax.bar(time_steps, charging_history, width=self.unit_minutes, color='skyblue', alpha=0.8, label=f'Charging Trucks')
                
                # (2) 대기 중인 트럭 수를 꺾은선 그래프로 표시
                ax.plot(time_steps, queue_history, marker='o', color='orangered', linestyle='-', markersize=4, label='Queued Trucks')

                # (3) 전체 충전기 용량을 나타내는 점선 추가
                ax.axhline(y=num_chargers, color='dodgerblue', linestyle='--', linewidth=1.5, label=f'Capacity ({num_chargers} Chargers)')
                
                # 🖼️ 그래프 디자인
                ax.set_title(f'Station {station_id}: Occupancy & Queue History', fontsize=16, weight='bold')
                ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
                ax.set_ylabel('Number of Trucks', fontsize=12)
                ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
                ax.legend(loc='upper left')
                
                # 축 설정
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Y축 눈금을 정수로
                
                # 💾 파일 저장
                file_name = f"station_{station_id}_occupancy_queue.png"
                save_path = os.path.join(graph_folder, file_name)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close(fig) # 메모리 해제를 위해 그래프 닫기

            print(f"{len(stations_with_activity)}개 충전소의 점유/대기열 추이 그래프 저장 완료.")
              
        if 'cumulative_arrivals_history' in self.station_results_df.columns:
            # 1. 그래프 저장을 위한 전용 폴더 생성
            graph_folder = os.path.join(timestamped_folder_path, "cumulative_queue_graphs")
            os.makedirs(graph_folder, exist_ok=True)
            print(f"누적 대기열 다이어그램이 다음 경로에 저장됩니다: {graph_folder}")

            stations_with_activity = self.station_results_df[
                self.station_results_df['cumulative_arrivals_history'].apply(lambda x: len(x) > 1 and pd.Series(x).max() > 0)
            ]
            
            for index, row in stations_with_activity.iterrows():
                station_id = int(row['station_id'])
                arrivals = row['cumulative_arrivals_history']
                num_chargers = int(row['num_of_charger'])
                departures = row['cumulative_departures_history']
                
                fig, ax = plt.subplots(figsize=(15, 7))
                time_steps = np.arange(len(arrivals)) * self.unit_minutes

                # 📊 그래프 그리기
                # (1) 누적 도착 곡선 (계단식으로 표현)
                ax.plot(time_steps, arrivals, drawstyle='steps-post', color='blue', label='Cumulative Arrivals')
                
                # (2) 누적 출발 곡선 (계단식으로 표현)
                ax.plot(time_steps, departures, drawstyle='steps-post', color='green', label='Cumulative Departures (Charging Start)')

                # (3) 대기열 영역을 반투명하게 채우기
                ax.fill_between(time_steps, arrivals, departures, step='post', color='gray', alpha=0.3, label='waiting time')
                
                # 🖼️ 그래프 디자인
                ax.set_title(f'Cumulative Queuing Diagram for Station {station_id}', fontsize=16, weight='bold')
                ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
                ax.set_ylabel('Cumulative Number of Trucks', fontsize=12)
                ax.grid(True, which='major', linestyle='--', linewidth=0.5)
                ax.legend(loc='upper left')
                
                # 축 설정
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0, right=time_steps[-1])
                
                # 💾 파일 저장
                file_name = f"station_{station_id}_cumulative_diagram.png"
                save_path = os.path.join(graph_folder, file_name)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close(fig) # 메모리 해제

            print(f"{len(stations_with_activity)}개 충전소의 누적 대기열 다이어그램 저장 완료.")
                
            print(f"운영 관련 그래프 저장 완료.")

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
                charger_specs=[{'power': 200, 'rate': 560}] * int(row['num_of_charger']),
                unit_minutes=self.unit_minutes 
            )
            for idx, row in df.iterrows() 
        ]
        return stations


# --- 전역 함수 ---

def run_simulation(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers, truck_step_freqency):
    """
    시뮬레이션을 준비, 실행하고 최종 OF 값을 반환합니다. 실행 시간도 측정합니다.
    """
    overall_start_time = time.time()
    print("\n=== 시뮬레이션 시작 ===") 
    sim = Simulator(car_paths_df, station_df, unit_minutes, simulating_hours, num_trucks, num_chargers, truck_step_freqency)
    
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
    car_paths_folder = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_90km)"
    station_file_path = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Final_Candidates_Selected.csv"

    simulating_hours = 36
    unit_time = 5 
    truck_step_frequency = 3
    number_of_trucks = 5946
    number_of_max_chargers = 10000 

    print("--- 데이터 로딩 시작 ---") 
    data_load_start = time.time()
    car_paths_df = load_car_path_df(car_paths_folder, number_of_trucks, estimated_areas=33)
    station_df = load_station_df(station_file_path)
    data_load_end = time.time()
    print(f"--- 데이터 로딩 완료 ({data_load_end - data_load_start:.2f}초 소요) ---") 

    # 데이터 로딩 성공 여부 확인 (데이터가 비어있지 않은지)
    if car_paths_df is not None and not car_paths_df.empty and station_df is not None and not station_df.empty:
        gc.collect() 
        run_simulation(car_paths_df, station_df, unit_time, simulating_hours, number_of_trucks, number_of_max_chargers, truck_step_frequency)
    else:
        print("\n--- 데이터 로딩 실패 또는 유효한 데이터 없음으로 시뮬레이션 중단 ---") 
