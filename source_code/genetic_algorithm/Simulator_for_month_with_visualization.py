from datetime import datetime, timedelta
import calendar
import random
import warnings
from matplotlib import pyplot as plt, ticker
import pandas as pd
import numpy as np
import os
import gc
import time
from charger import Charger
from station import Station
from truck import Truck
import pyarrow.parquet as pq
import pyarrow as pa
import re
import seaborn as sns
import multiprocessing

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
        시뮬레이션을 실행합니다. (Just-In-Time 생성 및 시간 출력 기능 개선)
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
                print(f"--- Day {day}, {hour_of_day:02d}:00 (활성: {len(self.trucks)}, 대기: {len(self.pending_trucks)}, 실행 시간: {elapsed_seconds:.1f}s) ---")
                last_printed_hour = int(current_total_hours)

            trucks_to_activate = []
            while self.pending_trucks and self.pending_trucks[0]['start_time'] <= self.current_time:
                trucks_to_activate.append(self.pending_trucks.pop(0))

            if trucks_to_activate:
                for truck_data in trucks_to_activate:
                    group = truck_data['group']
                    new_truck = Truck(group, self.simulating_hours, self.link_id_to_station, self, 10)
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

    def analyze_results(self):
        """
        시뮬레이션 결과를 분석하고 최종 OF 값을 계산합니다.
        """
        analysis_start_time = time.time()

        station_data = []
        for station in self.stations:
            total_charged_energy_station = sum(c.total_charged_energy for c in station.chargers)
            total_charging_events_station = sum(c.charging_events_count for c in station.chargers)

            total_available_charger_minutes = self.simulating_hours * 60 * station.num_of_chargers
            total_charger_occupied_minutes = sum(c.total_charging_duration_minutes for c in station.chargers) if station.chargers and hasattr(station.chargers[0], 'total_charging_duration_minutes') else 0

            utilization_percentage = (total_charger_occupied_minutes / total_available_charger_minutes * 100) if total_available_charger_minutes > 0 else 0

            station_data.append({
                'station_id': station.station_id,
                'link_id': station.link_id,
                'num_of_charger': station.num_of_chargers,
                'total_charged_energy_kWh': total_charged_energy_station,
                'total_charging_events': total_charging_events_station,
                'avg_queue_length': np.mean(station.queue_history) if station.queue_history else 0,
                'max_queue_length': np.max(station.queue_history) if station.queue_history else 0,
                'avg_waiting_time_min': np.mean(station.waiting_times) if station.waiting_times else 0,
                'utilization_percentage': round(utilization_percentage, 2),
                'queue_history_raw': station.queue_history,
                'charging_history_raw': station.charging_history,
                'cumulative_arrivals_history': station.cumulative_arrivals_history,
                'cumulative_departures_history': station.cumulative_departures_history,
            })
        self.station_results_df = pd.DataFrame(station_data)

        if self.truck_results_df is None or self.truck_results_df.empty:
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
        모든 충전소의 OPEX(운영 비용)를 계산합니다. (월 단위 비용)
        """
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
        모든 충전소의 CAPEX(자본 비용)를 계산합니다. (월 단위 비용 기준)
        """
        capex_results = []
        LIFESPAN_YEARS = 5
        MONTHS_PER_YEAR = 12

        monthly_divider = LIFESPAN_YEARS * MONTHS_PER_YEAR
        charger_installation_cost_per_unit = 96000000

        for station in self.stations:
            num_chargers = station.num_of_chargers
            if num_chargers == 0:
                charger_cost, station_capex = 0, 0
            else:
                charger_cost = (charger_installation_cost_per_unit * num_chargers) / monthly_divider
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
        """
        failed_truck_penalty = 0.0
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
            failed_truck_penalty = np.maximum(0, penalty).sum()
        
        total_truck_penalty = failed_truck_penalty

        charger_penalty = 0.0
        number_of_total_chargers = sum(station.num_of_chargers for station in self.stations)
        if number_of_total_chargers > self.number_of_max_chargers:
            charger_cost_per_unit = 96000000
            charger_penalty = float(charger_cost_per_unit * (number_of_total_chargers - self.number_of_max_chargers))

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
        
        print(f"\n--- Financial Summary (Monthly) ---")
        print(f"Total Revenue                  : {total_revenue:,.0f}")
        print(f"Total OPEX                     : {total_opex:,.0f}")
        print(f"Total CAPEX                    : {total_capex:,.0f}")
        print(f"Total Penalty                  : {total_penalty:,.0f}")
        print(f"  ├─ Truck Penalty (Total)      : {truck_p:,.0f}")
        print(f"  │  └─ Failed Truck Penalty   : {failed_truck_p:,.0f}")
        print(f"  ├─ Charger Penalty          : {charger_p:,.0f}")
        print(f"  └─ Waiting Penalty            : {waiting_p:,.0f}")
        print(f"------------------------------------")
        print(f"Objective Function (OF) Value    : {of_value:,.0f}")
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

        print("\n--- 후보지 선정 기준 & 시뮬레이션 결과 통합 분석 시작 ---")
        
        analysis_df = pd.merge(
            self.station_results_df,
            self.station_df,
            on=['link_id', 'num_of_charger'],
            how='left'
        )
        analysis_df.fillna(0, inplace=True)
        analysis_df['station_id'] = analysis_df['station_id'].astype(int)

        plot_data_scatter = analysis_df[analysis_df['num_of_charger'] > 0].copy()

        if not plot_data_scatter.empty:
            fig, ax = plt.subplots(figsize=(16, 10))
            scatter1 = ax.scatter(plot_data_scatter['num_of_charger'], plot_data_scatter['avg_waiting_time_min'], c=plot_data_scatter['point'], cmap='viridis', alpha=0.7, s=80)
            ax.set_xlabel('Number of Chargers per Station', fontsize=12)
            ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12)
            ax.set_title('Chargers vs. Waiting Time (Colored by Candidate Point Score)', fontsize=16)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(*scatter1.legend_elements(), title="Point Score")
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_chargers_vs_wait_time_by_point.png"), dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(16, 10))
            scatter2 = ax.scatter(plot_data_scatter['num_of_charger'], plot_data_scatter['avg_waiting_time_min'], c=plot_data_scatter['traffic'], cmap='plasma', alpha=0.7, s=80)
            ax.set_xlabel('Number of Chargers per Station', fontsize=12)
            ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12)
            ax.set_title('Chargers vs. Waiting Time (Colored by Traffic Score)', fontsize=16)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(*scatter2.legend_elements(), title="Traffic Score")
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_chargers_vs_wait_time_by_traffic.png"), dpi=300)
            plt.close(fig)
            print("Point/Traffic 점수별 대기시간 Scatter Plot 2종 저장 완료.")
        
        plot_data_wait = analysis_df[analysis_df['num_of_charger'] > 0]
        if not plot_data_wait.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.boxplot(x=plot_data_wait['avg_waiting_time_min'], ax=ax, color='lightblue')
            ax.set_xlabel('Average Waiting Time (minutes)', fontsize=12)
            ax.set_title('Distribution of Average Waiting Time at Stations (where chargers > 0)', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_avg_wait_time_boxplot.png"), dpi=300)
            plt.close(fig)
            print("평균 대기시간 Boxplot 저장 완료.")

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.histplot(data=plot_data_wait, x='num_of_charger', discrete=True, shrink=0.8, ax=ax)
            ax.set_xlabel('Number of Chargers', fontsize=12)
            ax.set_ylabel('Number of Stations', fontsize=12)
            ax.set_title('Distribution of Installed Chargers per Station (where chargers > 0)', fontsize=16)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_charger_count_histogram.png"), dpi=300)
            plt.close(fig)
            print("충전기 대수 분포 Histogram 저장 완료.")

        correlation_df = pd.merge(analysis_df, merged_df[['station_id', 'revenue', 'net_profit_before_penalty']], on='station_id', how='left')
        correlation_cols = [
            'point', 'od', 'rest_area', 'traffic', 'infra', 'interval',
            'total_charged_energy_kWh', 'total_charging_events', 'avg_waiting_time_min',
            'utilization_percentage', 'revenue', 'net_profit_before_penalty'
        ]
        existing_cols_for_corr = [col for col in correlation_cols if col in correlation_df.columns]
        
        if len(existing_cols_for_corr) > 1:
            corr_matrix = correlation_df[existing_cols_for_corr].corr()
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5, ax=ax)
            ax.set_title('Correlation Matrix: Candidate Criteria vs. Simulation Results', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "correlation_heatmap_criteria_vs_results.png"), dpi=300)
            plt.close(fig)
            print("후보지 선정 기준과 시뮬레이션 결과 간의 상관관계 Heatmap 저장 완료.")
        else:
            print("상관관계 분석을 위한 데이터가 부족하여 Heatmap을 생성하지 않았습니다.")
            
        print("--- 통합 분석 완료 ---")
        
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
        penalty_text = f"Truck Penalty: {round(truck_p):,.0f}\nCharger Penalty: {round(charger_p):,.0f}\nWaiting Penalty: {round(waiting_p):,.0f}"
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
                
                if config['y_col'] == 'avg_waiting_time_min' or config['y_col'] == 'utilization_percentage':
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
            if not plot_data.empty:
                plot_data['num_of_charger_jitter'] = plot_data['num_of_charger'] + np.random.normal(0, 0.1, size=len(plot_data))
                ax.scatter(plot_data['num_of_charger_jitter'], plot_data['avg_waiting_time_min'], alpha=0.6, s=50, label='Stations')

                x_data = plot_data['num_of_charger']
                y_data = plot_data['avg_waiting_time_min']
                
                slope, intercept = np.polyfit(x_data, y_data, 1)
                
                x_trend = np.array(sorted(x_data.unique()))
                ax.plot(x_trend, slope * x_trend + intercept, color='red', linestyle='--', label=f'Trend (y={slope:.2f}x + {intercept:.2f})')

                avg_wait_time = plot_data['avg_waiting_time_min'].mean()
                ax.axhline(y=avg_wait_time, color='green', linestyle=':', linewidth=2, label=f'Average Waiting Time: {avg_wait_time:.2f} min')

            ax.set_xlabel('Number of Chargers per Station', fontsize=12)
            ax.set_ylabel('Average Waiting Time (minutes)', fontsize=12)
            ax.set_title('Relationship between Number of Chargers and Average Waiting Time', fontsize=16)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            fig.tight_layout()
            plt.savefig(os.path.join(timestamped_folder_path, "station_chargers_vs_wait_time_scatter_original.png"), dpi=300)
            plt.close(fig)

            if 'queue_history_raw' in self.station_results_df.columns and 'charging_history_raw' in self.station_results_df.columns:
                graph_folder = os.path.join(timestamped_folder_path, "station_occupancy_graphs")
                os.makedirs(graph_folder, exist_ok=True)
                print(f"충전소별 점유/대기열 그래프가 다음 경로에 저장됩니다: {graph_folder}")
                
                stations_with_activity = self.station_results_df[
                    self.station_results_df.apply(lambda row: (len(row['queue_history_raw']) > 0 and pd.Series(row['queue_history_raw']).max() > 0) or \
                                                             (len(row['charging_history_raw']) > 0 and pd.Series(row['charging_history_raw']).max() > 0), axis=1)
                ]
                
                for index, row in stations_with_activity.iterrows():
                    station_id = int(row['station_id'])
                    queue_history = row['queue_history_raw']
                    charging_history = row['charging_history_raw']
                    num_chargers = int(row['num_of_charger'])
                    
                    fig, ax = plt.subplots(figsize=(15, 7))
                    time_steps = np.arange(len(queue_history)) * self.unit_minutes

                    ax.bar(time_steps, charging_history, width=self.unit_minutes, color='skyblue', alpha=0.8, label=f'Charging Trucks')
                    ax.plot(time_steps, queue_history, marker='o', color='orangered', linestyle='-', markersize=4, label='Queued Trucks')
                    ax.axhline(y=num_chargers, color='dodgerblue', linestyle='--', linewidth=1.5, label=f'Capacity ({num_chargers} Chargers)')
                    
                    ax.set_title(f'Station {station_id}: Occupancy & Queue History (Monthly)', fontsize=16, weight='bold')
                    ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
                    ax.set_ylabel('Number of Trucks', fontsize=12)
                    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
                    ax.legend(loc='upper left')
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=0)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                    
                    file_name = f"station_{station_id}_occupancy_queue.png"
                    save_path = os.path.join(graph_folder, file_name)
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)

                print(f"{len(stations_with_activity)}개 충전소의 점유/대기열 추이 그래프 저장 완료.")
                
            if 'cumulative_arrivals_history' in self.station_results_df.columns:
                graph_folder = os.path.join(timestamped_folder_path, "cumulative_queue_graphs")
                os.makedirs(graph_folder, exist_ok=True)
                print(f"누적 대기열 다이어그램이 다음 경로에 저장됩니다: {graph_folder}")

                stations_with_activity = self.station_results_df[
                    self.station_results_df['cumulative_arrivals_history'].apply(lambda x: len(x) > 1 and pd.Series(x).max() > 0)
                ]
                
                for index, row in stations_with_activity.iterrows():
                    station_id = int(row['station_id'])
                    arrivals = row['cumulative_arrivals_history']
                    departures = row['cumulative_departures_history']
                    
                    fig, ax = plt.subplots(figsize=(15, 7))
                    time_steps = np.arange(len(arrivals)) * self.unit_minutes

                    ax.plot(time_steps, arrivals, drawstyle='steps-post', color='blue', label='Cumulative Arrivals')
                    ax.plot(time_steps, departures, drawstyle='steps-post', color='green', label='Cumulative Departures (Charging Start)')
                    ax.fill_between(time_steps, arrivals, departures, step='post', color='gray', alpha=0.3, label='waiting time')
                    
                    ax.set_title(f'Cumulative Queuing Diagram for Station {station_id} (Monthly)', fontsize=16, weight='bold')
                    ax.set_xlabel('Simulation Time (minutes)', fontsize=12)
                    ax.set_ylabel('Cumulative Number of Trucks', fontsize=12)
                    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
                    ax.legend(loc='upper left')
                    ax.set_ylim(bottom=0)
                    ax.set_xlim(left=0, right=time_steps[-1])
                    
                    file_name = f"station_{station_id}_cumulative_diagram.png"
                    save_path = os.path.join(graph_folder, file_name)
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)

                print(f"{len(stations_with_activity)}개 충전소의 누적 대기열 다이어그램 저장 완료.")
                
            print(f"운영 관련 그래프 저장 완료.")

        self.generate_monthly_summary_stats(timestamped_folder_path)
        self.generate_overall_traffic_graphs(timestamped_folder_path)

        return of_value

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

    def generate_monthly_summary_stats(self, output_path):
        print("\n--- 월별 전체 요약 통계 계산 시작 ---")
        total_active_trucks = self.number_of_trucks_actual

        successful_trucks = self.truck_results_df[self.truck_results_df['destination_reached'] == True]
        failed_trucks_low_battery = self.truck_results_df[self.truck_results_df['stopped_due_to_low_battery'] == True]
        failed_trucks_sim_end = self.truck_results_df[self.truck_results_df['stopped_due_to_simulation_end'] == True]

        num_successful_trucks = len(successful_trucks)
        num_failed_trucks_low_battery = len(failed_trucks_low_battery)
        num_failed_trucks_sim_end = len(failed_trucks_sim_end)
        
        actual_stopped_trucks = self.truck_results_df[self.truck_results_df['stopped_due_to_low_battery'] | self.truck_results_df['stopped_due_to_simulation_end'] | self.truck_results_df['destination_reached']]
        avg_final_soc = actual_stopped_trucks['final_SOC'].mean() if not actual_stopped_trucks.empty else 0

        total_traveled_distance_km = self.car_paths_df['CUMULATIVE_LINK_LENGTH'].sum()
        total_charged_energy_kwh = self.station_results_df['total_charged_energy_kWh'].sum() if self.station_results_df is not None else 0
        
        total_waiting_time_minutes = sum(sum(s.waiting_times) for s in self.stations) if self.stations else 0

        monthly_summary = {
            'Total Active Trucks Simulated': total_active_trucks,
            'Successful Trips (Destination Reached)': num_successful_trucks,
            'Failed Trips (Low Battery)': num_failed_trucks_low_battery,
            'Failed Trips (Simulation End)': num_failed_trucks_sim_end,
            'Average Final SOC of Stopped Trucks (%)': round(avg_final_soc, 2),
            'Total Traveled Distance (km) (all loaded data)': round(total_traveled_distance_km, 2),
            'Total Charged Energy (kWh)': round(total_charged_energy_kwh, 2),
            'Total Waiting Time at Stations (minutes)': round(total_waiting_time_minutes, 2),
            'Total Chargers Installed': self.station_df['num_of_charger'].sum(),
            'Average Charger Utilization (%)': self.station_results_df['utilization_percentage'].mean() if self.station_results_df is not None and not self.station_results_df[self.station_results_df['num_of_charger'] > 0].empty else 0
        }
        
        summary_df = pd.DataFrame([monthly_summary]).T.rename(columns={0: 'Value'})
        summary_df.index.name = 'Metric'
        summary_df.to_csv(os.path.join(output_path, "monthly_overall_summary.csv"), encoding='utf-8-sig')
        print("월별 전체 요약 통계 저장 완료: monthly_overall_summary.csv")

    def generate_overall_traffic_graphs(self, output_path):
        print("\n--- 월별 전체 트래픽 추이 그래프 생성 시작 ---")
        
        all_queue_histories = [s.queue_history for s in self.stations if s.queue_history]
        all_charging_histories = [s.charging_history for s in self.stations if s.charging_history]

        if not all_queue_histories and not all_charging_histories:
            print("충분한 이력 데이터가 없어 전체 트래픽 추이 그래프를 생성할 수 없습니다.")
            return

        max_len = 0
        if all_queue_histories:
            max_len = max(max_len, max(len(h) for h in all_queue_histories))
        if all_charging_histories:
            max_len = max(max_len, max(len(h) for h in all_charging_histories))
        
        total_queued_trucks_by_step = np.zeros(max_len)
        total_charging_trucks_by_step = np.zeros(max_len)

        for history in all_queue_histories:
            total_queued_trucks_by_step[:len(history)] += history
        for history in all_charging_histories:
            total_charging_trucks_by_step[:len(history)] += history

        time_steps = np.arange(len(total_queued_trucks_by_step)) * self.unit_minutes
        
        if len(time_steps) > 1:
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.plot(time_steps, total_queued_trucks_by_step, label='Total Queued Trucks', color='orange')
            ax.plot(time_steps, total_charging_trucks_by_step, label='Total Charging Trucks', color='blue')
            ax.set_xlabel('Simulation Time (minutes)'); ax.set_ylabel('Number of Trucks'); ax.set_title('Overall Monthly Traffic and Charging Activity')
            ax.legend(); ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "overall_monthly_traffic_activity.png"))
            plt.close(fig)
            print("전체 월별 트래픽 추이 그래프 저장 완료: overall_monthly_traffic_activity.png")
        else:
            print("충분한 이력 데이터가 없어 전체 트래픽 추이 그래프를 생성할 수 없습니다.")

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
    target_month = 1

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