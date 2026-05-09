import os
import gc
import time
import random
import warnings
from datetime import datetime
import multiprocessing as mp
import re

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from matplotlib import pyplot as plt, ticker
from matplotlib import patches as mpatches
import seaborn as sns
import pyarrow.parquet as pq

# 필요한 클래스 임포트
from charger import Charger
from station import Station
from truck_for_analysis import Truck

# --- 초기 설정 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- [추가] Matplotlib 한글 경고 방지를 위한 설정 ---
# 모든 시각화 결과에서 영문 폰트만 사용하도록 명시
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Verdana'] # 시스템에 있는 영문 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


class AnalyticsSimulator:
    """
    Simulation, results aggregation, and visualization class.
    """
    def __init__(self, car_paths_df, station_df, unit_minutes, simulating_hours, number_of_trucks, number_of_max_chargers, truck_step_frequency):
        self.car_paths_df = car_paths_df
        self.station_df = station_df
        self.number_of_max_chargers = number_of_max_chargers
        self.unit_minutes = unit_minutes
        self.simulating_hours = simulating_hours
        self.number_of_trucks = number_of_trucks
        self.truck_step_frequency = truck_step_frequency
        
        self.stations = []
        self.link_id_to_station = {}
        self.trucks = []
        self.current_time = 0
        
        self.truck_results_df = pd.DataFrame(columns=[
            'truck_id', 'final_SOC', 'destination_reached',
            'stopped_due_to_low_battery', 'stopped_due_to_simulation_end',
            'total_distance_planned', 'actual_traveled_distance_km',
            'traveled_distance_at_last_stop85', 'total_charged_energy_kwh',
            'total_charging_events'
        ])
        self.station_results_df = None
        self.failed_trucks_df = None
        self.completed_trucks_data = {}

    def prepare_simulation(self):
        """Initializes the simulation environment."""
        self.stations = self._load_stations_from_df(self.station_df)
        self.link_id_to_station = {station.link_id: station for station in self.stations}

        operational_station_link_ids = {s.link_id for s in self.stations if s.num_of_chargers > 0}
        self.car_paths_df['EVCS'] = np.where(self.car_paths_df['LINK_ID'].isin(operational_station_link_ids), 1, 0)

        for _, group in self.car_paths_df.groupby('OBU_ID'):
            truck = Truck(group, self.simulating_hours, self.link_id_to_station, self, 10)
            self.trucks.append(truck)
            if truck.status == 'inactive' and truck.next_activation_time == 0:
                    truck.record_state(0)

        self.current_time = 0
        gc.collect()

    def run_simulation(self):
        """Executes the simulation loop."""
        total_steps = self.simulating_hours * (60 // self.unit_minutes)
        for i in range(total_steps):
            for station in self.stations:
                station.update_chargers(self.current_time)
            for station in self.stations:
                station.process_queue(self.current_time)

            if i % self.truck_step_frequency == 0:
                for truck in list(self.trucks):
                    if truck in self.trucks and truck.status != 'stopped':
                        if self.current_time >= truck.next_activation_time:
                            truck.step(self.current_time)

            for truck in self.trucks:
                if truck.status not in ['inactive', 'stopped']:
                    truck.record_state(self.current_time)

            self.current_time += self.unit_minutes

        for station in self.stations:
            station.finalize_unprocessed_trucks(self.current_time)

        for truck_to_cleanup in list(self.trucks):
            if truck_to_cleanup.status != 'stopped':
                truck_to_cleanup.stop()

    def add_completed_truck_data(self, truck_id, history, path_df):
        """Stores detailed data for a completed truck trip."""
        if truck_id not in self.completed_trucks_data:
            self.completed_trucks_data[truck_id] = {'history': history, 'path_df': path_df}

    def remove_truck(self, truck):
        """Removes a truck from the simulation."""
        if truck in self.trucks:
            self.trucks.remove(truck)

    def get_results(self):
        """[수정] Calculates and returns simulation results, including granular lists."""
        total_simulation_minutes = self.simulating_hours * 60
        station_data = []
        
        for station in self.stations:
            total_charger_minutes_available = station.num_of_chargers * total_simulation_minutes
            total_charger_minutes_used = sum(c.total_charging_duration_minutes for c in station.chargers) # 이미 계산 중인 값
            utilization = (total_charger_minutes_used / total_charger_minutes_available) * 100 if total_charger_minutes_available > 0 else 0

            station_data.append({
                'station_id': station.station_id, 'link_id': station.link_id, 'num_of_charger': station.num_of_chargers,
                'total_charged_energy_kWh': sum(c.total_charged_energy for c in station.chargers),
                'total_charging_events': sum(c.charging_events_count for c in station.chargers),
                'total_charging_duration_minutes': total_charger_minutes_used,
                'avg_queue_length': np.mean(station.queue_history) if station.queue_history else 0,
                'max_queue_length': np.max(station.queue_history) if station.queue_history else 0,
                'avg_waiting_time_min': np.mean(station.waiting_times) if station.waiting_times else 0,
                'max_simultaneous_charging_vehicles': np.max(station.charging_history) if station.charging_history else 0,
                'max_power_kW': np.max(station.power_history) if station.power_history else 0,
                'utilization_percentage': utilization,
                'queue_history_raw': station.queue_history, 'charging_history_raw': station.charging_history,
                'power_history_raw': station.power_history,
                'cumulative_arrivals_history': station.cumulative_arrivals_history, 'cumulative_departures_history': station.cumulative_departures_history
            })

        self.station_results_df = pd.DataFrame(station_data)

        if self.truck_results_df is not None and not self.truck_results_df.empty:
            self.failed_trucks_df = self.truck_results_df[
                (self.truck_results_df['destination_reached'] == False) & (self.truck_results_df['stopped_due_to_low_battery'] == True)
            ].copy()
        else:
            self.failed_trucks_df = pd.DataFrame(columns=self.truck_results_df.columns if self.truck_results_df is not None else [])

        revenue_df = self._calculate_revenue()
        opex_df = self._calculate_opex()
        capex_df = self._calculate_capex()
        penalty_summary_df, station_penalty_df = self._calculate_penalty()

        financial_df = pd.merge(revenue_df, opex_df, on='station_id', how='outer')
        financial_df = pd.merge(financial_df, capex_df, on='station_id', how='outer')
        financial_df = pd.merge(financial_df, station_penalty_df, on='station_id', how='outer').fillna(0)

        for col in penalty_summary_df.columns:
            if not penalty_summary_df[col].empty:
                financial_df[col] = penalty_summary_df[col].iloc[0]

        if 'station_id' in financial_df.columns:
            financial_df['station_id'] = financial_df['station_id'].astype(int)

        # --- [신규] 모든 개별 데이터 리스트 수집 ---
        
        # 1. 개별 대기 시간 (기존)
        all_individual_waiting_times = [wait for station in self.stations for wait in station.waiting_times]

        # 2. 충전기별 가동률 (per-charger-day) - (charger.py 수정 불필요)
        all_individual_charger_utilizations = [
            (c.total_charging_duration_minutes / total_simulation_minutes) * 100 
            for s in self.stations if s.num_of_chargers > 0 
            for c in s.chargers
        ]

        # 3. 이벤트별 충전 시간 (per-event) - (charger.py 수정 필요)
        all_individual_charging_durations = [
            dur for s in self.stations 
            for c in s.chargers 
            for dur in c.individual_durations
        ]

        # 4. 이벤트별 충전 에너지 (per-event) - (charger.py 수정 필요)
        all_individual_charging_energies = [
            eng for s in self.stations 
            for c in s.chargers 
            for eng in c.individual_energies
        ]
        # --- [신규 완료] ---

        # [수정] 4개의 리스트를 반환합니다.
        return (
            financial_df, 
            self.station_results_df.copy(), 
            all_individual_waiting_times,
            all_individual_charger_utilizations,  # [신규]
            all_individual_charging_durations,    # [신규]
            all_individual_charging_energies      # [신규]
        )

    def _load_stations_from_df(self, df):
        return [Station(station_id=idx, link_id=int(row['link_id']), num_of_chargers=int(row['num_of_charger']),
                        charger_specs=[{'power': 200, 'rate': 560}] * int(row['num_of_charger']),
                        unit_minutes=self.unit_minutes)
                for idx, row in df.iterrows()]

    def _calculate_revenue(self):
        return pd.DataFrame([{'station_id': s.station_id, 'revenue': sum(c.rate * c.total_charged_energy for c in s.chargers)} for s in self.stations])

    def _calculate_opex(self):
        base_rate_per_kw=2580/30; energy_rate_per_kwh=101.7+9+5; vat_multiplier=1.132
        labor_cost_per_charger=6250; maint_cost_per_charger=800
        results = []
        for s in self.stations:
            total_power=sum(c.power for c in s.chargers); total_energy=sum(c.total_charged_energy for c in s.chargers)
            energy_price=((total_power*base_rate_per_kw)+(total_energy*energy_rate_per_kwh))*vat_multiplier
            labor_cost=s.num_of_chargers*labor_cost_per_charger; maint_cost=s.num_of_chargers*maint_cost_per_charger
            results.append({'station_id':s.station_id,'opex':labor_cost+maint_cost+energy_price})
        return pd.DataFrame(results)

    def _calculate_capex(self):
        return pd.DataFrame([{'station_id':s.station_id,'capex':(96000000*s.num_of_chargers)/(5*365)} for s in self.stations])

    def _calculate_penalty(self):
        failed_truck_penalty = 0.0
        if self.failed_trucks_df is not None and not self.failed_trucks_df.empty:
            planned = self.failed_trucks_df['total_distance_planned']
            traveled = self.failed_trucks_df['traveled_distance_at_last_stop85'].fillna(0)
            dist_penalty = np.where(traveled <= 0, planned / 2, np.maximum(0, planned - traveled) / 2)
            choice = np.random.choice([True, False], size=len(self.failed_trucks_df))
            penalty_vals = np.where(choice, 136395.9 + 3221.87 * dist_penalty - 2.72 * dist_penalty**2,
                                    121628.18 + 2765.50 * dist_penalty - 2.00 * dist_penalty**2)
            failed_truck_penalty = np.maximum(0, penalty_vals).sum()

        charger_penalty = 0.0
        total_chargers = sum(s.num_of_chargers for s in self.stations)
        if total_chargers > self.number_of_max_chargers:
            charger_penalty = float(80000000 * (total_chargers - self.number_of_max_chargers))

        MINUTE_PENALTY_RATE = (11000000 / (10.9 * 22.4)) / 60.0
        station_waiting_penalties = {}
        for station in self.stations:
            station_penalty = sum(self._calculate_wait_penalty(wait_time, MINUTE_PENALTY_RATE) for wait_time in station.waiting_times)
            station_waiting_penalties[station.station_id] = station_penalty

        total_waiting_penalty = sum(station_waiting_penalties.values())
        total_penalty = failed_truck_penalty + charger_penalty + total_waiting_penalty

        summary_df = pd.DataFrame([{'failed_truck_penalty':failed_truck_penalty,'charger_penalty':charger_penalty,
                                    'waiting_penalty':total_waiting_penalty,'total_penalty':total_penalty}])
        station_penalty_df = pd.DataFrame(list(station_waiting_penalties.items()), columns=['station_id', 'station_waiting_penalty'])
        return summary_df, station_penalty_df
        
    def _calculate_wait_penalty(self, wait_time, rate):
        p = 0.0
        if wait_time > 60: p += (wait_time - 60) * rate * 8; wait_time = 60
        if wait_time > 40: p += (wait_time - 40) * rate * 4; wait_time = 40
        if wait_time > 20: p += (wait_time - 20) * rate * 2; wait_time = 20
        if wait_time > 0: p += wait_time * rate * 1
        return p


# --- Global Helper Functions ---
import pandas as pd

def process_soc_history_for_visualization(history_df):
    if history_df.empty:
        return pd.DataFrame()

    # 1. 이벤트 블록 생성
    history_df['event_block'] = (history_df['status'].ne(history_df['status'].shift()) |
                                 history_df['soc'].ne(history_df['soc'].shift())).cumsum()

    boundary_df = history_df.groupby('event_block').agg(['first', 'last'])
    firsts = boundary_df.loc[:, (slice(None), 'first')]
    firsts.columns = firsts.columns.droplevel(1)
    lasts = boundary_df.loc[:, (slice(None), 'last')]
    lasts.columns = lasts.columns.droplevel(1)
    temp_df = pd.concat([firsts, lasts])
    temp_df['time'] = pd.to_numeric(temp_df['time'])
    processed_events = temp_df.sort_values(by='time').drop_duplicates().reset_index()

    # 2. 첫 'driving' 상태 처리
    if not processed_events.empty and processed_events.loc[0, 'status'] == 'driving':
        first_block_id = processed_events.loc[0, 'event_block']
        processed_events.loc[processed_events['event_block'] == first_block_id, 'status'] = 'inactive'

    # 3. 연속된 'driving' 그룹 병합
    group_ids = (processed_events['status'] != processed_events['status'].shift()).cumsum()
    final_points_after_drive_merge = []
    for _, group in processed_events.groupby(group_ids):
        if group['status'].iloc[0] == 'driving':
            final_points_after_drive_merge.append(group.iloc[0].to_dict())
            if len(group) > 1:
                final_points_after_drive_merge.append(group.iloc[-1].to_dict())
        else:
            final_points_after_drive_merge.extend(group.to_dict('records'))
    processed_df = pd.DataFrame(final_points_after_drive_merge)

    # 4, 5, 6 규칙 적용
    final_path = []
    for i in range(len(processed_df)):
        current_event = processed_df.iloc[i].to_dict()
        
        is_last_of_charging_block = False
        if 'charging' in current_event['status']:
            if i + 1 == len(processed_df) or 'charging' not in processed_df.iloc[i+1]['status']:
                is_last_of_charging_block = True
                current_event['soc'] = 100.0

        if current_event['status'] == 'driving':
            is_first_of_driving_block = (i == 0 or processed_df.iloc[i-1]['status'] != 'driving')
            is_last_of_driving_block = (i + 1 == len(processed_df) or processed_df.iloc[i+1]['status'] != 'driving')
            if is_first_of_driving_block and i > 0:
                prev_event = final_path[-1]
                current_event['soc'] = 100.0 if 'charging' in prev_event['status'] else prev_event['soc']
            if is_last_of_driving_block and i + 1 < len(processed_df):
                next_event = processed_df.iloc[i+1].to_dict()
                current_event['soc'] = next_event['soc']

        final_path.append(current_event)

        # --- [핵심 수정] driving 전환 추가 규칙 확장 ---
        if i + 1 < len(processed_df):
            next_event_df = processed_df.iloc[i+1]
            
            # 조건 1: 충전 종료 후 정차/대기
            cond1 = is_last_of_charging_block and next_event_df['status'] in ['waiting_for_charge', 'stopping']
            
            # 조건 2: 정차 후 충전 대기 (새로 추가된 조건)
            cond2 = (current_event['status'] == 'stopping') and (next_event_df['status'] == 'waiting_for_charge')
            
            # 두 조건 중 하나라도 만족하면 driving 행 추가
            if cond1 or cond2:
                driving_transition_1 = current_event.copy()
                driving_transition_1['status'] = 'driving'
                final_path.append(driving_transition_1)
                
                driving_transition_2 = next_event_df.to_dict()
                driving_transition_2['status'] = 'driving'
                final_path.append(driving_transition_2)
    
    final_df = pd.DataFrame(final_path)
    if 'event_block' in final_df.columns:
        final_df = final_df.drop(columns=['event_block'])

    return final_df.drop_duplicates().reset_index(drop=True)

def generate_truck_visualizations(output_folder, truck_results_df, completed_trucks_data, link_geometries_gdf, num_trucks_to_visualize=20):
    """
    [수정] 시뮬레이션 결과 데이터를 받아 트럭 시각화를 생성합니다.
    SOC 그래프는 가공된 데이터를 사용하고, 지도 시각화는 원본 데이터를 사용합니다.
    """
    print(f"  - Generating truck visualizations...")
    
    if truck_results_df is None or truck_results_df.empty:
        print("  - Truck results data is not available for visualization.")
        return
    if not completed_trucks_data:
        print("  - Detailed truck data for visualization is not available.")
        return

    successful_trucks_df = truck_results_df[truck_results_df['destination_reached'] == True].copy()
    if successful_trucks_df.empty:
        print("  - No trucks successfully reached their destination. Skipping visualization.")
        return
        
    sorted_trucks_df = successful_trucks_df.sort_values(by='total_distance_planned', ascending=False)
    truck_ids_to_visualize = sorted_trucks_df['truck_id'].head(num_trucks_to_visualize).tolist()
    
    print(f"  - Selected for visualization: Top {len(truck_ids_to_visualize)} trucks by travel distance.")
    os.makedirs(output_folder, exist_ok=True)
    
    for truck_id in truck_ids_to_visualize:
        if truck_id in completed_trucks_data:
            data = completed_trucks_data[truck_id]
            # 원본 history 데이터를 DataFrame으로 변환
            history_df = pd.DataFrame(data['history'])
            path_df = data['path_df']
            
            # SOC 시각화를 위해 데이터를 가공하는 함수 호출
            processed_history_df = process_soc_history_for_visualization(history_df.copy())
            
            # --- [핵심 수정] 트럭의 SOC 임계값 조회 ---
            truck_info = truck_results_df.loc[truck_results_df['truck_id'] == truck_id]
            soc_threshold = None # 기본값 설정
            if not truck_info.empty and 'threshold_SOC' in truck_info.columns:
                # 'thredshold_SOC' 컬럼에서 값을 가져옴
                soc_threshold = truck_info['threshold_SOC'].iloc[0]

            # --- [핵심 수정] 수정된 함수에 soc_threshold 인자 전달 ---
            plot_truck_soc_history(processed_history_df, truck_id, output_folder, soc_threshold)
            
            if link_geometries_gdf is not None and path_df is not None:
                # 지도 시각화는 정확한 경로 추적을 위해 원본 데이터(history_df) 사용
                plot_truck_path_map(history_df, path_df, truck_id, output_folder, link_geometries_gdf)
        else:
            print(f"  - Warning: Detailed data for truck {truck_id} not found. Skipping.")
    print(f"  - ✅ Truck visualizations generated successfully.")


def plot_truck_soc_history(history_df, truck_id, output_folder, soc_threshold):
    """
    [수정] 트럭의 SOC 이력 그래프를 생성합니다. 
    비율을 5:3으로 조정하고, 제목과 범례를 제거하며, SOC 임계값 점선을 추가합니다.
    X축은 시간 단위로 표시됩니다.
    """
    if history_df.empty: return
    plt.rcParams['font.family'] = 'Times New Roman'

    # --- [수정] figsize를 (15, 9)로 변경하여 5:3 비율로 조정 ---
    fig, ax = plt.subplots(figsize=(15, 9))

    # --- [수정] X축 단위를 분에서 시간으로 변경 ---
    history_df['time_hours'] = history_df['time'] / 60.0

    ax.plot(history_df['time_hours'], history_df['soc'], color='black', zorder=10)

    # --- [추가] SOC 임계값을 붉은색 가로 점선으로 표시 ---
    if pd.notna(soc_threshold):
        ax.axhline(y=soc_threshold, color='red', linestyle='--', linewidth=5, zorder=11)

    status_colors = {
        'driving': 'skyblue',
        'stopping': (1, 1, 0, 0.7),
        'waiting_for_charge': (0.8, 0.6, 0.8, 0.6),
        'opportunity_charging': 'hotpink',
        'enroute_charging': 'orange',
        'inactive': 'lightgrey',
    }
    
    # 시간대별 상태를 배경색으로 표시
    for i in range(len(history_df) - 1):
        ax.axvspan(history_df['time_hours'].iloc[i], history_df['time_hours'].iloc[i+1], 
                     facecolor=status_colors.get(history_df['status'].iloc[i], 'white'),
                     edgecolor='none',
                     alpha=0.7)

    # --- [수정] 제목 제거, X축 라벨 변경, Y축 라벨 설정 ---
    ax.set_xlabel('Time (hours)', fontsize=40)
    ax.set_ylabel('State of Charge (%)', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=36)
    
    ax.set_ylim(0, 105)
    
    # --- [수정] X축 범위도 시간 단위로 설정 ---
    ax.set_xlim(history_df['time_hours'].min(), history_df['time_hours'].max())
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- [수정] 범례 제거 ---
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, f'soc_history_truck_{truck_id}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_truck_path_map(history_df, path_df, truck_id, output_folder, link_geometries_gdf):
    """
    Creates a map of the truck's path, prioritizing opportunity charging color.
    """
    if history_df.empty or path_df.empty or link_geometries_gdf is None: return
    plt.rcParams['font.family'] = 'Times New Roman'

    m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles='CartoDB positron', control_scale=True)

    opp_charge_links = history_df[history_df['status'] == 'opportunity_charging']['link_id'].unique()
    
    is_stopping_at_charge_location = (history_df['link_id'].isin(opp_charge_links)) & (history_df['status'] == 'stopping')
    history_df.loc[is_stopping_at_charge_location, 'status'] = 'opportunity_charging'

    status_color_map = {
        'driving': 'blue',
        'stopping': 'yellow',
        'opportunity_charging': 'pink',
        'enroute_charging': 'orange',
    }

    full_path_links_df = path_df[['LINK_ID']].merge(link_geometries_gdf, left_on='LINK_ID', right_on='link_id', how='inner')
    if not full_path_links_df.empty:
        full_path_links = gpd.GeoDataFrame(full_path_links_df, geometry='geometry')
        full_path_links = full_path_links[full_path_links.geometry.notna() & ~full_path_links.geometry.is_empty]
        if not full_path_links.empty:
            folium.GeoJson(
                full_path_links,
                style_function=lambda x: {'color': 'blue', 'weight': 4, 'opacity': 0.8},
                name='Full Path'
            ).add_to(m)

    history_df['status_block'] = (history_df['status'] != history_df['status'].shift()).cumsum()
    
    for block_id, segment_df in history_df.groupby('status_block'):
        status = segment_df['status'].iloc[0]
        link_ids = segment_df['link_id'].unique().tolist()
        
        if status not in status_color_map or status == 'driving':
            continue
            
        color = status_color_map.get(status)
        segment_geom_df = link_geometries_gdf[link_geometries_gdf['link_id'].isin(link_ids)]
        
        if not segment_geom_df.empty:
            folium.GeoJson(
                segment_geom_df,
                style_function=lambda x, c=color: {'color': c, 'weight': 5, 'opacity': 0.9}
            ).add_to(m)

    start_link_id = path_df['LINK_ID'].iloc[0]
    end_link_id = path_df['LINK_ID'].iloc[-1]
    
    start_geom = link_geometries_gdf[link_geometries_gdf['link_id'] == start_link_id]
    if not start_geom.empty:
        folium.GeoJson(start_geom, style_function=lambda x: {'color': 'red', 'weight': 6, 'opacity': 1.0}).add_to(m)

    end_geom = link_geometries_gdf[link_geometries_gdf['link_id'] == end_link_id]
    if not end_geom.empty:
        folium.GeoJson(end_geom, style_function=lambda x: {'color': 'red', 'weight': 6, 'opacity': 1.0}).add_to(m)

    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 270px; height: auto; 
                border: 2px solid grey; z-index:9999; font-size: 18px;
                background-color: rgba(255, 255, 255, 0.9);
                padding: 12px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
    <b style="font-size: 20px; display: block; margin-bottom: 5px;">Path Status Legend</b>
    <i class="fa fa-minus" style="color:red"></i>&nbsp; Start/End Link<br>
    <i class="fa fa-minus" style="color:blue"></i>&nbsp; Driving<br>
    <i class="fa fa-minus" style="color:yellow"></i>&nbsp; Stopping<br>
    <i class="fa fa-minus" style="color:pink"></i>&nbsp; Opportunistic Charging<br>
    <i class="fa fa-minus" style="color:orange"></i>&nbsp; En-route Charging
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl().add_to(m)
    m.save(os.path.join(output_folder, f'path_map_truck_{truck_id}.html'))

def get_simulation_dates(base_folder, num_weeks):
    date_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    valid_dates = [d for d in date_folders if pd.to_datetime(d, errors='coerce') is not pd.NaT]
    if not valid_dates: raise FileNotFoundError(f"'{base_folder}' not found.")
    df = pd.DataFrame({'date_str': valid_dates})
    df['date_obj'] = pd.to_datetime(df['date_str'])
    df['week_num'] = df['date_obj'].dt.isocalendar().week
    weeks_with_enough_data = {w: days.tolist() for w, days in df.groupby('week_num')['date_str'] if len(days) >= 5}
    if not weeks_with_enough_data: raise ValueError("No week found with >= 5 days of data.")
    available_weeks = list(weeks_with_enough_data.keys())
    num_weeks = min(num_weeks, len(available_weeks))
    selected_week_nums = random.sample(available_weeks, num_weeks)
    weeks_to_simulate = {week: sorted(weeks_with_enough_data[week]) for week in sorted(selected_week_nums)}
    total_days = sum(len(days) for days in weeks_to_simulate.values())
    print(f"--- Date selection complete: Simulating {len(weeks_to_simulate)} weeks, total {total_days} days. ---")
    return weeks_to_simulate

def load_car_path_df_for_day(day_folder_path, number_of_trucks):
    files = [f for f in os.listdir(day_folder_path) if f.endswith(".parquet")]
    if not files: raise FileNotFoundError(f"No Parquet files found in '{day_folder_path}'.")
    all_obu_ids = set(pd.concat([pq.read_table(os.path.join(day_folder_path, f), columns=['OBU_ID']).to_pandas() for f in files])['OBU_ID'].unique())
    selected_obu_ids = set(random.sample(list(all_obu_ids), number_of_trucks)) if len(all_obu_ids) >= number_of_trucks else all_obu_ids
    car_paths_df = pd.concat([pd.read_parquet(os.path.join(day_folder_path, f), filters=[('OBU_ID', 'in', list(selected_obu_ids))]) for f in files], ignore_index=True).dropna(subset=['DATETIME'])
    car_paths_df['DATETIME'] = pd.to_datetime(car_paths_df['DATETIME'], format='%H:%M', errors='coerce').dt.time
    car_paths_df['START_TIME_MINUTES'] = car_paths_df.groupby('OBU_ID')['DATETIME'].transform('first').apply(lambda x: x.hour * 60 + x.minute if pd.notnull(x) else np.nan)
    car_paths_df.dropna(subset=['START_TIME_MINUTES'], inplace=True)
    return car_paths_df

def load_station_df(station_file_path):
    df = pd.read_csv(station_file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    df[['link_id', 'num_of_charger']] = df[['link_id', 'num_of_charger']].astype(int)
    return df

def load_candidate_df(candidate_file_path):
    df = pd.read_csv(candidate_file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def load_link_geometries(shapefile_path):
    """
    Loads shapefile geometry and normalizes the link id column to `link_id`.
    Accepts typical link-id fields such as `link_id`, `LINK_ID`, or `id`.
    """
    gdf = gpd.read_file(shapefile_path)

    candidate_names = {'link_id', 'id'}
    link_id_col = next((col for col in gdf.columns if col.lower() in candidate_names), None)
    if link_id_col is None:
        non_geometry_cols = [col for col in gdf.columns if col.lower() != 'geometry']
        if len(non_geometry_cols) == 1:
            link_id_col = non_geometry_cols[0]

    if link_id_col is None:
        raise ValueError("Shapefile must contain a usable link-id column such as 'link_id', 'LINK_ID', or 'id'.")

    if link_id_col != 'link_id':
        gdf = gdf.rename(columns={link_id_col: 'link_id'})

    if 'geometry' not in gdf.columns:
        raise ValueError("Shapefile must contain a geometry column.")

    gdf['link_id'] = pd.to_numeric(gdf['link_id'], errors='coerce')
    gdf = gdf.dropna(subset=['link_id']).copy()
    gdf['link_id'] = gdf['link_id'].astype(int)

    return gdf[['link_id', 'geometry']].copy()

def generate_and_save_chargers_vs_wait_time_scatter(operational_df, output_path, title_prefix):
    os.makedirs(output_path, exist_ok=True)
    plot_data = operational_df[operational_df['num_of_charger'] > 0].copy()
    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(plot_data['num_of_charger'], plot_data['avg_waiting_time_min'], alpha=0.6, s=50)
        ax.set_title(f'{title_prefix}: Relationship between Charger Capacity and Average Waiting Time', fontsize=14)
        ax.set_xlabel('Number of Chargers', fontsize=12)
        ax.set_ylabel('Average Waiting Time (min)', fontsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.grid(True, linestyle='--'); plt.tight_layout()
        plt.savefig(os.path.join(output_path, "chargers_vs_wait_time_average.png"), dpi=150)
        plt.close(fig)

def generate_and_save_financial_metrics_boxplot(financial_df, output_path, title_prefix):
    os.makedirs(output_path, exist_ok=True)
    if not financial_df.empty:
        fig_box, ax_box = plt.subplots(figsize=(12, 8))
        metrics = ['revenue', 'opex', 'capex', 'station_waiting_penalty']
        sns.boxplot(data=financial_df[metrics], ax=ax_box, palette="Set2", width=0.4)
        ax_box.set_title(f'{title_prefix}: Distribution of Daily Financial Metrics per Station', fontsize=14)
        ax_box.set_ylabel('Value (KRW)', fontsize=12)
        ax_box.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "financial_metrics_boxplot_per_station.png"))
        plt.close(fig_box)

# --- [MODIFIED FUNCTION] ---
def generate_and_save_enhanced_summary_report(financial_df, operational_df, truck_results_df, output_path, title_prefix):
    """
    [MODIFIED] 재무, 인프라, 운영 KPI, 운행 성공률 등 상세 지표를 포함한 종합 요약 보고서를 생성하고,
    벌금(penalty) 항목을 세분화하여 보여줍니다.
    """
    print("  - Generating enhanced summary report...")
    os.makedirs(output_path, exist_ok=True)
    
    # --- 1. 재무 요약 ---
    total_revenue = financial_df['revenue'].sum()
    total_opex = financial_df['opex'].sum()
    total_capex = financial_df['capex'].sum()
    
    # Penalty breakdown (extracting from the first row as it's broadcasted)
    failed_truck_penalty = financial_df['failed_truck_penalty'].iloc[0] if 'failed_truck_penalty' in financial_df.columns and not financial_df.empty else 0
    charger_penalty = financial_df['charger_penalty'].iloc[0] if 'charger_penalty' in financial_df.columns and not financial_df.empty else 0
    waiting_penalty = financial_df['waiting_penalty'].iloc[0] if 'waiting_penalty' in financial_df.columns and not financial_df.empty else 0
    total_penalty = financial_df['total_penalty'].iloc[0] if 'total_penalty' in financial_df.columns and not financial_df.empty else (failed_truck_penalty + charger_penalty + waiting_penalty)
    
    of_value = total_revenue - total_opex - total_capex - total_penalty

    # --- 2. 인프라 요약 ---
    op_stations_df = operational_df[operational_df['num_of_charger'] > 0]
    num_operating_stations = len(op_stations_df)
    total_chargers = op_stations_df['num_of_charger'].sum()
    charger_dist = op_stations_df['num_of_charger'].describe()
    
    # --- 3. 운영 KPI 요약 ---
    avg_wait_time_all = operational_df['avg_waiting_time_min'].mean()
    avg_utilization_op = op_stations_df['utilization_percentage'].mean() if not op_stations_df.empty else 0
    total_charging_events = operational_df['total_charging_events'].sum()

    # --- 4. 운행 성공률 계산 ---
    total_trucks = len(truck_results_df)
    failed_trucks = len(truck_results_df[truck_results_df['stopped_due_to_low_battery'] == True])
    success_rate = ((total_trucks - failed_trucks) / total_trucks) * 100 if total_trucks > 0 else 0

    # --- 5. 보고서 텍스트 생성 ---
    summary_lines = [
        f"--- Enhanced Summary Report: {title_prefix} ---\n",
        "=== Financial Summary ===",
        f"Objective Function (OF) Value: {of_value:,.0f} KRW",
        f"  - Total Revenue: {total_revenue:,.0f} KRW",
        f"  - Total CAPEX: {total_capex:,.0f} KRW",
        f"  - Total OPEX: {total_opex:,.0f} KRW",
        f"  - Total Penalty: {total_penalty:,.0f} KRW",
        f"    * Trip Failure Penalty: {failed_truck_penalty:,.0f} KRW",
        f"    * Charger Installation Penalty: {charger_penalty:,.0f} KRW",
        f"    * Waiting Time Penalty: {waiting_penalty:,.0f} KRW\n",
        
        "=== Infrastructure Summary ===",
        f"Number of Operating Stations: {num_operating_stations}",
        f"Total Number of Chargers: {int(total_chargers)}",
        "Charger Distribution per Station:",
        f"  - Average: {charger_dist.get('mean', 0):.2f}",
        f"  - Min: {int(charger_dist.get('min', 0))}",
        f"  - 25th Pctl: {int(charger_dist.get('25%', 0))}",
        f"  - Median: {int(charger_dist.get('50%', 0))}",
        f"  - 75th Pctl: {int(charger_dist.get('75%', 0))}",
        f"  - Max: {int(charger_dist.get('max', 0))}\n",

        "=== Operational KPIs ===",
        f"Average Waiting Time (All Stations): {avg_wait_time_all:.2f} min",
        f"Average Utilization (Operating Stations): {avg_utilization_op:.2f}%",
        f"Total Charging Events (Station Usage): {total_charging_events}\n",
        
        "=== Trip Completion Summary ===",
        f"Total Truck Trips Simulated: {total_trucks}",
        f"Trips Failed due to Low Battery: {failed_trucks}",
        f"Trip Success Rate: {success_rate:.2f}%\n"
    ]
    
    # --- 6. 파일 저장 ---
    report_path = os.path.join(output_path, "enhanced_summary_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
        
    # 기존 상세 데이터도 저장
    financial_df.to_csv(os.path.join(output_path, "financial_summary_by_station.csv"), index=False, encoding='utf-8-sig')
    operational_df.to_csv(os.path.join(output_path, "station_operational_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"  - ✅ Enhanced summary report saved to {report_path}")

# --- [MODIFIED FUNCTION] ---
def generate_and_save_timeseries_graphs(operational_df, output_path, unit_minutes):
    """
    [MODIFIED] 스테이션별 시간대별 변화 그래프를 생성합니다. X축 단위를 시간으로 변경했습니다.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    
    os.makedirs(output_path, exist_ok=True)
    stations_with_activity = operational_df[operational_df['total_charging_events'] > 0]
    for _, row in stations_with_activity.iterrows():
        station_id=int(row['station_id']); num_chargers=int(row['num_of_charger'])
        if 'queue_history_raw' not in row or not isinstance(row['queue_history_raw'], list) or not row['queue_history_raw']:
            continue

        # 분 단위를 시간 단위로 변경
        time_steps_hours = (np.arange(len(row['queue_history_raw'])) * unit_minutes) / 60.0
        bar_width_hours = unit_minutes / 60.0

        # --- Occupancy and Queue Plot ---
        fig_occ, ax_occ = plt.subplots(figsize=(16, 9))
        ax_occ.bar(time_steps_hours, row['charging_history_raw'], width=bar_width_hours, label='Charging Trucks', color='skyblue')
        ax_occ.plot(time_steps_hours, row['queue_history_raw'], marker='o', markersize=4, linestyle='-', label='Queue Length', color='darkorange')
        ax_occ.axhline(y=num_chargers, color='red', linestyle='--', label=f'Capacity ({num_chargers})')
        ax_occ.legend(fontsize=24)
        ax_occ.set_xlabel('Time (hours)', fontsize=30) # X축 라벨 수정
        ax_occ.set_ylabel('Number of Trucks', fontsize=30)
        ax_occ.tick_params(axis='both', which='major', labelsize=26)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"station_{station_id}_occupancy_queue.png"), dpi=100)
        plt.close(fig_occ)

        # --- Cumulative Flow Plot ---
        if 'cumulative_arrivals_history' not in row or not isinstance(row['cumulative_arrivals_history'], list):
            continue
        
        # 분 단위를 시간 단위로 변경
        time_steps_cum_hours = (np.arange(len(row['cumulative_arrivals_history'])) * unit_minutes) / 60.0
        fig_cum, ax_cum = plt.subplots(figsize=(16, 9))
        
        # Modified colors and styles
        ax_cum.plot(time_steps_cum_hours, row['cumulative_arrivals_history'], label='Cumulative Arrivals', drawstyle='steps-post', color='silver', linewidth=2.5)
        ax_cum.plot(time_steps_cum_hours, row['cumulative_departures_history'], label='Cumulative Departures', drawstyle='steps-post', color='dimgray', linestyle='--', linewidth=2.5)
        
        ax_cum.fill_between(time_steps_cum_hours, row['cumulative_arrivals_history'], row['cumulative_departures_history'], step='post', alpha=0.2, color='grey', label='Trucks in System (Queue + Charging)')
        ax_cum.legend(fontsize=26)
        ax_cum.set_xlabel('Time (hours)', fontsize=30) # X축 라벨 수정
        ax_cum.set_ylabel('Cumulative Truck Count', fontsize=30)
        ax_cum.tick_params(axis='both', which='major', labelsize=26)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"station_{station_id}_cumulative_flow.png"), dpi=100)
        plt.close(fig_cum)

def _find_best_break_points(data, metrics, num_breaks=1):
    all_values = data[metrics].values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    sorted_values = pd.Series(np.unique(all_values)).sort_values()
    if len(sorted_values) < 4: return None
    gaps = sorted_values.diff().dropna()
    if gaps.empty: return None
    total_range = sorted_values.iloc[-1] - sorted_values.iloc[0]
    if total_range == 0: return None
    largest_gaps = gaps.nlargest(num_breaks)
    break_points = []
    significant_gaps_indices = largest_gaps[largest_gaps / total_range > 0.15].index
    if not significant_gaps_indices.any(): return None
    for gap_index in sorted(significant_gaps_indices):
        bottom_val = sorted_values.loc[gap_index - 1]
        top_val = sorted_values.loc[gap_index]
        gap_size = top_val - bottom_val
        break_points.append({"bottom": bottom_val + gap_size * 0.1, "top": top_val - gap_size * 0.1})
    return break_points if break_points else None

def generate_broken_axis_boxplot(data, metrics, output_path, filename, title):
    """
    [수정] Y축이 끊어진 Boxplot을 생성합니다.
    그래프 비율을 3:4로 조정하고, 전반적인 글씨 크기를 키웁니다.
    """
    if data.empty: return
    
    # 데이터에 따라 최적의 분리 지점 찾기
    break_points = _find_best_break_points(data, metrics, num_breaks=2)
    num_breaks = len(break_points) if break_points else 0
    if num_breaks < 2:
        break_points = _find_best_break_points(data, metrics, num_breaks=1)
        num_breaks = len(break_points) if break_points else 0
        
    y_max = data[metrics].max().max()
    y_min = data[metrics].min().min()
    
    # 축 분리 기호 설정
    d = .8
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    # --- [수정] 모든 케이스에 대해 figsize=(9, 12) 및 글씨 크기 상향 조정 ---
    
    if num_breaks == 0:
        fig, ax = plt.subplots(figsize=(16, 9)) # 3:4 비율
        sns.boxplot(data=data[metrics], ax=ax, palette="pastel", width=0.5)
        ax.set_title(title, fontsize=28, pad=15)
        ax.set_ylabel("Value (KRW)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=16)

    elif num_breaks == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 2]})
        fig.subplots_adjust(hspace=0.1)
        sns.boxplot(data=data[metrics], ax=ax1, palette="pastel", width=0.5)
        sns.boxplot(data=data[metrics], ax=ax2, palette="pastel", width=0.5)
        
        bp = break_points[0]
        ax1.set_ylim(bp['top'], y_max * 1.02)
        ax2.set_ylim(y_min * 0.98, bp['bottom'])
        
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False, axis='y', which='major', labelsize=16)
        ax2.xaxis.tick_bottom()
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        
        fig.suptitle(title, fontsize=28)
        fig.supylabel("Value (KRW)", fontsize=24) # 중앙 Y축 라벨

    elif num_breaks == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 2]})
        fig.subplots_adjust(hspace=0.08)
        sns.boxplot(data=data[metrics], ax=ax1, palette="pastel", width=0.5)
        sns.boxplot(data=data[metrics], ax=ax2, palette="pastel", width=0.5)
        sns.boxplot(data=data[metrics], ax=ax3, palette="pastel", width=0.5)
        
        bp1, bp2 = sorted(break_points, key=lambda x: x['bottom'])
        ax1.set_ylim(bp2['top'], y_max * 1.02)
        ax2.set_ylim(bp1['top'], bp2['bottom'])
        ax3.set_ylim(y_min * 0.98, bp1['bottom'])
        
        ax1.spines['bottom'].set_visible(False); ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False); ax3.spines['top'].set_visible(False)
        
        ax1.xaxis.tick_top(); ax1.tick_params(labeltop=False, axis='y', which='major', labelsize=16)
        ax2.xaxis.tick_top(); ax2.tick_params(labeltop=False, axis='y', which='major', labelsize=16)
        ax3.xaxis.tick_bottom(); ax3.tick_params(axis='both', which='major', labelsize=16)
        
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
        ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
        ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
        
        fig.suptitle(title, fontsize=28)
        fig.supylabel("Value (KRW)", fontsize=24) # 중앙 Y축 라벨

    # X축 라벨은 맨 아래 그래프에만 추가
    plt.xlabel("Scenarios", fontsize=24)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]) # 여백 조정
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close(fig)

def generate_and_save_aggregated_boxplots(list_of_financial_dfs, list_of_operational_dfs, list_of_summaries, output_path, title_prefix):
    os.makedirs(output_path, exist_ok=True)
    summary_df = pd.DataFrame(list_of_summaries)
    if not summary_df.empty:
        generate_broken_axis_boxplot(data=summary_df, metrics=['revenue', 'opex', 'capex', 'waiting_penalty'], output_path=output_path, filename="boxplot_daily_totals_broken.png", title=f'{title_prefix}: Distribution of Daily Total Metrics')
    all_station_financial_df = pd.concat(list_of_financial_dfs, ignore_index=True)
    if not all_station_financial_df.empty:
        generate_broken_axis_boxplot(data=all_station_financial_df, metrics=['revenue', 'opex', 'capex', 'station_waiting_penalty'], output_path=output_path, filename="boxplot_all_station_data_broken.png", title=f'{title_prefix}: Distribution of Per-Station Metrics (All Days)')
    
    all_station_operational_df = pd.concat(list_of_operational_dfs, ignore_index=True)
    utilization_data = all_station_operational_df[all_station_operational_df['num_of_charger'] > 0].copy()
    
    if not utilization_data.empty:
        fig, ax = plt.subplots(figsize=(14, 9))
        
        sns.boxplot(data=utilization_data, x='num_of_charger', y='utilization_percentage', ax=ax, palette="viridis")
        
        sns.regplot(
            x='num_of_charger', 
            y='utilization_percentage', 
            data=utilization_data, 
            ax=ax, 
            scatter=False,
            line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2.5, 'label': 'Trend Line'},
            ci=None
        )
        
        overall_avg = utilization_data['utilization_percentage'].mean()
        ax.axhline(y=overall_avg, color='darkorange', linestyle=':', linewidth=2.5, label=f'Overall Avg: {overall_avg:.2f}%')
        
        ax.legend()
        ax.set_title(f'{title_prefix}: Charger Utilization Distribution by Station Capacity', fontsize=16)
        ax.set_xlabel('Number of Chargers per Station', fontsize=12)
        ax.set_ylabel('Utilization (%)', fontsize=12)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "boxplot_charger_utilization_distribution.png"))
        plt.close(fig)

def generate_pooled_scatter_plot(list_of_dfs, output_path, title_prefix):
    os.makedirs(output_path, exist_ok=True)
    pooled_df = pd.concat(list_of_dfs, ignore_index=True)
    plot_data = pooled_df[pooled_df['num_of_charger'] > 0].copy()
    if not plot_data.empty:
        plot_data['jitter'] = plot_data['num_of_charger'] + np.random.normal(0, 0.1, size=len(plot_data))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(plot_data['jitter'], plot_data['avg_waiting_time_min'], alpha=0.3, s=40)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f'{title_prefix}: Pooled Daily Data - Chargers vs. Wait Time', fontsize=14)
        ax.set_xlabel('Number of Chargers', fontsize=12); ax.set_ylabel('Average Waiting Time (min)', fontsize=12)
        plt.grid(True, linestyle='--'); plt.tight_layout()
        plt.savefig(os.path.join(output_path, "chargers_vs_wait_time_pooled.png"), dpi=150)
        plt.close(fig)

def perform_and_save_correlation_analysis(financial_df, operational_df, candidate_df, output_path, title_prefix):
    os.makedirs(output_path, exist_ok=True)
    merged_df = pd.merge(financial_df, operational_df[['station_id', 'link_id']], on='station_id')
    final_df = pd.merge(merged_df, candidate_df, on='link_id', how='inner')
    corr_cols = ['od','rest_area','traffic','infra','interval','point','revenue','capex','opex','station_waiting_penalty']
    analysis_df = final_df[[col for col in corr_cols if col in final_df.columns]]
    if len(analysis_df.columns) < 2: return
    corr_matrix = analysis_df.corr()
    corr_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"), encoding='utf-8-sig')
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(f'{title_prefix}: Correlation Matrix of Site Attributes and Performance Metrics', fontsize=14); plt.tight_layout()
    plt.savefig(os.path.join(output_path, "correlation_heatmap.png"), dpi=200); plt.close()

def generate_and_save_period_average_scatter(average_operational_df, output_path, title_prefix):
    """
    Visualizes the relationship between the number of chargers and average wait time.
    """
    os.makedirs(output_path, exist_ok=True)
    plot_data = average_operational_df[average_operational_df['num_of_charger'] > 0].copy()
    
    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(plot_data['num_of_charger'], plot_data['avg_waiting_time_min'], alpha=0.8, s=60, color='royalblue')
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_title(f'{title_prefix}: Chargers vs. Period Average Wait Time', fontsize=14)
        ax.set_xlabel('Number of Chargers', fontsize=12)
        ax.set_ylabel('Period Average Waiting Time (min)', fontsize=12)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "chargers_vs_wait_time_period_average.png"), dpi=150)
        plt.close(fig)

def generate_and_save_charger_distribution_boxplot(operational_df, output_path, title_prefix):
    """
    [MODIFIED] 스테이션별 최종 충전기 개수의 분포를 세련된 스타일의 Boxplot으로 시각화합니다.
    """
    os.makedirs(output_path, exist_ok=True)
    plot_data = operational_df[operational_df['num_of_charger'] > 0].copy()
    
    if not plot_data.empty:
        plt.figure(figsize=(8, 8))
        ax = sns.boxplot(y=plot_data['num_of_charger'], palette="pastel", width=0.3) # 너비 얇게 조정
        plt.title(f'{title_prefix}: Distribution of Charger Counts per Station', fontsize=14)
        plt.ylabel('Number of Chargers', fontsize=12)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        save_path = os.path.join(output_path, "boxplot_charger_distribution.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

def generate_and_save_individual_metric_boxplots(financial_df, operational_df, output_path, title_prefix):
    """
    [MODIFIED] 주요 재무 및 운영 지표에 대한 Boxplot을 각각의 개별 이미지 파일로 저장합니다.
    (revenue, opex, capex, station_waiting_penalty, utilization_percentage)
    """
    print("  - Generating individual metric boxplots...")
    os.makedirs(output_path, exist_ok=True)
    
    # 두 데이터프레임 병합
    merged_df = pd.merge(financial_df, operational_df, on='station_id', how='inner')
    
    # 분석할 지표 목록
    metrics_to_plot = [
        'revenue', 'opex', 'capex', 'station_waiting_penalty', 'utilization_percentage'
    ]
    
    # 각 지표에 대해 개별 그래프 생성 및 저장
    for metric in metrics_to_plot:
        if metric not in merged_df.columns:
            print(f"  - Warning: Metric '{metric}' not found. Skipping.")
            continue
            
        plt.figure(figsize=(8, 8)) # 개별 그래프에 맞는 사이즈
        
        # 데이터 필터링 (e.g., utilization은 충전기가 있는 곳만)
        plot_data = merged_df
        if metric == 'utilization_percentage':
            plot_data = merged_df[merged_df['num_of_charger'] > 0]

        if plot_data.empty:
            continue

        ax = sns.boxplot(y=plot_data[metric], palette="pastel", width=0.3) # 너비 얇게 조정
        
        # 제목 및 라벨 설정
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f'{title_prefix}: Distribution of {metric_title}', fontsize=14)
        ax.set_ylabel('Value' if 'percentage' not in metric else 'Percentage (%)', fontsize=12)
        
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        
        # 개별 파일로 저장
        save_path = os.path.join(output_path, f"boxplot_{metric}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
    print("  - ✅ Individual metric boxplots generated.")

def generate_and_save_final_text_summary(
    all_daily_financials, 
    all_daily_operationals, 
    all_daily_truck_results, 
    all_daily_individual_waits,
    all_daily_individual_charging_durations,  # [신규] 전달받은 데이터 1
    all_daily_individual_charging_energies,   # [신규] 전달받은 데이터 2
    all_daily_individual_charger_utilizations, # [신규] 전달받은 데이터 3
    output_path
):
    """
    [수정]
    일별 시뮬레이션 결과를 기반으로 주요 성능 지표를 포함하는 최종 요약 보고서(텍스트 파일)를 생성합니다.
    [수정] 'Service Quality' 지표에 대해 '일별 평균의 평균'과 '전체 데이터 합산 평균'을 모두 리포트합니다.
    [수정] Method 2의 모든 지표를 개별 이벤트/데이터 기반의 통계(평균,표준편차,분위수)로 변경합니다.
    """
    print("  - Generating final text summary report...")
    os.makedirs(output_path, exist_ok=True)

    # --- Method 2 (전체 합산)을 위한 누적 변수 ---
    global_total_trucks = 0
    global_failed_trucks = 0
    
    # --- 1. 일별 결과 집계 (Method 1 기반) ---
    daily_summaries = []
    
    for i in range(len(all_daily_financials)):
        fin_df = all_daily_financials[i]
        op_df = all_daily_operationals[i]
        truck_df = all_daily_truck_results[i]
        
        if fin_df.empty or op_df.empty or truck_df.empty:
            continue
            
        # 재무 지표 (일별)
        total_revenue = fin_df['revenue'].sum()
        total_capex = fin_df['capex'].sum()
        total_opex = fin_df['opex'].sum()
        failed_truck_penalty = fin_df['failed_truck_penalty'].iloc[0]
        charger_penalty = fin_df['charger_penalty'].iloc[0]
        waiting_penalty = fin_df['waiting_penalty'].iloc[0]
        total_penalty = failed_truck_penalty + charger_penalty + waiting_penalty
        objective_function = total_revenue - total_capex - total_opex - total_penalty
        
        # === 서비스 품질 지표 (일별 집계) ===
        
        # 1. 성공률
        total_trucks_day = len(truck_df)
        failed_trucks_day = len(truck_df[truck_df['stopped_due_to_low_battery'] == True])
        success_rate_day = ((total_trucks_day - failed_trucks_day) / total_trucks_day) * 100 if total_trucks_day > 0 else 0
        
        # 2. 대기 시간 (일별 스테이션 평균들의 평균)
        avg_wait_time_day = op_df['avg_waiting_time_min'].mean() 
        
        # 3. 가동률 (일별 스테이션 가동률들의 평균)
        op_df_active_stations_day = op_df[op_df['num_of_charger'] > 0]
        avg_utilization_day = op_df_active_stations_day['utilization_percentage'].mean() if not op_df_active_stations_day.empty else 0

        # 4 & 5. 충전 시간 및 에너지 (일별 전체 평균)
        total_daily_events = op_df['total_charging_events'].sum()
        total_daily_energy = op_df['total_charged_energy_kWh'].sum()
        total_daily_duration = op_df['total_charging_duration_minutes'].sum()

        avg_charge_time_day = (total_daily_duration / total_daily_events) if total_daily_events > 0 else 0
        avg_charge_energy_day = (total_daily_energy / total_daily_events) if total_daily_events > 0 else 0

        distance_col = 'actual_traveled_distance_km' if 'actual_traveled_distance_km' in truck_df.columns else 'total_distance_planned'
        driving_energy_series_day = pd.to_numeric(truck_df[distance_col], errors='coerce').fillna(0) * 1.8
        charging_energy_series_day = pd.to_numeric(truck_df['total_charged_energy_kwh'], errors='coerce').fillna(0) if 'total_charged_energy_kwh' in truck_df.columns else pd.Series([0.0] * len(truck_df))
        charging_events_series_day = pd.to_numeric(truck_df['total_charging_events'], errors='coerce').fillna(0) if 'total_charging_events' in truck_df.columns else pd.Series([0.0] * len(truck_df))

        avg_driving_energy_day = driving_energy_series_day.mean() if not driving_energy_series_day.empty else 0
        avg_actual_charging_energy_per_vehicle_day = charging_energy_series_day.mean() if not charging_energy_series_day.empty else 0
        charged_vehicle_share_day = ((charging_energy_series_day > 0).mean() * 100.0) if not charging_energy_series_day.empty else 0
        avg_charging_events_per_vehicle_day = charging_events_series_day.mean() if not charging_events_series_day.empty else 0
        total_actual_charging_demand_day = charging_energy_series_day.sum() if not charging_energy_series_day.empty else 0

        installed_power_series_day = pd.to_numeric(op_df_active_stations_day['num_of_charger'], errors='coerce').fillna(0) * 200.0 if not op_df_active_stations_day.empty else pd.Series(dtype=float)
        peak_power_series_day = pd.to_numeric(op_df_active_stations_day['max_power_kW'], errors='coerce').fillna(0) if ('max_power_kW' in op_df_active_stations_day.columns and not op_df_active_stations_day.empty) else pd.Series(dtype=float)
        peak_ratio_series_day = pd.Series(
            np.where(installed_power_series_day > 0, (peak_power_series_day / installed_power_series_day) * 100.0, 0.0)
        ) if not installed_power_series_day.empty and not peak_power_series_day.empty else pd.Series(dtype=float)

        avg_installed_power_day = installed_power_series_day.mean() if not installed_power_series_day.empty else 0
        avg_peak_power_day = peak_power_series_day.mean() if not peak_power_series_day.empty else 0
        avg_peak_power_ratio_day = peak_ratio_series_day.mean() if not peak_ratio_series_day.empty else 0
        # --- (일별 집계 완료) ---

        daily_summaries.append({
            'revenue': total_revenue, 'capex': total_capex, 'opex': total_opex,
            'failed_truck_penalty': failed_truck_penalty, 'charger_penalty': charger_penalty,
            'waiting_penalty': waiting_penalty, 'objective_function': objective_function,
            'success_rate': success_rate_day,
            'avg_wait_time': avg_wait_time_day, 
            'avg_utilization': avg_utilization_day,
            'avg_charging_time': avg_charge_time_day,    
            'avg_charging_energy': avg_charge_energy_day,
            'avg_driving_energy_per_vehicle': avg_driving_energy_day,
            'avg_actual_charging_energy_per_vehicle': avg_actual_charging_energy_per_vehicle_day,
            'charged_vehicle_share': charged_vehicle_share_day,
            'avg_charging_events_per_vehicle': avg_charging_events_per_vehicle_day,
            'total_actual_charging_demand': total_actual_charging_demand_day,
            'avg_installed_power_per_station': avg_installed_power_day,
            'avg_peak_power_per_station': avg_peak_power_day,
            'avg_peak_power_ratio_per_station': avg_peak_power_ratio_day
        })
        
        # --- [수정] Method 2를 위해 트립 카운트만 누적 ---
        global_total_trucks += total_trucks_day
        global_failed_trucks += failed_trucks_day
        # (참고: 다른 개별 데이터 리스트는 이미 상위에서 집계되어 전달됨)
        
    if not daily_summaries:
        print("  - No valid daily data to generate final summary. Skipping.")
        return

    summary_df = pd.DataFrame(daily_summaries)
    num_days = len(summary_df)

    # --- [신규] 통계 계산 헬퍼 함수 ---
    def get_global_stats(data_list):
        """(일별 리스트의 리스트)를 1차원 리스트로 펼친 후 통계를 계산합니다."""
        # 1. 2D 리스트(일별)를 1D 리스트(전체)로 펼치기
        flattened_list = [item for sublist in data_list for item in sublist]
        
        if not flattened_list:
            return {'mean': 0, 'std': 0, 'q1': 0, 'q2': 0, 'q3': 0, 'p95': 0, 'n': 0}
        
        # 2. 1D 리스트로 통계 계산
        quantiles = np.quantile(flattened_list, [0.25, 0.50, 0.75, 0.95])
        return {
            'mean': np.mean(flattened_list),
            'std': np.std(flattened_list),
            'q1': quantiles[0],
            'q2': quantiles[1],
            'q3': quantiles[2],
            'p95': quantiles[3],
            'n': len(flattened_list)
        }

    def format_stats_line(stats_dict, unit):
        """통계 딕셔너리를 포맷팅된 문자열로 변환합니다."""
        return (f"{stats_dict['mean']:.2f} ({stats_dict['std']:.2f}) {unit} "
                f"  (Q1: {stats_dict['q1']:.2f}, Q2: {stats_dict['q2']:.2f}, "
                f"Q3: {stats_dict['q3']:.2f}, P95: {stats_dict['p95']:.2f}, N={stats_dict['n']})")

    # --- 2. Method 2 (전체 합산) 통계 계산 ---
    
    # 2a. 대기 시간 (Global) - 개별 "이벤트" 기반
    wait_time_stats_global = get_global_stats(all_daily_individual_waits)
    
    # 2b. 성공률 (Global) - 전체 "트립" 기반
    global_success_rate = ((global_total_trucks - global_failed_trucks) / global_total_trucks) * 100 if global_total_trucks > 0 else 0
    
    # 2c. [수정] 충전 시간 (Global) - "개별 이벤트" 기반
    charging_time_stats_global = get_global_stats(all_daily_individual_charging_durations)
    
    # 2d. [수정] 충전 에너지 (Global) - "개별 이벤트" 기반
    charging_energy_stats_global = get_global_stats(all_daily_individual_charging_energies)

    # 2e. [수정] 가동률 (Global) - "개별 충전기-일(per-charger-day)" 데이터 기반
    utilization_stats_global = get_global_stats(all_daily_individual_charger_utilizations)

    all_vehicle_driving_energies = []
    all_vehicle_actual_charging_energies = []
    all_vehicle_charging_events = []
    charged_vehicle_flags_global = []
    all_station_installed_powers = []
    all_station_peak_powers = []
    all_station_peak_power_ratios = []

    for truck_df in all_daily_truck_results:
        if truck_df is None or truck_df.empty:
            continue

        distance_col = 'actual_traveled_distance_km' if 'actual_traveled_distance_km' in truck_df.columns else 'total_distance_planned'
        distance_series = pd.to_numeric(truck_df[distance_col], errors='coerce').fillna(0)
        charging_energy_series = pd.to_numeric(truck_df['total_charged_energy_kwh'], errors='coerce').fillna(0) if 'total_charged_energy_kwh' in truck_df.columns else pd.Series([0.0] * len(truck_df))
        charging_event_series = pd.to_numeric(truck_df['total_charging_events'], errors='coerce').fillna(0) if 'total_charging_events' in truck_df.columns else pd.Series([0.0] * len(truck_df))

        all_vehicle_driving_energies.extend((distance_series * 1.8).tolist())
        all_vehicle_actual_charging_energies.extend(charging_energy_series.tolist())
        all_vehicle_charging_events.extend(charging_event_series.tolist())
        charged_vehicle_flags_global.extend((charging_energy_series > 0).astype(float).tolist())

    for operational_df in all_daily_operationals:
        if operational_df is None or operational_df.empty:
            continue

        active_operational_df = operational_df[operational_df['num_of_charger'] > 0].copy()
        if active_operational_df.empty:
            continue

        installed_power_series = pd.to_numeric(active_operational_df['num_of_charger'], errors='coerce').fillna(0) * 200.0
        peak_power_series = pd.to_numeric(active_operational_df['max_power_kW'], errors='coerce').fillna(0) if 'max_power_kW' in active_operational_df.columns else pd.Series([0.0] * len(active_operational_df))

        all_station_installed_powers.extend(installed_power_series.tolist())
        all_station_peak_powers.extend(peak_power_series.tolist())
        all_station_peak_power_ratios.extend(
            pd.Series(np.where(installed_power_series > 0, (peak_power_series / installed_power_series) * 100.0, 0.0)).tolist()
        )

    driving_energy_stats_global = get_global_stats([all_vehicle_driving_energies])
    actual_charging_energy_per_vehicle_stats_global = get_global_stats([all_vehicle_actual_charging_energies])
    charging_events_per_vehicle_stats_global = get_global_stats([all_vehicle_charging_events])
    charged_vehicle_share_stats_global = get_global_stats([pd.Series(charged_vehicle_flags_global) * 100.0])
    installed_power_stats_global = get_global_stats([all_station_installed_powers])
    observed_peak_power_stats_global = get_global_stats([all_station_peak_powers])
    peak_power_ratio_stats_global = get_global_stats([all_station_peak_power_ratios])


    # --- 3. 평균, 표준편차, 백분율 계산 (기존 로직 - Method 1용) ---
    def format_metric(series, scale=1e-6): # 백만 단위로 변환
        mean = series.mean() * scale
        std = series.std() * scale
        return f"{mean:,.2f} ({std:,.2f})"
        
    def format_metric_plain(series):
        mean = series.mean()
        std = series.std()
        return f"{mean:.2f} ({std:.2f})"
    
    def format_quantiles_plain(series): 
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50) # Median
        q3 = series.quantile(0.75)
        p95 = series.quantile(0.95)
        return f"  (Q1: {q1:.2f}, Q2: {q2:.2f}, Q3: {q3:.2f}, P95: {p95:.2f})"
    
    total_cost_series = summary_df['capex'] + summary_df['opex'] + summary_df['failed_truck_penalty'] + summary_df['charger_penalty'] + summary_df['waiting_penalty']
    total_penalty_series = summary_df['failed_truck_penalty'] + summary_df['charger_penalty'] + summary_df['waiting_penalty']
    total_cost_mean = total_cost_series.mean()

    def get_percentage(series, total_mean):
        if total_mean == 0: return "N/A"
        return f"{(series.mean() / total_mean) * 100:.1f}%"

    # --- 4. 보고서 문자열 생성 ---
    report_lines = [
        "--- Final Performance Summary ---\n",
        "전체 시뮬레이션 기간에 대한 일일 평균 및 표준편차입니다.\n",
        "=== Financial Metrics (Daily Average) ===",
        "값의 단위는 백만 원(million KRW)이며, '평균 (표준편차)' 형식입니다.\n",
        f"{'Objective Function (J):':<35} {format_metric(summary_df['objective_function'])}",
        f"{'Total Revenue (R_total):':<35} {format_metric(summary_df['revenue'])}",
        f"{'Total Cost:':<35} {format_metric(total_cost_series)}",
        f"  {'├─ Capital Cost (C_CAPEX):':<33} {format_metric(summary_df['capex'])}  [{get_percentage(summary_df['capex'], total_cost_mean)} of Total Cost]",
        f"  {'├─ Operation Cost (C_OPEX):':<33} {format_metric(summary_df['opex'])}  [{get_percentage(summary_df['opex'], total_cost_mean)} of Total Cost]",
        f"  {'└─ Total Penalty:':<33} {format_metric(total_penalty_series)}  [{get_percentage(total_penalty_series, total_cost_mean)} of Total Cost]",
        f"    {'  ├─ Trip Failure (Q_truck):':<31} {format_metric(summary_df['failed_truck_penalty'])}  [{get_percentage(summary_df['failed_truck_penalty'], total_cost_mean)} of Total Cost]",
        f"    {'  ├─ Over-installation (Q_budget):':<31} {format_metric(summary_df['charger_penalty'])}  [{get_percentage(summary_df['charger_penalty'], total_cost_mean)} of Total Cost]",
        f"    {'  └─ Delay (Q_delay):':<31} {format_metric(summary_df['waiting_penalty'])}  [{get_percentage(summary_df['waiting_penalty'], total_cost_mean)} of Total Cost]",
        "\n",
        "=== Service Quality Metrics ===\n",
        
        f"--- Method 1: Statistics of Daily Averages (N={num_days} days) ---",
        "'일별 평균값'들의 '평균 (표준편차)' 및 '(Q1, Q2, Q3, P95)' 형식입니다.\n",
        f"{'Successful Trip Rate:':<35} {format_metric_plain(summary_df['success_rate'])} % {format_quantiles_plain(summary_df['success_rate'])}",
        f"{'Queueing Time at Charger:':<35} {format_metric_plain(summary_df['avg_wait_time'])} min {format_quantiles_plain(summary_df['avg_wait_time'])}",
        f"{'Charger Utilization Rate:':<35} {format_metric_plain(summary_df['avg_utilization'])} % {format_quantiles_plain(summary_df['avg_utilization'])}",
        f"{'Average Charging Time:':<35} {format_metric_plain(summary_df['avg_charging_time'])} min {format_quantiles_plain(summary_df['avg_charging_time'])}",
        f"{'Average Charging Energy:':<35} {format_metric_plain(summary_df['avg_charging_energy'])} kWh {format_quantiles_plain(summary_df['avg_charging_energy'])}",
        "\n",
        "=== Demand & Infrastructure Interpretation ===\n",
        f"--- Method 1: Statistics of Daily Averages (N={num_days} days) ---",
        "'일별 평균값'들의 '평균 (표준편차)' 및 '(Q1, Q2, Q3, P95)' 형식입니다.\n",
        f"{'Total Actual Charging Demand:':<35} {format_metric_plain(summary_df['total_actual_charging_demand'])} kWh/day {format_quantiles_plain(summary_df['total_actual_charging_demand'])}",
        f"{'Driving Energy per Vehicle:':<35} {format_metric_plain(summary_df['avg_driving_energy_per_vehicle'])} kWh {format_quantiles_plain(summary_df['avg_driving_energy_per_vehicle'])}",
        f"{'Actual Charging Energy per Vehicle:':<35} {format_metric_plain(summary_df['avg_actual_charging_energy_per_vehicle'])} kWh {format_quantiles_plain(summary_df['avg_actual_charging_energy_per_vehicle'])}",
        f"{'Charging Events per Vehicle:':<35} {format_metric_plain(summary_df['avg_charging_events_per_vehicle'])} count {format_quantiles_plain(summary_df['avg_charging_events_per_vehicle'])}",
        f"{'Charged Vehicle Share:':<35} {format_metric_plain(summary_df['charged_vehicle_share'])} % {format_quantiles_plain(summary_df['charged_vehicle_share'])}",
        f"{'Installed Power per Station:':<35} {format_metric_plain(summary_df['avg_installed_power_per_station'])} kW {format_quantiles_plain(summary_df['avg_installed_power_per_station'])}",
        f"{'Observed Peak Power per Station:':<35} {format_metric_plain(summary_df['avg_peak_power_per_station'])} kW {format_quantiles_plain(summary_df['avg_peak_power_per_station'])}",
        f"{'Peak Power Ratio per Station:':<35} {format_metric_plain(summary_df['avg_peak_power_ratio_per_station'])} % {format_quantiles_plain(summary_df['avg_peak_power_ratio_per_station'])}",
        "\n",
        
        # --- [수정] Method 2 출력 부분 ---
        f"--- Method 2: Global Statistics (N={global_total_trucks} trips) ---",
        "전체 기간의 개별 이벤트/데이터를 합산하여 계산한 통계입니다.\n",
        f"{'Successful Trip Rate (Global):':<35} {global_success_rate:.2f} %",
        f"{'Queueing Time (per-event):':<35} {format_stats_line(wait_time_stats_global, 'min')}",
        f"{'Charger Utilization (per-charger-day):':<35} {format_stats_line(utilization_stats_global, '%')}",
        f"{'Average Charging Time (per-event):':<35} {format_stats_line(charging_time_stats_global, 'min')}",
        f"{'Average Charging Energy (per-event):':<35} {format_stats_line(charging_energy_stats_global, 'kWh')}",
        f"{'Driving Energy per Vehicle:':<35} {format_stats_line(driving_energy_stats_global, 'kWh')}",
        f"{'Actual Charging Energy per Vehicle:':<35} {format_stats_line(actual_charging_energy_per_vehicle_stats_global, 'kWh')}",
        f"{'Charging Events per Vehicle:':<35} {format_stats_line(charging_events_per_vehicle_stats_global, 'count')}",
        f"{'Charged Vehicle Share:':<35} {format_stats_line(charged_vehicle_share_stats_global, '%')}",
        f"{'Installed Power per Station:':<35} {format_stats_line(installed_power_stats_global, 'kW')}",
        f"{'Observed Peak Power per Station:':<35} {format_stats_line(observed_peak_power_stats_global, 'kW')}",
        f"{'Peak Power Ratio per Station:':<35} {format_stats_line(peak_power_ratio_stats_global, '%')}"
    ]

    report_string = "\n".join(report_lines)

    # --- 5. 보고서 파일 저장 ---
    report_path = os.path.join(output_path, "final_summary_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_string)
    print(f"  - ✅ Final summary report saved to {report_path}")


def _calculate_mean_sd(values):
    numeric_values = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    if numeric_values.empty:
        return 0.0, 0.0, 0

    mean_val = float(numeric_values.mean())
    std_val = float(numeric_values.std(ddof=1)) if len(numeric_values) > 1 else 0.0
    return mean_val, std_val, int(len(numeric_values))


def build_penetration_rate_summary(experiment_name, penetration_rate, all_daily_truck_results, all_daily_operationals):
    """
    Builds a compact summary for one penetration-rate experiment.

    Aggregation basis:
    - sample size: daily number of simulated vehicles
    - total energy demand: daily total charging energy demand
    - travel distance / charging energy: pooled vehicle-day observations
    - station metrics: pooled operational station-day observations
    """
    daily_sample_sizes = []
    daily_total_energy_demands = []
    travel_distances = []
    driving_energy_demands = []
    charging_energies = []
    charging_events = []
    charged_vehicle_flags = []
    station_max_powers = []
    station_max_charging_counts = []
    station_installed_powers = []
    station_peak_power_ratios = []

    for truck_df in all_daily_truck_results:
        if truck_df is None or truck_df.empty:
            continue

        daily_sample_sizes.append(len(truck_df))

        distance_col = 'actual_traveled_distance_km' if 'actual_traveled_distance_km' in truck_df.columns else 'total_distance_planned'
        distance_series = pd.to_numeric(truck_df[distance_col], errors='coerce').fillna(0)
        travel_distances.extend(distance_series.tolist())
        driving_energy_demands.extend((distance_series * 1.8).tolist())

        if 'total_charged_energy_kwh' in truck_df.columns:
            truck_energy_series = pd.to_numeric(truck_df['total_charged_energy_kwh'], errors='coerce').fillna(0)
            charging_energies.extend(truck_energy_series.tolist())
            daily_total_energy_demands.append(float(truck_energy_series.sum()))
            charged_vehicle_flags.extend((truck_energy_series > 0).astype(float).tolist())
        else:
            charging_energies.extend([0.0] * len(truck_df))
            daily_total_energy_demands.append(0.0)
            charged_vehicle_flags.extend([0.0] * len(truck_df))

        if 'total_charging_events' in truck_df.columns:
            charging_events.extend(pd.to_numeric(truck_df['total_charging_events'], errors='coerce').fillna(0).tolist())
        else:
            charging_events.extend([0.0] * len(truck_df))

    for operational_df in all_daily_operationals:
        if operational_df is None or operational_df.empty:
            continue

        active_operational_df = operational_df[operational_df['num_of_charger'] > 0].copy()
        if active_operational_df.empty:
            continue

        installed_power_series = pd.to_numeric(active_operational_df['num_of_charger'], errors='coerce').fillna(0) * 200.0
        station_installed_powers.extend(installed_power_series.tolist())

        if 'max_power_kW' in active_operational_df.columns:
            peak_power_series = pd.to_numeric(active_operational_df['max_power_kW'], errors='coerce').fillna(0)
            station_max_powers.extend(peak_power_series.tolist())
            peak_ratio_series = np.where(installed_power_series > 0, (peak_power_series / installed_power_series) * 100.0, 0.0)
            station_peak_power_ratios.extend(pd.Series(peak_ratio_series).fillna(0).tolist())

        if 'max_simultaneous_charging_vehicles' in active_operational_df.columns:
            station_max_charging_counts.extend(
                pd.to_numeric(active_operational_df['max_simultaneous_charging_vehicles'], errors='coerce').fillna(0).tolist()
            )

    sample_mean, sample_sd, sample_obs = _calculate_mean_sd(daily_sample_sizes)
    total_energy_demand_mean, total_energy_demand_sd, total_energy_demand_obs = _calculate_mean_sd(daily_total_energy_demands)
    distance_mean, distance_sd, distance_obs = _calculate_mean_sd(travel_distances)
    driving_energy_mean, driving_energy_sd, driving_energy_obs = _calculate_mean_sd(driving_energy_demands)
    energy_mean, energy_sd, energy_obs = _calculate_mean_sd(charging_energies)
    charging_events_mean, charging_events_sd, charging_events_obs = _calculate_mean_sd(charging_events)
    charged_share_mean, charged_share_sd, charged_share_obs = _calculate_mean_sd(pd.Series(charged_vehicle_flags) * 100.0)
    installed_power_mean, installed_power_sd, installed_power_obs = _calculate_mean_sd(station_installed_powers)
    power_mean, power_sd, power_obs = _calculate_mean_sd(station_max_powers)
    charging_count_mean, charging_count_sd, charging_count_obs = _calculate_mean_sd(station_max_charging_counts)
    peak_ratio_mean, peak_ratio_sd, peak_ratio_obs = _calculate_mean_sd(station_peak_power_ratios)

    return {
        'experiment_name': experiment_name,
        'penetration_rate_percent': penetration_rate,
        'simulated_days': len(daily_sample_sizes),
        'sample_size_mean': sample_mean,
        'sample_size_sd': sample_sd,
        'sample_size_observations': sample_obs,
        'total_energy_demand_mean_kWh_per_day': total_energy_demand_mean,
        'total_energy_demand_sd_kWh_per_day': total_energy_demand_sd,
        'total_energy_demand_observations': total_energy_demand_obs,
        'travel_distance_per_vehicle_mean_km': distance_mean,
        'travel_distance_per_vehicle_sd_km': distance_sd,
        'travel_distance_vehicle_observations': distance_obs,
        'driving_energy_per_vehicle_mean_kWh': driving_energy_mean,
        'driving_energy_per_vehicle_sd_kWh': driving_energy_sd,
        'driving_energy_vehicle_observations': driving_energy_obs,
        'charging_energy_per_vehicle_mean_kWh': energy_mean,
        'charging_energy_per_vehicle_sd_kWh': energy_sd,
        'charging_energy_vehicle_observations': energy_obs,
        'charging_events_per_vehicle_mean': charging_events_mean,
        'charging_events_per_vehicle_sd': charging_events_sd,
        'charging_events_vehicle_observations': charging_events_obs,
        'charged_vehicle_share_mean_percent': charged_share_mean,
        'charged_vehicle_share_sd_percent': charged_share_sd,
        'charged_vehicle_share_observations': charged_share_obs,
        'installed_power_per_station_mean_kW': installed_power_mean,
        'installed_power_per_station_sd_kW': installed_power_sd,
        'installed_power_station_observations': installed_power_obs,
        'maximum_power_per_station_mean_kW': power_mean,
        'maximum_power_per_station_sd_kW': power_sd,
        'maximum_power_station_observations': power_obs,
        'maximum_charging_vehicles_per_station_mean': charging_count_mean,
        'maximum_charging_vehicles_per_station_sd': charging_count_sd,
        'maximum_charging_vehicles_station_observations': charging_count_obs,
        'peak_power_ratio_per_station_mean_percent': peak_ratio_mean,
        'peak_power_ratio_per_station_sd_percent': peak_ratio_sd,
        'peak_power_ratio_station_observations': peak_ratio_obs,
    }


def save_penetration_rate_summary(summary_record, output_path):
    """Saves the penetration-rate summary for a single experiment."""
    print("  - Saving penetration-rate vehicle/station summary...")
    os.makedirs(output_path, exist_ok=True)

    summary_df = pd.DataFrame([summary_record])
    csv_path = os.path.join(output_path, "penetration_rate_vehicle_station_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    txt_lines = [
        "--- Penetration Rate Vehicle/Station Summary ---",
        f"Experiment: {summary_record['experiment_name']}",
        f"Penetration Rate: {summary_record['penetration_rate_percent']}%",
        f"Simulated days: {summary_record['simulated_days']}",
        "",
        "Aggregation basis:",
        "- Sample size: daily number of simulated vehicles",
        "- Total energy demand: daily total charging energy demand",
        "- Vehicle metrics: pooled vehicle-day observations across all simulated days",
        "- Station metrics: pooled operational station-day observations across all simulated days",
        "",
        f"Sample size (number of vehicles): {summary_record['sample_size_mean']:.2f} ({summary_record['sample_size_sd']:.2f})",
        f"Total energy demand (kWh/day): {summary_record['total_energy_demand_mean_kWh_per_day']:.2f} ({summary_record['total_energy_demand_sd_kWh_per_day']:.2f})",
        f"Travel distance per vehicle (km): {summary_record['travel_distance_per_vehicle_mean_km']:.2f} ({summary_record['travel_distance_per_vehicle_sd_km']:.2f})",
        f"Driving energy per vehicle (kWh): {summary_record['driving_energy_per_vehicle_mean_kWh']:.2f} ({summary_record['driving_energy_per_vehicle_sd_kWh']:.2f})",
        f"Actual charging energy per vehicle (kWh): {summary_record['charging_energy_per_vehicle_mean_kWh']:.2f} ({summary_record['charging_energy_per_vehicle_sd_kWh']:.2f})",
        f"Charging events per vehicle: {summary_record['charging_events_per_vehicle_mean']:.2f} ({summary_record['charging_events_per_vehicle_sd']:.2f})",
        f"Charged vehicle share (%): {summary_record['charged_vehicle_share_mean_percent']:.2f} ({summary_record['charged_vehicle_share_sd_percent']:.2f})",
        f"Installed power per station (kW): {summary_record['installed_power_per_station_mean_kW']:.2f} ({summary_record['installed_power_per_station_sd_kW']:.2f})",
        f"Peak observed power per station (kW): {summary_record['maximum_power_per_station_mean_kW']:.2f} ({summary_record['maximum_power_per_station_sd_kW']:.2f})",
        f"Peak power ratio per station (%): {summary_record['peak_power_ratio_per_station_mean_percent']:.2f} ({summary_record['peak_power_ratio_per_station_sd_percent']:.2f})",
        f"Peak simultaneous charging vehicles per station: {summary_record['maximum_charging_vehicles_per_station_mean']:.2f} ({summary_record['maximum_charging_vehicles_per_station_sd']:.2f})",
    ]

    txt_path = os.path.join(output_path, "penetration_rate_vehicle_station_summary.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(txt_lines))

    print(f"  - ✅ Penetration-rate summary saved to {csv_path}")


def save_penetration_rate_comparison(summary_records, output_path):
    """Saves a comparison table across all penetration-rate experiments."""
    if not summary_records:
        return

    os.makedirs(output_path, exist_ok=True)
    comparison_df = pd.DataFrame(summary_records).sort_values(by='penetration_rate_percent').reset_index(drop=True)
    csv_path = os.path.join(output_path, "penetration_rate_comparison.csv")
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    display_columns = [
        'penetration_rate_percent',
        'simulated_days',
        'sample_size_mean',
        'sample_size_sd',
        'total_energy_demand_mean_kWh_per_day',
        'total_energy_demand_sd_kWh_per_day',
        'travel_distance_per_vehicle_mean_km',
        'travel_distance_per_vehicle_sd_km',
        'driving_energy_per_vehicle_mean_kWh',
        'driving_energy_per_vehicle_sd_kWh',
        'charging_energy_per_vehicle_mean_kWh',
        'charging_energy_per_vehicle_sd_kWh',
        'charging_events_per_vehicle_mean',
        'charging_events_per_vehicle_sd',
        'charged_vehicle_share_mean_percent',
        'charged_vehicle_share_sd_percent',
        'installed_power_per_station_mean_kW',
        'installed_power_per_station_sd_kW',
        'maximum_power_per_station_mean_kW',
        'maximum_power_per_station_sd_kW',
        'peak_power_ratio_per_station_mean_percent',
        'peak_power_ratio_per_station_sd_percent',
        'maximum_charging_vehicles_per_station_mean',
        'maximum_charging_vehicles_per_station_sd',
    ]
    display_df = comparison_df[display_columns].copy()
    for col in display_df.columns:
        if col not in ['penetration_rate_percent', 'simulated_days']:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

    txt_path = os.path.join(output_path, "penetration_rate_comparison.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("--- Penetration Rate Comparison ---\n")
        f.write("Sample size and total energy demand are summarized per simulated day; vehicle metrics use pooled vehicle-day observations; station metrics use pooled operational station-day observations.\n\n")
        f.write(display_df.to_string(index=False))

    print(f"✅ Penetration-rate comparison saved to {csv_path}")

def run_single_simulation_task(day_str, car_paths_base_folder, station_df_copy, sim_params, main_output_folder):
    """Worker function for multiprocessing."""
    random.seed(); np.random.seed()
    day_folder_path = os.path.join(car_paths_base_folder, day_str)
    try:
        car_paths_df = load_car_path_df_for_day(day_folder_path, sim_params['NUMBER_OF_TRUCKS'])
        if car_paths_df.empty: return None
    except FileNotFoundError: return None

    sim = AnalyticsSimulator(
        car_paths_df, station_df_copy,
        sim_params['UNIT_TIME_MIN'], sim_params['SIMULATING_HOURS'],
        sim_params['NUMBER_OF_TRUCKS'], sim_params['NUMBER_OF_MAX_CHARGERS'], sim_params['TRUCK_STEP_FREQUENCY']
    )
    sim.prepare_simulation()
    sim.run_simulation()
    
    # [수정] 6개의 값을 반환받음
    financial_df, operational_df, all_waiting_times_day, \
    all_charger_utilizations_day, all_charging_durations_day, \
    all_charging_energies_day = sim.get_results()

    summary = {
        'day': day_str, 'week_num': pd.to_datetime(day_str).isocalendar().week,
        'revenue': financial_df['revenue'].sum(), 'opex': financial_df['opex'].sum(),
        'capex': financial_df['capex'].sum(),
        'waiting_penalty': financial_df['station_waiting_penalty'].sum() if 'station_waiting_penalty' in financial_df.columns else 0
    }
    
    # [수정] 9개의 값을 튜플로 반환
    return (
        financial_df, operational_df, summary, 
        sim.completed_trucks_data, sim.truck_results_df, 
        all_waiting_times_day, all_charger_utilizations_day, 
        all_charging_durations_day, all_charging_energies_day
    )

def load_chargers_from_ga_result(ga_folder_path):
    """
    Loads charger counts from the last line of the latest GA result file.
    """
    try:
        best_ind_path = os.path.join(ga_folder_path, 'best_individuals')
        if not os.path.isdir(best_ind_path):
            print(f"  - Warning: '{best_ind_path}' not found. Skipping.")
            return None

        files = os.listdir(best_ind_path)
        gen_files = {}
        for f_name in files:
            match = re.match(r'g\d+-(\d+)', f_name)
            if match:
                end_generation = int(match.group(1))
                gen_files[end_generation] = f_name
        
        if not gen_files:
            print(f"  - Warning: No generation files found in '{best_ind_path}'. Skipping.")
            return None
            
        latest_file_name = gen_files[max(gen_files.keys())]
        file_path = os.path.join(best_ind_path, latest_file_name)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines: return None
            last_line = lines[-1]
        
        charger_counts_str = last_line.strip().split(',')[:-2]
        charger_counts = [int(count) for count in charger_counts_str]
        
        return charger_counts

    except Exception as e:
        print(f"  - Error processing charger data in {ga_folder_path}: {e}")
        return None

def generate_and_save_ga_history_plot(ga_folder_path, output_path, downsample_n=500):
    """
    [MODIFIED] GA 결과 폴더의 모든 세대 파일을 읽어, Fitness와 총 충전기 개수 변화를
    각각의 개별 이미지 파일로 시각화하고 저장합니다. (폰트: Times New Roman, 제목 없음)
    """
    print(f"  - Generating GA history plots...")
    best_ind_path = os.path.join(ga_folder_path, 'best_individuals')
    if not os.path.isdir(best_ind_path):
        print(f"  - Warning: '{best_ind_path}' directory not found. Skipping GA history plot.")
        return

    try:
        # Set the font to Times New Roman for all plot elements
        plt.rcParams['font.family'] = 'Times New Roman'
        
        files = [f for f in os.listdir(best_ind_path) if f.startswith('g')]
        if not files:
            print(f"  - Warning: No GA result files found in '{best_ind_path}'.")
            return

        all_gen_data = [pd.read_csv(os.path.join(best_ind_path, f_name)) for f_name in sorted(files)]
        
        if not all_gen_data:
            print("  - Error: No valid data could be loaded from GA result files.")
            return

        full_df = pd.concat(all_gen_data, ignore_index=True)
        full_df.sort_values(by='Actual_Generation', inplace=True)
        full_df.drop_duplicates(subset=['Actual_Generation'], keep='last', inplace=True)

        # --- 데이터 가공 (이동평균 및 수렴값 계산) ---
        station_cols = [col for col in full_df.columns if col.lower().startswith('station_')]
        if not station_cols:
            print("  - Error: Could not find station columns to calculate total chargers.")
            return
            
        full_df['Total_Chargers'] = full_df[station_cols].sum(axis=1)
        
        window_size = 15
        full_df['Fitness_MA'] = full_df['Fitness'].rolling(window=window_size, min_periods=1).mean()
        full_df['Total_Chargers_MA'] = full_df['Total_Chargers'].rolling(window=window_size, min_periods=1).mean()

        # [REMOVED] 수렴 범위 계산 로직 제거
        # final_fitness_val = full_df['Fitness'].tail(window_size).mean()
        # final_chargers_val = full_df['Total_Chargers'].tail(window_size).mean()
        # fitness_upper = final_fitness_val * 1.0075
        # fitness_lower = final_fitness_val * 0.9925
        # chargers_upper = final_chargers_val * 1.0075
        # chargers_lower = final_chargers_val * 0.9925

        plot_df = full_df
        num_generations = len(plot_df)
        
        if num_generations > downsample_n * 1.5:
            step = num_generations // downsample_n
            plot_df = full_df.iloc[::step, :]
            print(f"  - Downsampling GA history from {num_generations} to {len(plot_df)} points for plotting.")

        os.makedirs(output_path, exist_ok=True)

        # --- [수정] Plot 1: Fitness Score (개별 파일) ---
        fig1, ax1 = plt.subplots(figsize=(21, 9)) # 21:9 비율 설정

        ax1.plot(plot_df['Actual_Generation'], plot_df['Fitness'], marker='', linestyle='-', color='lightskyblue', alpha=0.6, label='Fitness Score (100 million KRW/day)')
        ax1.plot(plot_df['Actual_Generation'], plot_df['Fitness_MA'], marker='.', markersize=5, linestyle='-', color='dodgerblue', label=f'Moving Average of Fitness Score over {window_size} Generations')
        
        # [REMOVED] 수렴 범위 점선 제거
        # ax1.axhline(y=fitness_upper, color='red', linestyle='--', linewidth=1.5, label='Convergence Range (±0.75%)')
        # ax1.axhline(y=fitness_lower, color='red', linestyle='--', linewidth=1.5)

        # ax1.set_title('Fitness Score over Generations', fontsize=30) # 제목 제거
        ax1.set_xlabel('Generation', fontsize=30)
        ax1.set_ylabel('Fitness Score', fontsize=30)
        ax1.tick_params(axis='both', which='major', labelsize=26)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=26)
        
        plt.tight_layout()
        fitness_save_path = os.path.join(output_path, 'ga_fitness_history.png')
        plt.savefig(fitness_save_path, dpi=150)
        plt.close(fig1)
        print(f"  - ✅ Fitness history plot saved to {fitness_save_path}")

        # --- [수정] Plot 2: Total Chargers (개별 파일) ---
        fig2, ax2 = plt.subplots(figsize=(21, 9)) # 21:9 비율 설정

        ax2.plot(plot_df['Actual_Generation'], plot_df['Total_Chargers'], marker='', linestyle='-', color='lightgreen', alpha=0.7, label='Number of Chargers in Solution')
        ax2.plot(plot_df['Actual_Generation'], plot_df['Total_Chargers_MA'], marker='.', markersize=5, linestyle='-', color='seagreen', label=f'Moving Average of Number of Chargers over {window_size} Generations')

        # [REMOVED] 수렴 범위 점선 제거
        # ax2.axhline(y=chargers_upper, color='red', linestyle='--', linewidth=1.5, label='Convergence Range (±0.75%)')
        # ax2.axhline(y=chargers_lower, color='red', linestyle='--', linewidth=1.5)

        # ax2.set_title('Total Number of Chargers over Generations', fontsize=30) # 제목 제거
        ax2.set_xlabel('Generation', fontsize=30)
        ax2.set_ylabel('Total Chargers', fontsize=30)
        ax2.tick_params(axis='both', which='major', labelsize=26)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=26)
        
        plt.tight_layout()
        chargers_save_path = os.path.join(output_path, 'ga_chargers_history.png')
        plt.savefig(chargers_save_path, dpi=150)
        plt.close(fig2)
        print(f"  - ✅ Total chargers history plot saved to {chargers_save_path}")

    except Exception as e:
        print(f"  - An unexpected error occurred while generating GA history plots: {e}")



if __name__ == '__main__':
    # --- Basic Path Settings ---
    CAR_PATHS_BASE_FOLDER = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Trajectory(DAY_90km)"
    STATION_TEMPLATE_PATH = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\Final_Candidates_Selected.csv"
    CANDIDATE_FILE_PATH = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\candidate\Final_Candidates\Final_Candidates_Selected.csv"
    RESULTS_BASE_FOLDER = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\result_for_kori"
    SHAPEFILE_PATH = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Raw_Data\main_road_network_level_5.5\level5_5_link_probe_32_2020.shp"
    GA_RESULTS_BASE_FOLDER = r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\GA_results"

    # --- Basic Simulation Parameters ---
    SIM_PARAMS = {
        'SIMULATING_HOURS': 30, 
        'UNIT_TIME_MIN': 5, 
        'TRUCK_STEP_FREQUENCY': 1,
        'NUMBER_OF_TRUCKS': 0, 
        'NUMBER_OF_MAX_CHARGERS': 10000
    }
    NUM_WEEKS_TO_SIMULATE = 2
    NUM_PROCESSES = min(os.cpu_count(), NUM_WEEKS_TO_SIMULATE * 5)

    # --- Load Common Data ---
    overall_start_time = time.time()
    print("Loading common data (Shapefile, Candidate Info)...")
    link_geometries_gdf = None
    try:
        station_template_df = load_station_df(STATION_TEMPLATE_PATH)
        candidate_df = load_candidate_df(CANDIDATE_FILE_PATH)
        weeks_to_run = get_simulation_dates(CAR_PATHS_BASE_FOLDER, NUM_WEEKS_TO_SIMULATE)
        print("✅ Common data loaded successfully.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading common data: {e}"); exit()

    try:
        link_geometries_gdf = load_link_geometries(SHAPEFILE_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not load shapefile geometry for map visualization: {e}")
        print("         Truck map visualization will be skipped, but the simulation will continue.")

    # --- Run Analysis for Each GA Experiment Folder ---
    penetration_rate_summary_records = []
    for experiment_folder_name in os.listdir(GA_RESULTS_BASE_FOLDER):
        experiment_folder_path = os.path.join(GA_RESULTS_BASE_FOLDER, experiment_folder_name)
        if not os.path.isdir(experiment_folder_path):
            continue

        print(f"\n{'='*25} Starting Analysis for: {experiment_folder_name} {'='*25}")
        
        main_output_folder = os.path.join(RESULTS_BASE_FOLDER, experiment_folder_name)
        
        # 1. Generate GA history plots
        generate_and_save_ga_history_plot(experiment_folder_path, main_output_folder)

        try:
            # 2. Set up simulation parameters
            match = re.search(r'(\d+)%', experiment_folder_name)
            if not match:
                print(f"  - Warning: Could not parse penetration rate from '{experiment_folder_name}'. Skipping.")
                continue
            
            penetration_rate = int(match.group(1))
            SIM_PARAMS['NUMBER_OF_TRUCKS'] = int(3050 * (penetration_rate / 10.0))
            print(f"  - Parameter Set: Penetration Rate={penetration_rate}%, Num Trucks={SIM_PARAMS['NUMBER_OF_TRUCKS']}")
            
            charger_counts = load_chargers_from_ga_result(experiment_folder_path)
            if charger_counts is None: continue

            station_df = station_template_df.copy()
            if len(station_df) != len(charger_counts):
                print(f"  - Error: Mismatch between stations ({len(station_df)}) and charger counts ({len(charger_counts)}).")
                continue
            station_df['num_of_charger'] = charger_counts
            print(f"  - Successfully loaded {sum(charger_counts)} chargers across {len(station_df)} stations.")

        except Exception as e:
            print(f"  - Failed to set up parameters for '{experiment_folder_name}': {e}")
            continue

        # 3. Execute simulations in parallel
        tasks = [(day, CAR_PATHS_BASE_FOLDER, station_df.copy(), SIM_PARAMS, None) for week, days in weeks_to_run.items() for day in days]
        
        print(f"  - Executing {len(tasks)} simulation tasks in parallel...")
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            all_results = pool.starmap(run_single_simulation_task, tasks)
        print("  - All parallel tasks completed.")

        valid_results = [res for res in all_results if res is not None]
        if not valid_results:
            print("  - No valid simulation results were generated for this experiment. Skipping post-processing.")
            continue
        
        print(f"  - Aggregating results and generating reports in: {main_output_folder}")

        # 4. Aggregate all results from parallel tasks
        all_daily_financials = [res[0] for res in valid_results]
        all_daily_operationals = [res[1] for res in valid_results]
        all_daily_summaries = [res[2] for res in valid_results]
        all_completed_trucks_data = [res[3] for res in valid_results]
        all_daily_truck_results = [res[4] for res in valid_results]
        all_daily_individual_waits = [res[5] for res in valid_results]
        all_daily_individual_charger_utilizations = [res[6] for res in valid_results] # [신규]
        all_daily_individual_charging_durations = [res[7] for res in valid_results]   # [신규]
        all_daily_individual_charging_energies = [res[8] for res in valid_results]    # [신규]

        print("  - Generating daily reports and visualizations...")
        for i in range(len(valid_results)):
            financial_df = all_daily_financials[i]
            operational_df = all_daily_operationals[i]
            summary = all_daily_summaries[i]
            completed_trucks_data = all_completed_trucks_data[i]
            truck_results_df = all_daily_truck_results[i]
            
            day_str = summary['day']
            daily_path = os.path.join(main_output_folder, "Daily_Reports", day_str)
            title_prefix = f"Daily Analysis ({day_str})"
            
            # 일별 요약 보고서 및 시각화 생성
            generate_and_save_enhanced_summary_report(financial_df, operational_df, truck_results_df, daily_path, title_prefix)
            
            # # 일별 스테이션 시간대별 그래프 생성
            # timeseries_path = os.path.join(daily_path, "Timeseries_Graphs")
            # generate_and_save_timeseries_graphs(operational_df, timeseries_path, SIM_PARAMS['UNIT_TIME_MIN'])
            
            # # 일별 트럭 경로 시각화 생성
            # truck_vis_path = os.path.join(daily_path, "Truck_Visualizations")
            # generate_truck_visualizations(
            #     output_folder=truck_vis_path,
            #     truck_results_df=truck_results_df,
            #     completed_trucks_data=completed_trucks_data,
            #     link_geometries_gdf=link_geometries_gdf,
            #     num_trucks_to_visualize=100
            # )
        print("  - ✅ Daily reports completed.")

        # 5. Generate weekly reports
        summary_df_all = pd.DataFrame(all_daily_summaries)
        for week_num in summary_df_all['week_num'].unique():
            print(f"  - Generating reports for Week {week_num}...")
            week_indices = summary_df_all.index[summary_df_all['week_num'] == week_num].tolist()
            
            weekly_financials = [all_daily_financials[i] for i in week_indices]
            weekly_operationals = [all_daily_operationals[i] for i in week_indices]
            weekly_summaries = [all_daily_summaries[i] for i in week_indices]
            weekly_truck_results = [all_daily_truck_results[i] for i in week_indices]

            if not weekly_financials: continue
            
            weekly_path = os.path.join(main_output_folder, "Weekly_Reports", f"Week_{week_num:02d}")
            
            avg_fin_wk = pd.concat(weekly_financials).groupby('station_id').mean(numeric_only=True).reset_index()
            op_for_avg_wk = pd.concat(weekly_operationals)
            avg_op_wk = op_for_avg_wk.drop(
                columns=['queue_history_raw', 'charging_history_raw', 'power_history_raw',
                         'cumulative_arrivals_history', 'cumulative_departures_history'],
                errors='ignore'
            )
            avg_op_wk = avg_op_wk.groupby(['station_id', 'num_of_charger']).mean(numeric_only=True).reset_index()
            concat_truck_results_wk = pd.concat(weekly_truck_results, ignore_index=True)

            generate_and_save_enhanced_summary_report(avg_fin_wk, avg_op_wk, concat_truck_results_wk, weekly_path, f"Weekly Average (W{week_num})")
            #generate_and_save_aggregated_boxplots(weekly_financials, weekly_operationals, weekly_summaries, weekly_path, f"Weekly (W{week_num})")
            #generate_pooled_scatter_plot(weekly_operationals, weekly_path, f"Weekly (W{week_num})")
            #perform_and_save_correlation_analysis(avg_fin_wk, avg_op_wk, candidate_df, weekly_path, f"Weekly Average (W{week_num})")
            
        # 6. Generate total period reports
        if all_daily_financials:
            print(f"  - Generating Total Period Reports...")
            total_path = os.path.join(main_output_folder, "Total_Report")

            avg_fin_tot = pd.concat(all_daily_financials).groupby('station_id').mean(numeric_only=True).reset_index()
            op_for_avg_tot = pd.concat(all_daily_operationals).drop(
                columns=['queue_history_raw', 'charging_history_raw', 'power_history_raw',
                         'cumulative_arrivals_history', 'cumulative_departures_history'],
                errors='ignore'
            )
            avg_op_tot = op_for_avg_tot.groupby(['station_id', 'num_of_charger']).mean(numeric_only=True).reset_index()
            concat_truck_results_tot = pd.concat(all_daily_truck_results, ignore_index=True)

            generate_and_save_enhanced_summary_report(avg_fin_tot, avg_op_tot, concat_truck_results_tot, total_path, "Total Period Average")
            #generate_and_save_aggregated_boxplots(all_daily_financials, all_daily_operationals, all_daily_summaries, total_path, "Total Period")
            #generate_pooled_scatter_plot(all_daily_operationals, total_path, "Total Period")
            #perform_and_save_correlation_analysis(avg_fin_tot, avg_op_tot, candidate_df, total_path, "Total Period Average")
            
            #generate_and_save_charger_distribution_boxplot(avg_op_tot, total_path, "Total Period")
            #generate_and_save_individual_metric_boxplots(avg_fin_tot, avg_op_tot, total_path, "Total Period Average")

            generate_and_save_final_text_summary(
                all_daily_financials, 
                all_daily_operationals, 
                all_daily_truck_results,
                all_daily_individual_waits,
                all_daily_individual_charging_durations,    # [신규]
                all_daily_individual_charging_energies,     # [신규]
                all_daily_individual_charger_utilizations,  # [신규]
                total_path
            )

            penetration_summary = build_penetration_rate_summary(
                experiment_folder_name,
                penetration_rate,
                all_daily_truck_results,
                all_daily_operationals
            )
            save_penetration_rate_summary(penetration_summary, total_path)
            penetration_rate_summary_records.append(penetration_summary)

    save_penetration_rate_comparison(penetration_rate_summary_records, RESULTS_BASE_FOLDER)
    print(f"\n=== All Experiment Analyses Completed (Total time: {time.time() - overall_start_time:.2f} seconds) ===")
