import random
import pandas as pd
import warnings
import numpy as np
import gc
import time

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
    category=FutureWarning
)

class Truck:
    """
    계층적 의사결정 로직을 포함한 최종 버전의 전기 화물차 에이전트 클래스.
    1순위: 중요 정차, 2순위: 충전소 선택의 명확한 우선순위를 따릅니다.
    """
    def __init__(self, path_df, simulating_hours, link_id_to_station, model, links_to_move=40):
        # --- 내부 헬퍼 함수 (기존과 동일) ---
        def get_starting_soc_random():
            initial_soc_ranges = [(30, 40, 0.01), (40, 50, 0.02), (50, 60, 0.03), (60, 70, 0.04), (70, 80, 0.05), (80, 90, 0.07), (90, 100, 0.78)]
            rand_val = random.random()
            cumulative_prob = 0
            for lower, upper, prob in initial_soc_ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob: return random.randint(lower, upper)
            return random.randint(90, 100)

        def get_charge_decide_soc():
            charge_decide_soc_ranges = [(10, 20, 0.05), (20, 30, 0.13), (30, 40, 0.18), (40, 50, 0.19), (50, 60, 0.16), (60, 70, 0.13), (70, 80, 0.10), (80, 90, 0.05), (90, 100, 0.02)]
            rand_val = random.random()
            cumulative_prob = 0
            for lower, upper, prob in charge_decide_soc_ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob: return random.randint(lower, upper)
            return random.randint(40, 50)

        # --- 기본 속성 및 상태 변수 (기존과 동일) ---
        self.path_df = path_df.reset_index(drop=True)
        self.simulating_hours = simulating_hours
        self.model = model
        self.BATTERY_CAPACITY = 540
        self.SOC = float(get_starting_soc_random())
        self.unique_id = self.path_df['OBU_ID'].iloc[0]
        self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[0]
        self.next_activation_time = float(self.path_df['START_TIME_MINUTES'].iloc[0])
        self.current_path_index = 0
        self.is_charging = False
        self.waiting = False
        self.wants_to_charge = False
        self.charge_decide = min(float(get_charge_decide_soc()), 80)
        self.links_to_move = links_to_move
        self.link_id_to_station = link_id_to_station
        self.unit_minutes = self.model.unit_minutes
        min_stop_duration_for_significant = self.model.unit_minutes
        self.significant_stop_indices_set = set(self.path_df[self.path_df['STOPPING_TIME'] >= min_stop_duration_for_significant].index)
        if 'EVCS' in self.path_df.columns:
            self.all_evcs_indices = self.path_df[self.path_df['EVCS'] == 1].index.tolist()
        else:
            self.all_evcs_indices = []
        self.status = 'inactive'
        self.stop_end_time = None
        self.just_finished_stopping = False
        self.actual_stop_events = []

    def update_soc(self, energy_change_kwh):
        delta_soc = (energy_change_kwh / self.BATTERY_CAPACITY) * 100.0
        self.SOC += delta_soc
        self.SOC = max(0.0, min(100.0, self.SOC))

    def step(self, current_time):
        current_time = float(current_time)
        if self.current_path_index >= len(self.path_df) - 1 or current_time >= (self.simulating_hours * 60.0):
            if self.status != 'stopped': self.stop()
            return
        if self.status == 'stopped' or current_time < self.next_activation_time:
            return

        if self.status == 'inactive': self.status = 'driving'
        if self.waiting or self.is_charging: return

        if self.status == 'stopping':
            if current_time >= self.stop_end_time:
                self.status = 'driving'
                self.stop_end_time = None
                self.next_activation_time = current_time
                self.just_finished_stopping = True
            else:
                return

        if self.SOC <= 0.001:
            self.stop(); return

        if self.status == 'driving':
            current_row = self.path_df.iloc[self.current_path_index]
            is_at_significant_stop_location = self.current_path_index in self.significant_stop_indices_set

            if is_at_significant_stop_location and not self.just_finished_stopping:
                self.status = 'stopping'
                stopping_time_here = float(current_row['STOPPING_TIME'])
                self.stop_end_time = current_time + stopping_time_here
                self.next_activation_time = self.stop_end_time
                if current_row['EVCS'] == 1 and self.SOC < 100.0:
                    station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                    if station:
                        self.wants_to_charge = True
                        station.add_truck_to_queue(self, current_time)
                return

            if self.just_finished_stopping: self.just_finished_stopping = False
            if self.SOC <= self.charge_decide and not self.wants_to_charge: self.wants_to_charge = True

            ## [수정 및 추가] 계층적 의사결정 로직
            max_links_for_this_step = min(self.links_to_move, len(self.path_df) - 1 - self.current_path_index)
            if max_links_for_this_step <= 0: self.stop(); return
            
            potential_end_index = self.current_path_index + max_links_for_this_step
            
            # 1순위: 중요 정차 지점 탐색
            upcoming_stops = [idx for idx in self.significant_stop_indices_set if self.current_path_index < idx <= potential_end_index]
            
            move_to_charge_station = False
            
            if upcoming_stops:
                # 1순위인 중요 정차 지점이 있으면, 그곳을 목적지로 설정하고 다른 탐색은 하지 않음
                actual_end_path_index = min(upcoming_stops)
            else:
                # 중요 정차 지점이 없을 경우
                actual_end_path_index = potential_end_index # 기본 목적지로 초기화
                
                # [추가] 선제적 충전 탐색 로직
                # 만약 기본 목적지까지 이동했을 때 SOC가 15% 미만으로 예상된다면, 충전 의사를 강제로 활성화
                if not self.wants_to_charge:
                    # 예상 이동 거리 및 에너지 소비 계산
                    start_dist_pred = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index - 1] if self.current_path_index > 0 else 0
                    end_dist_pred = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[actual_end_path_index]
                    dist_pred = max(0, end_dist_pred - start_dist_pred)
                    energy_pred = (dist_pred / 100.0) * 180.0
                    
                    # 예상 SOC 계산
                    projected_soc = self.SOC - (energy_pred / self.BATTERY_CAPACITY) * 100
                    
                    if projected_soc < 15.0:
                        self.wants_to_charge = True # 충전 필요 상태로 전환
                
                # 2순위: 충전소 탐색
                if self.wants_to_charge:
                    chosen_station_idx = -1
                    candidate_indices = sorted(list(set(
                        [idx for idx in self.all_evcs_indices if self.current_path_index < idx <= potential_end_index]
                    )))

                    # [수정] 긴급/일반 충전 로직 분리
                    if self.SOC < 15.0:
                        # 긴급 상황: 무조건 가장 가까운 충전소로 이동 (혼잡도, 충전기 수 무시)
                        if candidate_indices:
                            # path_index가 가장 작은, 즉 가장 가까운 충전소를 선택
                            chosen_station_idx = min(candidate_indices)
                    else:
                        # 일반 상황: 혼잡도, 충전기 수, 거리를 종합적으로 고려
                        valid_candidates = []
                        for station_path_idx in candidate_indices:
                            station_link_id = self.path_df.iloc[station_path_idx]['LINK_ID']
                            station_obj = self.link_id_to_station.get(station_link_id)
                            if not station_obj: continue

                            # 혼잡하지 않은 충전소만 후보로 추가
                            is_congested = len(station_obj.waiting_trucks_queue) >= station_obj.num_of_chargers
                            if is_congested: continue
                            
                            valid_candidates.append({'path_index': station_path_idx, 'num_chargers': station_obj.num_of_chargers})

                        if valid_candidates:
                            # 1순위: 충전기 많은 순, 2순위: 가까운 순
                            best_candidate = sorted(valid_candidates, key=lambda x: (-x['num_chargers'], x['path_index']))[0]
                            chosen_station_idx = best_candidate['path_index']
                    
                    if chosen_station_idx != -1:
                        actual_end_path_index = chosen_station_idx
                        move_to_charge_station = True
            
            # 최종 이동 실행 (이하 로직은 기존과 거의 동일)
            actual_end_path_index = min(actual_end_path_index, len(self.path_df) - 1)
            if actual_end_path_index <= self.current_path_index:
                 actual_end_path_index = self.current_path_index + 1
            if actual_end_path_index >= len(self.path_df): self.stop(); return

            start_cum_dist = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index - 1] if self.current_path_index > 0 else 0
            end_cum_dist = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[actual_end_path_index]
            distance_traveled = max(0, end_cum_dist - start_cum_dist)

            start_cum_drive_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index - 1] if self.current_path_index > 0 else 0
            end_cum_drive_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[actual_end_path_index]
            driving_time_segment = max(0, end_cum_drive_time - start_cum_drive_time)

            stopping_time_segment = self.path_df['STOPPING_TIME'].iloc[self.current_path_index:actual_end_path_index].sum()
            
            energy_consumed = (distance_traveled / 100.0) * 180.0
            self.update_soc(-energy_consumed)
            
            self.current_path_index = actual_end_path_index
            self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[self.current_path_index]
            self.next_activation_time = current_time + driving_time_segment + stopping_time_segment

            if move_to_charge_station and not (self.current_path_index in self.significant_stop_indices_set):
                station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                if station:
                    if self.SOC < 100.0:
                        self.wants_to_charge = True
                        station.add_truck_to_queue(self, self.next_activation_time)
            return
            
      
    def get_info(self):
        """
        Collects final information about the truck when the simulation ends for this truck.
        """
        traveled_distance_at_last_stop85_km = 0.0
        if hasattr(self, 'actual_stop_events') and self.actual_stop_events: 
            stops_85_min = [event for event in self.actual_stop_events if event['stopping_time'] >= 85]
            if stops_85_min:
                last_stop_85_event = stops_85_min[-1]
                traveled_distance_at_last_stop85_km = last_stop_85_event['cumulative_length']
        
        total_distance_planned_km = 0.0
        if hasattr(self, 'path_df') and self.path_df is not None and not self.path_df.empty: 
                total_distance_planned_km = float(self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1])

        destination_reached_flag = False
        if hasattr(self, 'path_df') and self.path_df is not None and not self.path_df.empty:
            destination_reached_flag = (self.current_path_index >= len(self.path_df) - 1) and \
                                     (self.status == 'stopped') and \
                                     not self.is_charging and not self.waiting
        
        stopped_low_battery_flag = (self.SOC <= 0.01) and not destination_reached_flag
        
        is_sim_time_over_at_stop = False
        if hasattr(self.model, 'current_time') and hasattr(self, 'simulating_hours'): 
            is_sim_time_over_at_stop = (self.model.current_time >= (self.simulating_hours * 60.0))
        
        stopped_sim_end_flag = is_sim_time_over_at_stop and \
                               not destination_reached_flag and \
                               not stopped_low_battery_flag and \
                               (self.status == 'stopped')

        info_df = pd.DataFrame([{
            'truck_id': self.unique_id,
            'final_SOC': self.SOC,
            'destination_reached': destination_reached_flag,
            'stopped_due_to_low_battery': stopped_low_battery_flag,
            'stopped_due_to_simulation_end': stopped_sim_end_flag, 
            'total_distance_planned': total_distance_planned_km,
            'traveled_distance_at_last_stop85': traveled_distance_at_last_stop85_km,
            'starting_time': self.path_df['START_TIME_MINUTES'].iloc[0],
            'actual_reached_time': self.model.current_time if hasattr(self.model, 'current_time') else None,
            'final_path_index': self.current_path_index,
            'final_status': self.status
        }])
        return info_df

    def stop(self):
        """
        Stops the truck, records its final information, and requests its removal from the simulation.
        """
        if self.status != 'stopped': # Prevent multiple stop calls for the same truck
            self.status = 'stopped'

            # Record final information
            final_info_df = self.get_info() 
            # print(f"[TRUCK {self.unique_id} STOP_FINAL_INFO @ {getattr(self.model, 'current_time', 'N/A'):.2f}]: \n{final_info_df.to_string()}") # For debugging if needed

            # Add results to the main model's DataFrame
            try:
                if not hasattr(self.model, 'truck_results_df') or self.model.truck_results_df is None:
                    self.model.truck_results_df = pd.DataFrame() 

                current_cols = self.model.truck_results_df.columns
                if not current_cols.empty:
                    info_df_reordered = final_info_df.reindex(columns=current_cols).fillna(np.nan)
                    for col in final_info_df.columns: 
                        if col not in info_df_reordered.columns: 
                            info_df_reordered[col] = final_info_df[col]
                else: 
                    info_df_reordered = final_info_df.copy()

                self.model.truck_results_df = pd.concat([self.model.truck_results_df, info_df_reordered], ignore_index=True, axis=0)
            except Exception as e:
                # Fallback in case of error during DataFrame concatenation
                try:
                    if not hasattr(self.model, 'truck_results_df') or self.model.truck_results_df is None:
                         self.model.truck_results_df = pd.DataFrame()
                    self.model.truck_results_df = pd.concat([self.model.truck_results_df, final_info_df], ignore_index=True, sort=False) 
                except Exception: # nested_e_safe:
                    pass # Log if necessary, but prevent crash
            
            # Clean up truck's internal data to free memory
            if hasattr(self, 'path_df'): 
                del self.path_df
                self.path_df = None
            if hasattr(self, 'actual_stop_events'): 
                del self.actual_stop_events
                self.actual_stop_events = []
            
            # Request removal from the main simulation model
            if self.model and hasattr(self.model, 'remove_truck'):
                self.model.remove_truck(self) 
        
        return True
