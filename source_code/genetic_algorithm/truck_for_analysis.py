import pandas as pd
import numpy as np
import random

class Truck:
    """
    상세한 운행 상태 및 SOC 이력을 기록하는 전기 화물차 에이전트 클래스.
    계층적 의사결정 로직(1순위: 중요 정차, 2순위: 충전)을 따릅니다.
    """
    def __init__(self, path_df, simulating_hours, link_id_to_station, model, links_to_move=40):
        # --- 내부 헬퍼 함수 ---
        def get_starting_soc_random():
            """사전 정의된 분포에 따라 시작 SOC를 랜덤하게 반환합니다."""
            ranges = [(30, 40, 0.01), (40, 50, 0.02), (50, 60, 0.03), (60, 70, 0.04), 
                      (70, 80, 0.05), (80, 90, 0.07), (90, 100, 0.78)]
            rand_val, cumulative_prob = random.random(), 0
            for lower, upper, prob in ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob: return random.randint(lower, upper)
            return random.randint(90, 100)

        def get_charge_decide_soc():
            """사전 정의된 분포에 따라 충전 결정 SOC를 랜덤하게 반환합니다."""
            ranges = [(10, 20, 0.05), (20, 30, 0.13), (30, 40, 0.18), (40, 50, 0.19),
                      (50, 60, 0.16), (60, 70, 0.13), (70, 80, 0.10), (80, 90, 0.05), (90, 100, 0.02)]
            rand_val, cumulative_prob = random.random(), 0
            for lower, upper, prob in ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob: return random.randint(lower, upper)
            return random.randint(40, 50)

        # --- 기본 속성 및 상태 변수 초기화 ---
        self.path_df = path_df.reset_index(drop=True)
        self.simulating_hours = simulating_hours
        self.model = model
        self.BATTERY_CAPACITY = 540  # kWh
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
        
        self.significant_stop_indices_set = set(self.path_df[self.path_df['STOPPING_TIME'] >= self.unit_minutes].index)
        self.all_evcs_indices = self.path_df[self.path_df['EVCS'] == 1].index.tolist() if 'EVCS' in self.path_df.columns else []
        
        self.status = 'inactive'  # 'inactive', 'driving', 'stopping', 'stopped'
        self.stop_end_time = None
        self.just_finished_stopping = False
        
        # 상태 기록 및 충전 타입 변수
        self.history = []
        self.charging_type = 'none'  # 'none', 'opportunity_charging', 'enroute_charging'
        self.total_charged_energy_kwh = 0.0
        self.total_charging_events = 0

    def get_current_detailed_status(self):
        """세분화된 현재 상태(충전 유형 포함)를 문자열로 반환합니다."""
        if self.is_charging:
            return self.charging_type
        if self.waiting:
            return 'waiting_for_charge'
        return self.status

    def record_state(self, current_time):
        """현재 트럭의 상태 (시간, SOC, 상태, 위치)를 기록합니다."""
        self.history.append({
            'time': current_time,
            'soc': self.SOC,
            'status': self.get_current_detailed_status(),
            'link_id': self.CURRENT_LINK_ID
        })

    def update_soc(self, energy_change_kwh):
        """SOC를 업데이트하고 0%와 100% 사이로 유지합니다."""
        delta_soc = (energy_change_kwh / self.BATTERY_CAPACITY) * 100.0
        self.SOC = max(0.0, min(100.0, self.SOC + delta_soc))

    def step(self, current_time):
        """시뮬레이션의 한 단계를 진행합니다. 운전, 정차, 충전 결정을 포함합니다."""
        # 시뮬레이션 종료 조건 확인
        if self.current_path_index >= len(self.path_df) - 1 or current_time >= (self.simulating_hours * 60.0):
            if self.status != 'stopped': self.stop()
            return
        
        # 비활성 상태이거나 충전/대기 중일 경우 행동하지 않음
        if self.status == 'stopped' or current_time < self.next_activation_time:
            return
        if self.status == 'inactive': self.status = 'driving'
        if self.waiting or self.is_charging: return

        # 정차 상태 처리
        if self.status == 'stopping':
            if current_time >= self.stop_end_time:
                self.status = 'driving'
                self.stop_end_time = None
                self.next_activation_time = current_time
                self.just_finished_stopping = True
            else:
                return

        # 배터리 방전 시 운행 중단
        if self.SOC <= 0.001:
            self.stop(); return

        if self.status == 'driving':
            current_row = self.path_df.iloc[self.current_path_index]
            is_at_significant_stop_location = self.current_path_index in self.significant_stop_indices_set

            # --- 1순위 의사결정: 계획된 주요 정차 ---
            if is_at_significant_stop_location and not self.just_finished_stopping:
                self.status = 'stopping'
                self.stop_end_time = current_time + float(current_row['STOPPING_TIME'])
                self.next_activation_time = self.stop_end_time
                # 정차지에 충전소가 있다면 '기회 충전' 시도
                if current_row['EVCS'] == 1 and self.SOC < 100.0:
                    station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                    if station:
                        self.wants_to_charge = True
                        self.charging_type = 'opportunity_charging'
                        station.add_truck_to_queue(self, current_time)
                return

            if self.just_finished_stopping: self.just_finished_stopping = False
            
            # 충전 필요 여부 판단
            if self.SOC <= self.charge_decide and not self.wants_to_charge:
                self.wants_to_charge = True
            
            # --- 이동 계획 수립 ---
            max_links_to_move = len(self.path_df) - 1 - self.current_path_index
            potential_end_index = self.current_path_index + min(self.links_to_move, max_links_to_move)
            if potential_end_index <= self.current_path_index: self.stop(); return
            
            actual_end_path_index = potential_end_index
            upcoming_stops = [idx for idx in self.significant_stop_indices_set if self.current_path_index < idx <= potential_end_index]
            
            # 다음 목적지 설정 (정차지 우선)
            if upcoming_stops:
                actual_end_path_index = min(upcoming_stops)
            else:
                # --- 2순위 의사결정: 충전소 탐색 ---
                if not self.wants_to_charge:
                    # 선제적 충전 필요성 검토
                    start_dist = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index -1 if self.current_path_index > 0 else 0]
                    end_dist = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[potential_end_index]
                    energy_pred = ((end_dist - start_dist) / 100.0) * 180.0
                    if self.SOC - (energy_pred / self.BATTERY_CAPACITY * 100) < 15.0:
                        self.wants_to_charge = True

                if self.wants_to_charge:
                    candidate_indices = [idx for idx in self.all_evcs_indices if self.current_path_index < idx <= potential_end_index]
                    chosen_station_idx = -1
                    if self.SOC < 15.0 and candidate_indices: # 긴급 상황: 가장 가까운 충전소로
                        chosen_station_idx = min(candidate_indices)
                    elif candidate_indices: # 일반 상황: 충전기 많고, 덜 붐비고, 가까운 곳으로
                        valid_candidates = []
                        for idx in candidate_indices:
                            s_obj = self.link_id_to_station.get(self.path_df.iloc[idx]['LINK_ID'])
                            if s_obj and len(s_obj.waiting_trucks_queue) < s_obj.num_of_chargers:
                                valid_candidates.append({'path_index': idx, 'num_chargers': s_obj.num_of_chargers})
                        if valid_candidates:
                            best = sorted(valid_candidates, key=lambda x: (-x['num_chargers'], x['path_index']))[0]
                            chosen_station_idx = best['path_index']
                    
                    if chosen_station_idx != -1:
                        actual_end_path_index = chosen_station_idx
                        self.charging_type = 'enroute_charging'

            # --- 최종 이동 실행 ---
            start_idx = self.current_path_index
            end_idx = min(actual_end_path_index, len(self.path_df) - 1)
            if end_idx <= start_idx: end_idx = start_idx + 1
            if end_idx >= len(self.path_df): self.stop(); return

            start_dist = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[start_idx -1 if start_idx > 0 else 0]
            dist_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[end_idx] - start_dist
            
            start_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[start_idx -1 if start_idx > 0 else 0]
            drive_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[end_idx] - start_time
            
            self.update_soc(-(dist_traveled / 100.0) * 180.0)
            self.current_path_index = end_idx
            self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[self.current_path_index]
            self.next_activation_time = current_time + drive_time
            
            # '경로상 충전' 목적지 도착 시 대기열 추가
            is_charging_destination = self.charging_type == 'enroute_charging' and self.path_df.iloc[end_idx]['EVCS'] == 1
            if is_charging_destination:
                station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                if station and self.SOC < 100.0:
                    station.add_truck_to_queue(self, self.next_activation_time)
            
      
    def get_info(self):
        """
        Collects final information about the truck when the simulation ends for this truck.
        """
        actual_traveled_distance_km = 0.0
        if hasattr(self, 'path_df') and self.path_df is not None and not self.path_df.empty:
            capped_index = min(max(int(self.current_path_index), 0), len(self.path_df) - 1)
            actual_traveled_distance_km = float(self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[capped_index])

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
            'threshold_SOC': self.charge_decide,
            'destination_reached': destination_reached_flag,
            'stopped_due_to_low_battery': stopped_low_battery_flag,
            'stopped_due_to_simulation_end': stopped_sim_end_flag, 
            'total_distance_planned': total_distance_planned_km,
            'actual_traveled_distance_km': actual_traveled_distance_km,
            'traveled_distance_at_last_stop85': traveled_distance_at_last_stop85_km,
            'total_charged_energy_kwh': self.total_charged_energy_kwh,
            'total_charging_events': self.total_charging_events,
            'starting_time': self.path_df['START_TIME_MINUTES'].iloc[0],
            'actual_reached_time': self.model.current_time if hasattr(self.model, 'current_time') else None,
            'final_path_index': self.current_path_index,
            'final_status': self.status
        }])
        return info_df

    def stop(self):
        """트럭 운행을 중지하고, 최종 데이터를 기록 및 전달한 후 메모리에서 정리합니다."""
        if self.status != 'stopped':
            self.status = 'stopped'
            
            if self.model:
                self.record_state(self.model.current_time) # 마지막 상태 기록

            final_info_df = self.get_info()
            if self.model:
                # 시뮬레이터에 시각화 데이터 및 최종 결과 전달
                if hasattr(self.model, 'add_completed_truck_data'):
                    self.model.add_completed_truck_data(self.unique_id, self.history, self.path_df)
                
                if not hasattr(self.model, 'truck_results_df') or self.model.truck_results_df is None:
                    self.model.truck_results_df = pd.DataFrame()
                self.model.truck_results_df = pd.concat([self.model.truck_results_df, final_info_df], ignore_index=True)

            # 메모리 관리를 위해 내부 데이터 정리
            del self.path_df
            del self.history
            self.path_df = None
            self.history = []

            # 시뮬레이터에서 트럭 객체 제거 요청
            if self.model and hasattr(self.model, 'remove_truck'):
                self.model.remove_truck(self)
