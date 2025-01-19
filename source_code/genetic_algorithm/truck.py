# truck.py
import random
import pandas as pd

class Truck:
    """
    트럭 클래스입니다. MESA Agent 클래스를 상속받아 전기 트럭의 속성과 동작을 정의합니다.

    Attributes:
        unique_id (int): 트럭의 고유 ID (TRIP_ID)
        path_df (DataFrame): 트럭의 경로 정보를 담은 데이터프레임
        simulating_hours (int): 시뮬레이션 시간 (시간)
        BATTERY_CAPACITY (float): 배터리 용량 (kWh) (초기값: 400)
        SOC (float): 배터리 잔량 (%) (초기값: 50 ~ 90)
        CURRENT_LINK_ID (int): 트럭의 현재 링크 ID
        NEXT_LINK_ID (int): 트럭의 다음 링크 ID
        current_path_index (int): 현재 경로 인덱스 (초기값: 0)
        next_activation_time (float): 다음 활성화 시간 (분)
        is_charging (bool): 충전 중 여부 (초기값: False)
        waiting (bool): 충전 대기 중 여부 (초기값: False)
        wants_to_charge (bool): 충전 의사 여부 (초기값: False)
        charging_station_id (int): 충전 중인 충전소 ID
        charger_id (int): 충전 중인 충전기 ID
        charge_start_time (float): 충전 시작 시간
        charge_end_time (float): 충전 종료 시간
        charging_time (float): 충전 시간
        charge_cost (float): 충전 비용
        charging_history (list): 충전 이력 리스트
        links_to_move (int): 한 번에 이동할 링크 수

    Methods:
        __init__: 트럭 객체 초기화
        find_next_link_id: 다음 링크 ID 찾기
        move: 트럭 이동
        update_soc: SOC 업데이트
        step: 스텝 함수 (매 스텝마다 실행)
        get_info: 트럭 정보 반환
        stop: 트럭 정지
    """
    def __init__(self, path_df, simulating_hours, link_id_to_station, model, links_to_move=20):
        """
        트럭 객체를 초기화합니다.

        Args:
            path_df (DataFrame): 트럭의 경로 정보를 담은 데이터프레임
            simulating_hours (int): 시뮬레이션 시간 (시간)
            links_to_move (int): 한 번에 이동할 링크 수 (초기값: 30)
        """
        self.path_df = path_df.reset_index(drop=True)  # 경로 데이터프레임 인덱스 초기화
        self.simulating_hours = simulating_hours  # 시뮬레이션 시간 저장
        self.model = model

        # 배터리 용량 및 초기 SOC 설정
        self.BATTERY_CAPACITY = 450  # kWh
        self.SOC = random.randint(30, 90) # %

        self.unique_id = path_df['TRIP_ID'].iloc[0]  # 트럭 고유 ID 설정
        self.CURRENT_LINK_ID = path_df['LINK_ID'].iloc[0]  # 현재 링크 ID 설정
        self.NEXT_LINK_ID = None  # 다음 링크 ID 초기화
        self.current_path_index = 0  # 현재 경로 인덱스 초기화
        self.next_activation_time = self.path_df['START_TIME_MINUTES'].iloc[0]  # 다음 활성화 시간 설정 (분)

        # 충전 관련 변수 초기화
        self.is_charging = False  # 충전 중 여부
        self.waiting = False  # 충전 대기 중 여부
        self.wants_to_charge = False  # 충전 의사 여부
        self.charging_station_id = None  # 충전 중인 충전소 ID
        self.charger_id = None  # 충전 중인 충전기 ID
        self.charge_start_time = None  # 충전 시작 시간
        self.charge_end_time = None  # 충전 종료 시간
        self.charging_time = None  # 충전 시간
        self.charge_cost = None  # 충전 비용

        # 한 번에 이동할 링크 수 설정
        self.links_to_move = links_to_move

        self.link_id_to_station = link_id_to_station

        #print(f"Truck agent created with TRIP_ID: {self.unique_id}")

        self.find_next_link_id()  # 다음 링크 ID 찾기

    def find_next_link_id(self):
        """
        n개 이후의 링크 ID와 운전 시간(분)을 찾습니다.
        """
        target_index = self.current_path_index + self.links_to_move  # 이동할 링크 수만큼 인덱스 계산
        if target_index < len(self.path_df):  # 계산된 인덱스가 경로 데이터프레임 범위 내에 있는 경우
            next_row = self.path_df.iloc[target_index]  # 해당 인덱스의 행 가져오기
            self.NEXT_LINK_ID = next_row['LINK_ID']  # 다음 링크 ID 설정
        else:  # target_index가 데이터프레임 범위를 벗어나는 경우 (경로의 마지막 부분)
            target_index = len(self.path_df) - 1  # target_index를 마지막 인덱스로 설정
            next_row = self.path_df.iloc[target_index]  # 마지막 행 가져오기
            self.NEXT_LINK_ID = next_row['LINK_ID']  # 다음 링크 ID 설정

    def update_soc(self, energy_added):
            """
            트럭의 SOC를 업데이트하는 함수입니다.
            충전된 에너지량을 기반으로 SOC를 계산하고, 최대값을 100으로 제한합니다.
            SOC 업데이트 후 로그를 출력합니다.

            Args:
                energy_added (float): 충전된 에너지량 (kWh)
            """
            added_soc = (energy_added / self.BATTERY_CAPACITY) * 100  # 충전된 에너지량에 따른 SOC 증가량 계산
            self.SOC = min(100, self.SOC + added_soc)  # SOC 업데이트, 최대값 100 제한
           #print(f"Truck {self.unique_id} SOC updated to {self.SOC:.2f}%")

    def step(self, current_time):  # current_time 인자 추가
        """
        트럭의 이동 조건을 확인하고, 충전이 필요한 경우 충전소를 찾아 이동하거나,
        충전이 필요하지 않은 경우 목적지까지 이동합니다.
        """

        # 정지 조건 확인
        if self.SOC <= 0 or self.current_path_index >= len(self.path_df) - 1 or current_time >= self.simulating_hours * 60:  
            self.stop()  # 트럭 정지
            return  # 정지 조건에 해당하면 함수 종료

        # 이동 조건 확인 (시작 시간 이전, 다음 활성화 시간 이전, 충전 중, 충전 대기 중)
        if current_time < self.path_df['START_TIME_MINUTES'].iloc[0] or \
        current_time < self.next_activation_time or \
        self.is_charging or \
        self.waiting:
            return  # 조건에 해당하면 이동하지 않고 함수 종료

        # SOC가 60% 이하로 떨어지면 충전 의사 설정
        if self.SOC <= 60:
            self.wants_to_charge = True

        # 이동할 링크 수 설정 (남은 링크가 적으면 남은 만큼 이동)
        links_to_move = min(self.links_to_move, len(self.path_df) - self.current_path_index)

        # 다음 링크 ID와 운전 시간 업데이트
        self.find_next_link_id()

        # 충전이 필요한지 확인
        if self.wants_to_charge:
            # 현재 인덱스부터 이동하려는 인덱스 사이에 충전소가 있는지 확인
            target_index = self.current_path_index + links_to_move
            candidate_stations = list(map(lambda i: i if self.path_df['EVCS'].iloc[i] == 1 else None,
                                            range(self.current_path_index, min(target_index, len(self.path_df)))))
            # None 값 제거
            candidate_stations = [station for station in candidate_stations if station is not None]
            found_station = len(candidate_stations) > 0

            if found_station:  # 범위 내에 충전소가 있는 경우
                if self.SOC <= 30:  # 배터리가 30% 이하인 경우
                    # 가장 가까운 충전소 선택
                    nearest_station_index = min(candidate_stations)
                    i = nearest_station_index
                else:
                    # 각 충전소의 고속 충전기 개수를 저장하는 딕셔너리 생성
                    station_fc_counts = {}
                    for station_index in candidate_stations:
                        link_id = self.path_df['LINK_ID'].iloc[station_index] # dataframe의 인덱스 접근 방법
                        station = self.link_id_to_station.get(link_id)
                        fc_count = len([charger for charger in station.chargers if charger.power == 200])  # 고속 충전기 개수 계산
                        station_fc_counts[station_index] = fc_count

                    # 고속 충전기 개수가 가장 많은 충전소 선택
                    max_fc_count = max(station_fc_counts.values())
                    candidate_stations_with_max_fc = [station_index for station_index, fc_count in station_fc_counts.items() if
                                                        fc_count == max_fc_count]

                    # 고속 충전기 개수가 동일한 충전소가 여러 개인 경우 가장 가까운 충전소 선택
                    nearest_station_index = min(candidate_stations_with_max_fc)  # 인덱스 값을 직접 비교
                    i = nearest_station_index

                # 충전소까지 이동하는 데 걸리는 시간 및 에너지 소비량 계산
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    distance_to_station = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[i]
                else:
                    distance_to_station = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[i] - self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[
                        self.current_path_index]
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    time_to_station = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[i]
                else:
                    time_to_station = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[i] - \
                                    self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index]

                energy_consumed_to_station = (distance_to_station / 100) * 100  # 100km 당 100kWh 소비 가정

                # SOC 및 next_activation_time 업데이트
                self.SOC = max(0, self.SOC - (energy_consumed_to_station / self.BATTERY_CAPACITY) * 100)
                self.next_activation_time = current_time + time_to_station 

                # 충전소 링크 ID로 이동
                self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[i]
                self.current_path_index = i

                # 충전 시작
                station = self.link_id_to_station.get(self.CURRENT_LINK_ID)  # 링크 ID를 이용하여 충전소 객체 가져오기
                #print(f"Truck {self.unique_id} moved to charging station {station.station_id}")
                station.add_truck_to_queue(self)  # 충전소 대기열에 트럭 추가
                self.charging_station_id = station.station_id  # 트럭 객체에 충전소 ID 저장
                
            # 충전소를 찾지 못한 경우
            if not found_station:
                # 충전소를 찾지 못했으므로 원래 목적지까지 이동
                end_index = self.current_path_index + links_to_move  # 이동할 링크 수만큼 인덱스 계산
                if end_index < len(self.path_df):  # 계산된 인덱스가 경로 데이터프레임 범위 내에 있는 경우
                    total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[end_index] - \
                                            self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index]
                    total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[end_index] - \
                                        self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index]
                else:  # 마지막 링크까지 이동하는 경우
                    if self.current_path_index < 1:  # 초기 위치에서 이동 시
                        total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1]
                    else:
                        total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1] - \
                                                self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index]
                    if self.current_path_index < 1:  # 초기 위치에서 이동 시
                        total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[-1]
                    else:
                        total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[-1] - \
                                            self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index]
                    energy_consumed_to_station = (total_distance_traveled / 100) * 100  # 100km 당 78kWh 소비 가정

                total_energy_consumed = (total_distance_traveled / 100) * 100  # 100km 당 78kWh 소비 가정
                self.SOC = max(0, self.SOC - (total_energy_consumed / self.BATTERY_CAPACITY) * 100)  # SOC 업데이트
                self.CURRENT_LINK_ID = self.NEXT_LINK_ID  # 현재 링크 ID 업데이트
                self.current_path_index += links_to_move  # 현재 경로 인덱스 업데이트
                self.next_activation_time = current_time + total_driving_time  
                #print(f"Truck {self.unique_id} is moving on link {self.CURRENT_LINK_ID} (SOC: {self.SOC:.2f}%)")

        # 충전을 원하지 않거나 충전소가 없는 경우
        else:
            # 원래 목적지까지 이동
            end_index = self.current_path_index + links_to_move  # 이동할 링크 수만큼 인덱스 계산
            if end_index < len(self.path_df):  # 계산된 인덱스가 경로 데이터프레임 범위 내에 있는 경우
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[end_index]
                else:
                    total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[end_index] - \
                                            self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index]
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[end_index]
                else:
                    total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[end_index] - \
                                        self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index]
            else:  # 마지막 링크까지 이동하는 경우
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1]
                else:
                    total_distance_traveled = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1] - \
                                            self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[self.current_path_index]
                if self.current_path_index < 1:  # 초기 위치에서 이동 시
                    total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[-1]
                else:
                    total_driving_time = self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[-1] - \
                                        self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[self.current_path_index]

            total_energy_consumed = (total_distance_traveled / 100) * 100  # 100km 당 100kWh 소비 가정
            self.SOC = max(0, self.SOC - (total_energy_consumed / self.BATTERY_CAPACITY) * 100)  # SOC 업데이트
            self.CURRENT_LINK_ID = self.NEXT_LINK_ID  # 현재 링크 ID 업데이트
            self.current_path_index += links_to_move  # 현재 경로 인덱스 업데이트
            self.next_activation_time = current_time + total_driving_time  
            #print(f"Truck {self.unique_id} is moving on link {self.CURRENT_LINK_ID} (SOC: {self.SOC:.2f}%)")

    def get_info(self):
        """
        트럭의 정보를 DataFrame 형태로 반환합니다.
        """

        # 총 이동 거리 계산
        total_distance = self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[-1]

        # DataFrame 생성
        info_df = pd.DataFrame([{
            'truck_id': self.unique_id,
            'final_SOC': self.SOC,
            'destination_reached': self.current_path_index >= len(self.path_df) - 1,
            'stopped_due_to_low_battery': self.SOC <= 0,
            'total_distance': total_distance
        }])

        return info_df

    def stop(self):
        """
        트럭 정지 시 동작을 정의합니다.
        get_info() 함수를 이용하여 현재 상태 정보를 DataFrame으로 수집하고, 
        시뮬레이터의 결과 DataFrame에 추가합니다.
        """

        # 트럭 정보 DataFrame 가져오기
        info_df = self.get_info()
        """
        if self.SOC <= 0:
            print(f"Truck {self.unique_id} stopped due to low battery (SOC: {self.SOC}) on link {self.CURRENT_LINK_ID}")  # 배터리 부족으로 정지
        elif self.current_path_index >= len(self.path_df) - 1:
            print(f"Truck {self.unique_id} reached its destination!")  # 목적지 도착
        elif self.model.current_time >= self.simulating_hours * 60:  # 시뮬레이션 종료 조건
            print(f"Truck {self.unique_id} stopped due to simulation end on link {self.CURRENT_LINK_ID}")  # 시뮬레이션 종료로 정지
        """
        # 시뮬레이터의 결과 DataFrame에 추가
        if self.model.truck_results_df is None:  # truck_results_df가 None인 경우 초기화
            self.model.truck_results_df = pd.DataFrame(columns=info_df.columns)
        self.model.truck_results_df = pd.concat([self.model.truck_results_df, info_df], ignore_index=True)  # truck_results_df에 할당

        self.model.remove_truck(self)

        return True