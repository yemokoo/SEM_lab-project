# station.py
from charger import Charger
import numpy as np # np.mean, np.max 사용을 위해 추가

class Station:
    """
    충전소 클래스입니다. 충전소의 속성과 동작을 정의합니다.

    Attributes:
        station_id (int): 충전소 ID
        link_id (int): 충전소가 위치한 링크 ID
        num_of_chargers (int): 충전기 개수
        chargers (list): 충전기 객체 리스트
        waiting_trucks_queue (list): 충전 대기 중인 (트럭 객체, 대기열 진입 시간) 튜플 리스트
        queue_history (list): 매 시간 단계별 대기열 길이 기록
        waiting_times (list): 각 트럭의 실제 대기 시간(분) 기록
    """
    def __init__(self, station_id, link_id, num_of_chargers, charger_specs):
        """
        충전소 객체를 초기화합니다.

        Args:
            station_id (int): 충전소 ID
            link_id (int): 충전소가 위치한 링크 ID
            num_of_chargers (int): 충전기 개수
            charger_specs (list): 충전기 사양 리스트 (각 사양은 'power'와 'rate' 키를 가진 딕셔너리)
        """
        self.station_id = station_id
        self.link_id = int(link_id)  # 링크 ID를 정수형으로 변환
        self.num_of_chargers = num_of_chargers
        self.chargers = []  # 충전기 객체를 저장할 리스트 초기화
        
        # 각 충전소 내에서 고유한 charger_id를 부여하기 위한 카운터
        # Simulator 전체에서 고유 ID를 보장하려면 station_id와 조합하는 것이 좋음
        charger_id_counter = 1 
        for spec in charger_specs:
            # Charger 객체 생성 시 고유 ID 부여 (예: "stationID-chargerCounter")
            # Charger 클래스가 charger_id, power, station_id, rate를 인자로 받는다고 가정
            charger = Charger(
                charger_id=f"{self.station_id}-{charger_id_counter}", 
                power=spec['power'],
                station_id=self.station_id, # Charger가 자신이 속한 Station ID를 알도록 함
                rate=spec['rate']
            )
            self.chargers.append(charger)
            charger_id_counter += 1
        
        self.waiting_trucks_queue = []  # 충전 대기 중인 (트럭 객체, 대기열 진입 시간) 튜플을 저장할 리스트

        # 통계 수집을 위한 변수 초기화
        self.queue_history = []  # 매 시간 단계(또는 특정 이벤트 발생 시)의 대기열 길이를 저장
        self.waiting_times = []  # 충전을 시작한 각 트럭의 대기 시간을 저장 (분 단위)

    def add_truck_to_queue(self, truck, current_time): # current_time 인자 추가
        """
        충전 대기열에 트럭과 해당 트럭의 대기열 진입 시간을 추가합니다.
        Truck 클래스에서 이 메소드 호출 시 current_time을 전달해야 합니다.

        Args:
            truck (Truck): 충전 대기열에 추가할 트럭 객체
            current_time (float): 현재 시뮬레이션 시간 (트럭이 대기열에 진입하는 시간)
        """
        # 이미 대기열에 있는 트럭인지 확인 (중복 추가 방지)
        # waiting_trucks_queue는 (truck, entry_time) 튜플의 리스트이므로, truck 객체만 비교
        if truck not in [t[0] for t in self.waiting_trucks_queue]:
            self.waiting_trucks_queue.append((truck, current_time))  # 트럭 객체와 대기열 진입 시간(current_time)을 튜플로 저장
            truck.waiting = True  # 트럭의 waiting 속성을 True로 설정 (대기 중)

            # 대기열 정렬: 트럭의 다음 활성화 시간(도착/요청 시간으로 간주), 동일 시간일 경우 트럭 고유 ID 순
            # x[0]은 튜플의 첫 번째 요소인 truck 객체를 의미
            self.waiting_trucks_queue.sort(key=lambda x: (x[0].next_activation_time, x[0].unique_id))
            # print(f"시간 {current_time:.2f}: 트럭 {truck.unique_id}이(가) 충전소 {self.station_id} 대기열에 추가됨. 대기 시작 시간: {current_time:.2f}. 현재 대기열: {len(self.waiting_trucks_queue)}대")
        # else:
            # print(f"시간 {current_time:.2f}: 트럭 {truck.unique_id}은(는) 이미 충전소 {self.station_id} 대기열에 존재합니다.")


    def process_queue(self, current_time):
        """
        충전 대기열을 처리합니다. 대기 중인 트럭을 가능한 경우 충전기에 할당하고,
        이때 해당 트럭의 대기 시간을 계산하여 기록합니다.

        Args:
            current_time (float): 현재 시뮬레이션 시간
        """
        # 대기열에 트럭이 있고, 대기열의 첫 번째 트럭이 행동할 준비가 되었는지(next_activation_time) 확인
        if not self.waiting_trucks_queue or current_time < self.waiting_trucks_queue[0][0].next_activation_time:
            return  # 처리할 트럭이 없거나, 첫 번째 트럭이 아직 행동할 시간이 아님

        # truck_assigned_in_this_step = False # 이번 스텝에서 트럭이 할당되었는지 추적
        for charger in self.chargers:  # 사용 가능한 충전기 탐색
            if charger.current_truck is None and self.waiting_trucks_queue:  # 충전기가 비어있고 대기 트럭이 있는 경우
                # 대기열의 첫 번째 트럭이 활성화될 시간인지 다시 확인 (루프 중 시간은 흐르지 않지만 명시적)
                if current_time >= self.waiting_trucks_queue[0][0].next_activation_time:
                    truck_tuple = self.waiting_trucks_queue.pop(0)  # 대기열에서 첫 번째 (트럭, 진입시간) 튜플을 꺼냄
                    truck_to_charge = truck_tuple[0]
                    queue_entry_time = truck_tuple[1]

                    # 실제 대기 시간 계산 및 기록
                    actual_wait_time = current_time - queue_entry_time
                    if actual_wait_time < 0: # 혹시 모를 오류 방지 (current_time이 entry_time보다 작을 수 없음)
                        actual_wait_time = 0 
                    self.waiting_times.append(actual_wait_time)
                    
                    # print(f"시간 {current_time:.2f}: 트럭 {truck_to_charge.unique_id}이(가) 충전소 {self.station_id}의 충전기 {charger.charger_id}에 할당됨. 대기 시간: {actual_wait_time:.2f}분.")
                    
                    charger.start_charging(truck_to_charge, current_time)  # 충전 시작
                    # truck_to_charge.status='charging' # Truck의 상태는 Truck.step 또는 Charger.start_charging에서 관리하는 것이 좋음
                                                      # 여기서는 Charger가 트럭의 is_charging, waiting 등을 업데이트한다고 가정
                    # truck_assigned_in_this_step = True
                else:
                    # 대기열의 첫 번째 트럭이 아직 활성화될 시간이 아니므로 더 이상 진행하지 않음
                    break 
            
            if not self.waiting_trucks_queue: # 대기열에 더 이상 트럭이 없으면 루프 종료
                break
        
        # if truck_assigned_in_this_step:
            # print(f"시간 {current_time:.2f}: 충전소 {self.station_id} 대기열 처리 후 남은 대기 트럭: {len(self.waiting_trucks_queue)}대")


    def update_chargers(self, current_time):
        """
        모든 충전기의 상태를 업데이트하고 (예: 충전 완료 처리),
        매 시뮬레이션 스텝마다 현재 대기열의 길이를 기록합니다.

        Args:
            current_time (float): 현재 시뮬레이션 시간
        """
        for charger in self.chargers:  # 각 충전기에 대해
            truck = charger.current_truck
            if truck and truck.charge_end_time is not None and current_time >= truck.charge_end_time:
                # print(f"시간 {current_time:.2f}: 트럭 {truck.unique_id} 충전 완료 (충전소 {self.station_id}, 충전기 {charger.charger_id}).")
                charger.finish_charging()  # 충전 종료 처리 (Charger 내부에서 트럭 상태 변경 및 에너지 기록)
                truck.status = 'driving' # Truck의 상태는 Truck.step에서 관리하는 것이 좋음

        # 매 시뮬레이션 스텝(또는 Station이 업데이트되는 시점)의 대기열 길이를 기록
        self.queue_history.append(len(self.waiting_trucks_queue))
        # if current_time % 60 == 0 and current_time > 0: # 예: 매 시간마다 로깅 (디버깅용)
            # print(f"시간 {current_time:.2f}: 충전소 {self.station_id} 현재 대기열 길이: {len(self.waiting_trucks_queue)} (기록됨)")

