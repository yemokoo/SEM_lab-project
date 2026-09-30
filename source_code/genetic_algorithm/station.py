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
    def __init__(self, station_id, link_id, num_of_chargers, charger_specs, unit_minutes):
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
        self.unit_minutes = unit_minutes # 시뮬레이션 단위 시간 (분 단위)
        
        # 각 충전소 내에서 고유한 charger_id를 부여하기 위한 카운터
        charger_id_counter = 1 
        for spec in charger_specs:
            charger = Charger(
                charger_id=f"{self.station_id}-{charger_id_counter}", 
                power=spec['power'],
                station_id=self.station_id,
                rate=spec['rate']
            )
            self.chargers.append(charger)
            charger_id_counter += 1
        
        self.waiting_trucks_queue = []  # 충전 대기 중인 (트럭 객체, 대기열 진입 시간) 튜플을 저장할 리스트

        # 통계 수집을 위한 변수 초기화
        self.queue_history = []
        self.charging_history = []
        self.power_history = []
        self.waiting_times = []  # 충전을 시작한 각 트럭의 대기 시간을 저장 (분 단위)
        self.total_arrivals = 0
        self.total_departures = 0
        self.counted_physical_arrivals = set()
        
        # history 기록용 리스트 (그래프의 Y축 데이터)
        self.cumulative_arrivals_history = [0]
        self.cumulative_departures_history = [0]

    def add_truck_to_queue(self, truck, current_time):
        """
        충전 대기열에 트럭과 해당 트럭의 대기열 진입 시간을 추가합니다.
        """
        if not truck.waiting: # waiting 플래그로 O(1) 중복 체크 (대기열 순회 제거)
            self.waiting_trucks_queue.append((truck, truck.next_activation_time))
            truck.waiting = True
            self.waiting_trucks_queue.sort(key=lambda x: (x[0].next_activation_time, x[0].unique_id))


    def process_queue(self, current_time):
        """
        충전 대기열을 처리합니다. 대기 중인 트럭을 가능한 경우 충전기에 할당하고,
        이때 해당 트럭의 대기 시간을 계산하여 기록합니다.
        """
        # 대기열에 트럭이 있고, 첫 번째 트럭이 행동할 준비가 되었는지 확인
        if not self.waiting_trucks_queue or current_time < self.waiting_trucks_queue[0][0].next_activation_time:
            return

        for charger in self.chargers:
            if charger.current_truck is None and self.waiting_trucks_queue:
                if current_time >= self.waiting_trucks_queue[0][0].next_activation_time:
                    
                    # 대기열에서 (트럭, 진입 시간) 튜플을 꺼냄
                    truck_to_charge, queue_entry_time = self.waiting_trucks_queue.pop(0)
                    
                    charger_free_time = charger.last_charge_finish_time
                    
                    # 실제 충전 시작 시간 = max(대기열 진입 시간, 충전기 가용 시간)
                    charge_start_time = max(queue_entry_time, charger_free_time)

                    # 트럭 기준 실제 대기 시간
                    actual_wait_time = charge_start_time - queue_entry_time
                    
                    self.waiting_times.append(actual_wait_time)
                    self.total_departures += 1
                    
                    # 충전 시작
                    charger.start_charging(truck_to_charge, current_time)
                else:
                    break 
            
            if not self.waiting_trucks_queue:
                break


    def update_chargers(self, current_time):
        """
        모든 충전기의 상태를 업데이트합니다 (충전 완료 처리).
        GA 평가에 불필요한 history 기록은 스킵하여 속도를 높입니다.
        """
        for charger in self.chargers:
            truck = charger.current_truck
            if truck and truck.charge_end_time is not None and current_time >= truck.charge_end_time:
                charger.finish_charging()
                truck.status = 'driving'


    def finalize_unprocessed_trucks(self, final_time):
        """
        시뮬레이션 종료 시, 대기열에 처리되지 않고 남아있는 트럭들의 대기 시간을 계산하여 기록합니다.
        """
        while self.waiting_trucks_queue:
            truck, queue_entry_time = self.waiting_trucks_queue.pop(0)
            wait_time = max(0, final_time - queue_entry_time)
            self.waiting_times.append(wait_time)
