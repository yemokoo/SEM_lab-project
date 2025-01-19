# station.py
from charger import Charger

class Station:
    """
    충전소 클래스입니다. 충전소의 속성과 동작을 정의합니다.

    Attributes:
        station_id (int): 충전소 ID
        link_id (int): 충전소가 위치한 링크 ID
        num_of_chargers (int): 충전기 개수
        chargers (list): 충전기 객체 리스트
        waiting_trucks (list): 충전 대기 중인 트럭 객체 리스트
        total_charged_energy (float): 충전소에서 충전된 총 에너지 (kWh) (초기값: 0)

    Methods:
        __init__: 충전소 객체 초기화
        add_truck_to_queue: 충전 대기열에 트럭 추가
        process_queue: 충전 대기열 처리
        update_chargers: 충전기 상태 업데이트
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
        charger_id = 1  # 충전기 ID 초기값 설정
        for spec in charger_specs:  # charger_specs 리스트를 순회하며 각 충전기 생성
            charger = Charger(
                charger_id=charger_id,  # 충전기 ID
                power=spec['power'],  # 충전기 전력
                station_id=station_id,  # 충전소 ID
                rate=spec['rate']  # 1kW당 충전 비용
            )
            self.chargers.append(charger)  # 생성된 충전기 객체를 리스트에 추가
            charger_id += 1  # 다음 충전기 ID 증가
        self.waiting_trucks = []  # 충전 대기 중인 트럭을 저장할 리스트 초기화

    def add_truck_to_queue(self, truck):
        """
        충전 대기열에 트럭을 추가합니다.

        Args:
            truck (Truck): 충전 대기열에 추가할 트럭 객체
        """
        self.waiting_trucks.append(truck)  # 트럭을 대기열에 추가
        truck.waiting = True  # 트럭의 waiting 속성을 True로 설정 (대기 중)

        # 트럭 다음 활성 시간, 트럭 ID 순으로 정렬
        # next_activation_time이 빠른 트럭 우선, 동일 시간일 경우 unique_id가 작은 트럭 우선
        self.waiting_trucks.sort(key=lambda x: (x.next_activation_time, x.unique_id))
        #print(f"Truck {truck.unique_id} added to station {self.station_id} queue")

    def process_queue(self, current_time):
        """
        충전 대기열을 처리합니다. 대기 중인 트럭을 충전기에 할당합니다.

        Args:
            current_time (float): 현재 시뮬레이션 시간
        """
        if self.waiting_trucks:  # 대기열에 트럭이 있는 경우
            first_truck = self.waiting_trucks[0]  # 대기열의 맨 앞 트럭
            if current_time < first_truck.next_activation_time:  # 맨 앞 트럭의 활성화 시간이 아직 안 된 경우
                return  # 함수 종료 (아직 충전 시작하면 안 됨)

        for charger in self.chargers:  # 충전기 리스트를 순회
            if charger.current_truck is None and self.waiting_trucks:  # 충전기가 사용 가능하고 대기열에 트럭이 있는 경우
                truck = self.waiting_trucks.pop(0)  # 대기열에서 트럭을 꺼냄 (FIFO)
                charger.start_charging(truck, current_time)  # 충전 시작

    def update_chargers(self, current_time):
        """
        충전기 상태를 업데이트합니다. 충전이 완료된 트럭을 처리합니다.

        Args:
            current_time (float): 현재 시뮬레이션 시간
        """
        for charger in self.chargers:  # 충전기 리스트를 순회
            truck = charger.current_truck  # 충전기에서 현재 충전 중인 트럭 객체 가져오기
            if truck and truck.charge_end_time is not None and current_time >= truck.charge_end_time:  # 충전 중인 트럭이 있고, 충전 종료 시간이 설정되어 있고, 현재 시간이 충전 종료 시간보다 같거나 큰 경우
                charger.finish_charging()  # 충전 종료 처리