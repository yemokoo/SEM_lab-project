# charger.py
class Charger:
    """
    충전기 클래스입니다. 충전기의 속성과 동작을 정의합니다.

    Attributes:
        charger_id (int): 충전기 ID
        power (float): 충전기 전력 (kW)
        station_id (int): 충전소 ID
        rate (float): 1kWh당 충전 비용
        current_truck (Truck): 현재 충전 중인 트럭 객체 (초기값: None)
        total_charged_energy (float): 충전기가 충전한 총 에너지 (kWh) (초기값: 0)

    Methods:
        __init__: 충전기 객체 초기화
        start_charging: 트럭 충전 시작
        finish_charging: 트럭 충전 종료
    """
    def __init__(self, charger_id, power, station_id, rate):
        """
        충전기 객체를 초기화합니다.

        Args:
            charger_id (int): 충전기 ID
            power (float): 충전기 전력 (kWh)
            station_id (int): 충전소 ID
            rate (float): 1kWh당 충전 비용
        """
        self.charger_id = charger_id
        self.power = power  # hkW
        self.station_id = station_id
        self.rate = rate  # 1kWh 당 비용
        self.current_truck = None  # 현재 충전 중인 트럭 객체
        self.total_charged_energy = 0.0  # 이 충전기가 충전한 총 에너지 (kWh)
        self.charging_events_count = 0   # 이 충전기가 처리한 총 충전 이벤트 수
        self.last_charge_finish_time = 0.0 # 마지막 충전 종료 시간 
        
    def start_charging(self, truck, current_time):
        """
        트럭 충전을 시작합니다.

        Args:
            truck (Truck): 충전할 트럭 객체
            current_time (float): 현재 시뮬레이션 시간
        """
        self.current_truck = truck
        truck.waiting = False  # 충전 대기 상태 해제
        truck.is_charging = True  # 충전 중 상태 설정
        truck.charging_station_id = self.station_id  # 트럭 객체에 충전소 ID 저장
        truck.charger_id = self.charger_id  # 트럭 객체에 충전기 ID 저장
        truck.charge_start_time = current_time  # 트럭 객체에 충전 시작 시간 저장

        remaining_energy = ((100 - truck.SOC) / 100) * truck.BATTERY_CAPACITY  # 남은 충전 필요 에너지 (kWh) 계산
        charging_time = (remaining_energy / self.power) * 60  # 충전 시간 (분) 계산
        charge_cost = remaining_energy * self.rate  # 충전 비용 계산

        truck.charge_end_time = current_time + charging_time  # 트럭 객체에 충전 종료 시간 저장
        truck.charging_time = charging_time  # 트럭 객체에 충전 시간 저장
        truck.charge_cost = charge_cost  # 트럭 객체에 충전 비용 저장
        truck.next_activation_time = truck.charge_end_time  # 트럭 객체의 다음 활성화 시간을 충전 종료 시간으로 설정

        # 충전 시작 시 충전량 추가
        self.total_charged_energy += remaining_energy  # 충전기가 충전한 총 에너지 업데이트
        self.charging_events_count += 1

    def finish_charging(self):
        """
        트럭 충전을 종료합니다.
        """
        truck = self.current_truck  # 현재 충전 중인 트럭 객체 가져오기
        if truck:  # 트럭 객체가 존재하는 경우
            if truck.charge_end_time is not None:
                self.last_charge_finish_time = truck.charge_end_time
                
            remaining_energy = ((100 - truck.SOC) / 100) * truck.BATTERY_CAPACITY  # 남은 에너지 계산 (kWh) - SOC 업데이트를 위해
            truck.update_soc(remaining_energy)  # 트럭의 SOC 업데이트 (완전히 충전된 상태로 설정)
            truck.is_charging = False  # 충전 중 상태 해제
            truck.wants_to_charge = False  # 충전 의사 해제
            truck.charging_station_id = None  # 트럭 객체의 충전소 ID 초기화
            truck.charger_id = None  # 트럭 객체의 충전기 ID 초기화
            truck.charge_start_time = None  # 트럭 객체의 충전 시작 시간 초기화
            truck.charge_end_time = None  # 트럭 객체의 충전 종료 시간 초기화
            self.current_truck = None  # 충전기의 현재 트럭 정보 초기화