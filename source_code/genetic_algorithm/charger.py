# charger.py (수정)

class Charger:
    """
    충전기 클래스입니다. 충전기의 속성과 동작을 정의합니다.
    """
    def __init__(self, charger_id, power, station_id, rate):
        """
        충전기 객체를 초기화합니다.
        """
        self.charger_id = charger_id
        self.power = power
        self.station_id = station_id
        self.rate = rate
        self.current_truck = None
        self.total_charged_energy = 0.0
        self.charging_events_count = 0
        self.last_charge_finish_time = 0.0
        # ❗[수정] 총 누적 충전 시간 변수 초기화
        self.total_charging_duration_minutes = 0.0

        # --- [신규] 개별 이벤트 저장을 위한 리스트 ---
        self.individual_durations = []
        self.individual_energies = []
        # --- [신규] ---

    def start_charging(self, truck, current_time):
        """
        트럭 충전을 시작합니다.
        """
        self.current_truck = truck
        truck.waiting = False
        truck.is_charging = True
        truck.charging_station_id = self.station_id
        truck.charger_id = self.charger_id
        truck.charge_start_time = current_time

        remaining_energy = ((100 - truck.SOC) / 100) * truck.BATTERY_CAPACITY
        charging_time = (remaining_energy / self.power) * 60
        charge_cost = remaining_energy * self.rate

        truck.charge_end_time = current_time + charging_time
        truck.charging_time = charging_time
        truck.charge_cost = charge_cost
        truck.next_activation_time = truck.charge_end_time

        self.total_charged_energy += remaining_energy
        self.charging_events_count += 1
        truck.total_charged_energy_kwh = getattr(truck, 'total_charged_energy_kwh', 0.0) + remaining_energy
        truck.total_charging_events = getattr(truck, 'total_charging_events', 0) + 1
        
        # ❗[수정] 계산된 충전 시간을 총 누적 충전 시간에 더함
        self.total_charging_duration_minutes += charging_time

        # --- [신규] 계산된 개별 이벤트 값을 리스트에 추가 ---
        self.individual_durations.append(charging_time)
        self.individual_energies.append(remaining_energy)
        # --- [신규] ---

    def finish_charging(self):
        """
        트럭 충전을 종료합니다.
        """
        truck = self.current_truck
        if truck:
            if truck.charge_end_time is not None:
                self.last_charge_finish_time = truck.charge_end_time
                
            remaining_energy = ((100 - truck.SOC) / 100) * truck.BATTERY_CAPACITY
            truck.update_soc(remaining_energy) 
            truck.is_charging = False
            truck.wants_to_charge = False
            truck.charging_station_id = None
            truck.charger_id = None
            truck.charge_start_time = None
            truck.charge_end_time = None
            self.current_truck = None
