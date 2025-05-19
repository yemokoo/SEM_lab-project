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
    Represents an electric truck in the simulation.
    Trucks move along a predefined path, manage their battery (SOC), 
    decide when to charge, and stop at designated locations.
    Charging aims for 100% SOC without time restrictions imposed by the truck itself.
    """
    def __init__(self, path_df, simulating_hours, link_id_to_station, model, links_to_move=40):
        # Internal helper functions for randomized initial SOC and charge decision threshold
        def get_starting_soc_random():
            initial_soc_ranges = [
                (30, 40, 0.01), (40, 50, 0.02), (50, 60, 0.03),
                (60, 70, 0.04), (70, 80, 0.05), (80, 90, 0.07),
                (90, 100, 0.78),
            ]
            rand_val = random.random()
            cumulative_prob = 0
            for lower, upper, prob in initial_soc_ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    return random.randint(lower, upper)
            return random.randint(90,100)

        def get_charge_decide_soc():
            charge_decide_soc_ranges = [
                (10, 20, 0.05), (20, 30, 0.13), (30, 40, 0.18),
                (40, 50, 0.19), (50, 60, 0.16), (60, 70, 0.13),
                (70, 80, 0.10), (80, 90, 0.05), (90, 100, 0.02),
            ]
            rand_val = random.random()
            cumulative_prob = 0
            for lower, upper, prob in charge_decide_soc_ranges:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    return random.randint(lower, upper)
            return random.randint(40,50)

        self.path_df = path_df.reset_index(drop=True)
        self.simulating_hours = simulating_hours
        self.model = model # Reference to the main simulation model
        self.BATTERY_CAPACITY = 540  # kWh
        
        self.SOC = float(get_starting_soc_random()) # Initial State of Charge
        self.unique_id = self.path_df['OBU_ID'].iloc[0]
        self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[0] # Current link ID on the path
        self.next_activation_time = float(self.path_df['START_TIME_MINUTES'].iloc[0]) # Next time this truck should be processed
        self.current_path_index = 0 # Current index in its path_df

        # Charging related states
        self.is_charging = False # True if currently connected to a charger and charging
        self.waiting = False     # True if in a station's queue
        self.wants_to_charge = False # True if truck has decided it needs to charge soon

        # Information about the charger/station if charging/waiting
        self.charging_station_id = None
        self.charger_id = None
        self.charge_start_time = None # Simulation time when charging started
        self.charge_end_time = None   # Simulation time when charging is expected to end (set by Charger)
        self.charging_time = None     # Duration of the charge (set by Charger)
        self.charge_cost = None       # Cost of the charge (set by Charger)
        self.pending_charge_energy = 0.0 # Energy to be added to SOC upon charge completion (set by Charger)
        
        # SOC threshold at which the truck actively starts looking for a charger
        self.charge_decide = min(float(get_charge_decide_soc()), 80) 
        
        self.links_to_move = links_to_move # Max number of path links to traverse in one driving segment
        self.link_id_to_station = link_id_to_station # Mapping of link IDs to Station objects
        self.unit_minutes = self.model.unit_minutes # Simulation time step unit
        
        # Identifying significant stops in the path (e.g., mandatory breaks)
        min_stop_duration_for_significant = self.model.unit_minutes 
        self.significant_stop_indices = self.path_df[self.path_df['STOPPING_TIME'] >= min_stop_duration_for_significant].index.tolist()
        self.significant_stop_indices_set = set(self.significant_stop_indices)
        
        # Identifying all EV charging station locations on the truck's path
        if 'EVCS' in self.path_df.columns:
            self.all_evcs_indices = self.path_df[self.path_df['EVCS'] == 1].index.tolist()
        else:
            self.all_evcs_indices = []
            
        self.status = 'inactive' # Initial status; becomes 'driving', 'stopping', 'waiting_queue', or 'stopped'
        self.stop_end_time = None # If stopping, when the stop is scheduled to end
        self.just_finished_stopping = False # Flag to manage behavior immediately after a stop
        self.actual_stop_events = [] # Records details of actual stops made

    def update_soc(self, energy_change_kwh):
        """
        Updates the truck's State of Charge (SOC) based on energy change.
        Energy change is positive for charging, negative for consumption.
        SOC is capped between 0% and 100%.
        """
        # previous_soc = self.SOC # Kept for potential future debugging, not printed now
        delta_soc = (energy_change_kwh / self.BATTERY_CAPACITY) * 100.0
        self.SOC += delta_soc
        self.SOC = max(0.0, min(100.0, self.SOC)) 
        # print(f"[TRUCK {self.unique_id} SOC_UPDATE]: 에너지: {energy_change_kwh:.2f}kWh, 이전SOC: {previous_soc:.2f}%, 현재SOC: {self.SOC:.2f}%") # Removed

    def step(self, current_time):
        """
        Processes the truck's actions for the current simulation time step.
        This includes moving, stopping, deciding to charge, and interacting with stations.
        """
        current_time = float(current_time)

        is_last_index = self.current_path_index >= len(self.path_df) - 1
        is_time_over = current_time >= (self.simulating_hours * 60.0)

        # Handle simulation end time
        if is_time_over:
            if self.status != 'stopped':
                self.stop()
            return

        # 1. Depot (Final Destination) Processing
        if is_last_index and not self.is_charging and not self.waiting:
            if self.status == 'stopped': 
                return # Already stopped and processed

            current_row = self.path_df.iloc[self.current_path_index]
            has_charger_here = current_row['EVCS'] == 1
            needs_charge = self.SOC < 100.0 # Charge if SOC is not exactly 100%
            
            if has_charger_here and needs_charge: 
                self.wants_to_charge = True # Intend to charge if needed and possible
            
            # Attempt to queue for charging if conditions met
            if has_charger_here and self.wants_to_charge and self.status not in ['charging', 'waiting_queue', 'stopped']:
                station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                if station:
                    station.add_truck_to_queue(self, current_time)
                else: # No station object found at this link
                    self.stop()
                return 
            else: # No charger, or no need/intent to charge, or already handled
                self.stop()
                return

        # 2. Check if it's time for this truck to act
        if current_time < self.next_activation_time:
            return

        if self.status == 'stopped': 
            return # Do not process trucks that are already stopped

        # 3. Process truck based on its current status
        if self.status == 'inactive' and self.next_activation_time <= current_time: # First time activation
            self.status = 'driving'

        if self.waiting: # If in a station's queue
            if self.status != 'waiting_queue':
                self.status = 'waiting_queue'
            return # Actions are handled by the Station while waiting

        if self.status == 'stopping': # If currently at a scheduled stop
            if self.stop_end_time is not None and current_time >= self.stop_end_time:
                self.status = 'driving' # Stop duration ended, resume driving
                self.stop_end_time = None
                self.next_activation_time = current_time # Evaluate next move immediately
                self.just_finished_stopping = True 
            else: 
                return # Continue stopping

        # Handle SOC depletion
        if self.SOC <= 0.001 and not self.is_charging : 
            if self.status != 'stopped':
                self.stop() # Stop if out of battery
            return

        # 4. Driving Logic
        if self.status == 'driving':
            if self.current_path_index >= len(self.path_df): # Should not happen if depot logic is correct
                if self.status != 'stopped':
                    self.stop()
                return
            
            current_row = self.path_df.iloc[self.current_path_index]
            is_at_significant_stop_location = self.current_path_index in self.significant_stop_indices_set

            # 4a. Arrival at a Significant Stop
            if is_at_significant_stop_location and not self.just_finished_stopping:
                self.status = 'stopping' # Change status to 'stopping'
                stopping_time_here = float(current_row['STOPPING_TIME'])
                self.stop_end_time = current_time + stopping_time_here
                self.next_activation_time = self.stop_end_time # Truck must wait out the stop duration
                
                self.actual_stop_events.append({
                    'index': self.current_path_index,
                    'cumulative_length': float(current_row['CUMULATIVE_LINK_LENGTH']),
                    'stopping_time': stopping_time_here
                })

                has_charger_here = current_row['EVCS'] == 1
                needs_charge = self.SOC < 100.0 # Check if charging is needed (SOC < 100%)

                if has_charger_here and needs_charge:
                    station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                    if station:
                        self.wants_to_charge = True 
                        station.add_truck_to_queue(self, current_time) # Attempt to queue for charging
                else: 
                    self.wants_to_charge = False # No charger or SOC is full

                self.just_finished_stopping = False 
                return # Current action is to start stopping or queueing

            if self.just_finished_stopping: # Reset flag after processing the first step post-stop
                self.just_finished_stopping = False

            # 4b. Decide to look for charging based on SOC threshold
            if self.SOC <= self.charge_decide and not self.wants_to_charge:
                self.wants_to_charge = True # Set intent to charge if SOC is low

            if self.current_path_index >= len(self.path_df) - 1: # Defensive check for end of path
                 if self.status != 'stopped':
                    self.stop()
                 return

            # 4c. Plan Movement and Find Charging Station if needed
            max_links_for_this_step = min(self.links_to_move, len(self.path_df) - 1 - self.current_path_index)
            if max_links_for_this_step <= 0: 
                 if self.status != 'stopped':
                    self.stop()
                 return
            
            potential_end_index_default_move = self.current_path_index + max_links_for_this_step
            actual_end_path_index = potential_end_index_default_move 
            move_to_charge_station = False # Flag if the destination is a charging station

            if self.wants_to_charge:
                # Simplified station search: 
                # 1. Check upcoming significant stops with chargers.
                # 2. If none, check any upcoming EVCS locations.
                chosen_station_idx = -1
                upcoming_significant_stops_with_charger = [
                    idx for idx in self.significant_stop_indices
                    if self.current_path_index < idx <= potential_end_index_default_move and \
                       self.path_df.iloc[idx]['EVCS'] == 1
                ]
                if upcoming_significant_stops_with_charger:
                    chosen_station_idx = min(upcoming_significant_stops_with_charger)
                else:
                    candidate_opportunistic_stations_indices = [
                        idx for idx in self.all_evcs_indices
                        if self.current_path_index < idx <= potential_end_index_default_move
                    ]
                    if candidate_opportunistic_stations_indices:
                        chosen_station_idx = min(candidate_opportunistic_stations_indices) 
                
                if chosen_station_idx != -1:
                    actual_end_path_index = chosen_station_idx
                    move_to_charge_station = True
            
            # --- Movement Calculation ---
            start_cum_dist_km = 0.0
            start_cum_time_min = 0.0
            if self.current_path_index > 0 : 
                prev_idx = self.current_path_index -1
                start_cum_dist_km = float(self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[prev_idx])
                start_cum_time_min = float(self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[prev_idx])
            
            actual_end_path_index = min(actual_end_path_index, len(self.path_df) - 1)
            if actual_end_path_index < self.current_path_index: # Should not happen if logic is correct
                actual_end_path_index = self.current_path_index # Prevent moving backward

            end_cum_dist_km = float(self.path_df['CUMULATIVE_LINK_LENGTH'].iloc[actual_end_path_index])
            end_cum_time_min = float(self.path_df['CUMULATIVE_DRIVING_TIME_MINUTES'].iloc[actual_end_path_index])

            distance_traveled_segment_km = end_cum_dist_km - start_cum_dist_km
            driving_time_segment_min = end_cum_time_min - start_cum_time_min

            if distance_traveled_segment_km < 0.0: distance_traveled_segment_km = 0.0
            if driving_time_segment_min < 0.001: # Prevent zero driving time to avoid loops
                driving_time_segment_min = 0.01 
            # --- Movement Calculation End ---

            energy_consumed_kwh = (distance_traveled_segment_km / 100.0) * 180.0 # Energy consumption formula
            self.update_soc(-energy_consumed_kwh) 
            
            self.current_path_index = actual_end_path_index
            self.CURRENT_LINK_ID = self.path_df['LINK_ID'].iloc[self.current_path_index]
            self.next_activation_time = current_time + driving_time_segment_min

            # 4d. Arrival at an Opportunistic (non-significant stop) Charging Station
            # Check if the destination was a charging station AND it's not a significant stop where charging is handled by 4a.
            is_now_at_significant_stop_after_move = self.current_path_index in self.significant_stop_indices_set
            
            if move_to_charge_station and not is_now_at_significant_stop_after_move : 
                station = self.link_id_to_station.get(self.CURRENT_LINK_ID)
                if station:
                    if self.SOC < 100.0: # Only queue if SOC is less than 100%
                        self.wants_to_charge = True 
                        station.add_truck_to_queue(self, current_time) 
                    else: 
                        self.wants_to_charge = False # SOC is full, no need to queue
            return # End of driving step
            
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
