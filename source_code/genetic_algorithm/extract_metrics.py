import os
import glob
import pandas as pd
import numpy as np
import sys
import random
import re

sys.path.append('/home/juhyeong/Desktop/화물차/Code/SEM_lab-project/source_code/genetic_algorithm')
from Simulator_for_day import Simulator, load_car_path_df, load_station_df

base_dir = '/home/juhyeong/Desktop/화물차/Data/Processed_Data/GA_results'
rates = ['2%', '5%', '10%', '15%', '20%']

car_paths_folder = r"/home/juhyeong/Desktop/화물차/Data/Processed_Data/simulator/Trajectory(DAY_90km)"
station_file_path = r"/home/juhyeong/Desktop/화물차/Data/Processed_Data/simulator/Final_Candidates_Selected.csv"

# The user explicitly set 10% to 3050.
# So the base number is 3050. The multiplier is based on rate.
def get_num_trucks(rate_str):
    rate_val = float(rate_str.replace('%', '')) / 10.0 # 10% -> 1.0, 20% -> 2.0
    return int(3050 * rate_val)

def extract_gen(filepath):
    match = re.search(r'g(\d+)\.csv', filepath)
    return int(match.group(1)) if match else -1

results = []

for rate in rates:
    num_trucks = get_num_trucks(rate)
    # GA uses TOTAL_CHARGERS = 10000 for penalty calculation
    num_chargers = 10000 
    
    dirs = glob.glob(os.path.join(base_dir, f"{rate} *"))
    for d in dirs:
        try:
            seed_str = d.split('=')[-1].replace(')','')
            seed_val = int(seed_str)
        except:
            continue
            
        best_csvs = glob.glob(os.path.join(d, 'best_individual', 'at_g*.csv'))
        if not best_csvs:
            best_csvs = glob.glob(os.path.join(d, 'best_individual_overall_final_g*.csv'))
            
        if not best_csvs:
            continue
            
        best_csv = sorted(best_csvs, key=extract_gen)[-1]
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        car_paths_df = load_car_path_df(car_paths_folder, num_trucks, estimated_areas=33)
        
        df_best = pd.read_csv(best_csv)
        counts = df_best.iloc[0, :-1].astype(int).values
        
        station_df = load_station_df(station_file_path)
        station_df['num_of_charger'] = counts
        
        # 5 min unit, 30 hours, 3 truck_step_frequency
        sim = Simulator(car_paths_df, station_df, 5, 30, num_trucks, num_chargers, 3)
        sim.prepare_simulation()
        sim.run_simulation()
        
        of_value = sim.analyze_results()
        
        total_chargers = sum(counts)
        total_stations = sum(counts > 0)
        max_chargers_per_station = max(counts)
        
        avg_chargers_per_station = total_chargers / total_stations if total_stations > 0 else 0
        
        # avg waiting time
        st_res = sim.station_results_df
        active_st = st_res[st_res['num_of_charger'] > 0]
        avg_waiting = active_st['avg_waiting_time_min'].mean() if not active_st.empty else 0
        
        # util rate
        tot_energy = st_res['total_charged_energy_kWh'].sum()
        avg_util = (tot_energy / (total_chargers * 200 * 24)) * 100 if total_chargers > 0 else 0
        
        results.append({
            '전동화율': rate,
            '시드': seed_val,
            '충전소 수': total_stations,
            '충전소당 충전기 수': avg_chargers_per_station,
            '최대 충전기 수': max_chargers_per_station,
            '전체 충전기 개수': total_chargers,
            '평균 대기시간(min)': avg_waiting,
            '평균 가동률(%)': avg_util,
            '목적함수(Mil. KRW)': of_value / 1000000.0
        })

df_results = pd.DataFrame(results)

# Filter logic requested by user
final_dfs = []

for rate in ['2%', '5%', '20%']:
    final_dfs.append(df_results[df_results['전동화율'] == rate])

df_10 = df_results[df_results['전동화율'] == '10%'].sort_values(by=['충전소 수', '시드']).head(5)
final_dfs.append(df_10)

df_15 = df_results[(df_results['전동화율'] == '15%') & (df_results['시드'].isin([42, 43, 45, 46, 47]))]
final_dfs.append(df_15)

df_final = pd.concat(final_dfs)

# Sorting for display
rate_order = {'2%': 1, '5%': 2, '10%': 3, '15%': 4, '20%': 5}
df_final['rate_order'] = df_final['전동화율'].map(rate_order)
df_final = df_final.sort_values(by=['rate_order', '시드']).drop('rate_order', axis=1)

df_final.to_csv(os.path.join(base_dir, 'simulation_metrics_all.csv'), index=False, encoding='utf-8-sig')
print("Extracted and filtered successfully")
