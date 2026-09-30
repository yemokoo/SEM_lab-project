import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/juhyeong/Desktop/화물차/Data/Processed_Data/GA_results/simulation_metrics_all.csv')

summary = df.drop(columns=['시드']).groupby('전동화율').agg(['mean', 'std'])

rate_order = {'2%': 1, '5%': 2, '10%': 3, '15%': 4, '20%': 5}
summary['order'] = summary.index.map(rate_order)
summary = summary.sort_values('order').drop('order', axis=1)

latex = []
latex.append("\\begin{table*}[ht!]")
latex.append("\\caption[Planning and Operation Performance for Varying Electrification Rate]{%")
latex.append("{Planning and Operation Performance for Varying Electrification Rate}}")
latex.append("\\centering")
latex.append("\\resizebox{\\textwidth}{!}{%")
latex.append("\\begin{tabular}{ccccccc}")
latex.append("\\toprule")
latex.append("\\makecell[l]{\\textbf{Scenario$^*$}} & ")
latex.append("\\makecell{\\textbf{Number of} \\\\ \\textbf{charging} \\\\ \\textbf{stations} \\\\ \\textbf{(Units)}} & ")
latex.append("\\makecell{\\textbf{Number of} \\\\ \\textbf{chargers} \\\\ \\textbf{per station} \\\\ \\textbf{(Average)}} & ")
latex.append("\\makecell{\\textbf{Number of} \\\\ \\textbf{chargers} \\\\ \\textbf{per station} \\\\ \\textbf{(Max)}} & ")
latex.append("\\makecell{\\textbf{Average} \\\\ \\textbf{queuing} \\\\ \\textbf{time at} \\\\ \\textbf{charger (min)}} & ")
latex.append("\\makecell{\\textbf{Average} \\\\ \\textbf{utilization} \\\\ \\textbf{rate (\\%)}} & ")
latex.append("\\makecell{\\textbf{Objective} \\\\ \\textbf{function} \\\\ \\textbf{value} \\\\ \\textbf{(Mil. KRW)}}\\\\")
latex.append("\\midrule")

prev_stations = None
prev_chargers = None

for rate in ['2%', '5%', '10%', '15%', '20%']:
    if rate not in summary.index:
        continue
    row = summary.loc[rate]
    
    curr_stations = row[('충전소 수', 'mean')]
    curr_chargers = row[('충전소당 충전기 수', 'mean')]
    
    s_stations = f"{curr_stations:.0f} $\\pm$ {row[('충전소 수', 'std')]:.1f}"
    s_avg_chargers = f"{curr_chargers:.2f} $\\pm$ {row[('충전소당 충전기 수', 'std')]:.2f}"
    s_max_chargers = f"{row[('최대 충전기 수', 'mean')]:.0f} $\\pm$ {row[('최대 충전기 수', 'std')]:.1f}"
    s_queue = f"{row[('평균 대기시간(min)', 'mean')]:.2f} $\\pm$ {row[('평균 대기시간(min)', 'std')]:.2f}"
    s_util = f"{row[('평균 가동률(%)', 'mean')]:.2f} $\\pm$ {row[('평균 가동률(%)', 'std')]:.2f}"
    s_obj = f"{row[('목적함수(Mil. KRW)', 'mean')]:.2f} $\\pm$ {row[('목적함수(Mil. KRW)', 'std')]:.2f}"
    
    latex.append(f"\\textbf{{{rate}}} & {s_stations} & {s_avg_chargers} & {s_max_chargers} & {s_queue} & {s_util} & {s_obj} \\\\")
    
    if prev_stations is not None:
        diff_stations = ((curr_stations - prev_stations) / prev_stations) * 100
        diff_chargers = ((curr_chargers - prev_chargers) / prev_chargers) * 100
        latex.append(f" & ({diff_stations:+.1f}\\%) & ({diff_chargers:+.1f}\\%) & & & & \\\\[1ex]")
    
    prev_stations = curr_stations
    prev_chargers = curr_chargers

latex.append("\\bottomrule")
latex.append("\\multicolumn{7}{l}{$^*$Electrification Rate}")
latex.append("\\end{tabular}}")
latex.append("\\label{tab:sensitive_result_table}")
latex.append("\\end{table*}")

print('\n'.join(latex))
