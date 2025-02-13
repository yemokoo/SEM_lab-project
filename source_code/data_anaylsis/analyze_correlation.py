import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ranksums
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF 계산


def calculate_effect_size(group1, group2, test_type='mannwhitneyu'):
    """
    Calculate the effect size for Mann-Whitney U test or Wilcoxon rank-sum test.

    Args:
        group1 (pd.Series): First group data.
        group2 (pd.Series): Second group data.
        test_type (str, optional):  Type of test ('mannwhitneyu' or 'wilcoxon').
                                    Defaults to 'mannwhitneyu'.

    Returns:
        float: Effect size (rank-biserial correlation).  Returns NaN if calculation is not possible.
    """
    # Remove NaN values before performing the test
    group1 = group1.dropna()
    group2 = group2.dropna()

    if group1.empty or group2.empty:
        return np.nan  # Return NaN if either group is empty after removing NaNs

    if test_type == 'mannwhitneyu':
        try:
            statistic, _ = mannwhitneyu(group1, group2, alternative='two-sided')
            n1 = len(group1)
            n2 = len(group2)
            # Rank-biserial correlation formula for Mann-Whitney U
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            return effect_size

        except ValueError:
            return np.nan  # Return NaN if all numbers are identical in mannwhitneyu

    elif test_type == 'wilcoxon':
        try:
            statistic, _ = ranksums(group1, group2)
            n1 = len(group1)
            n2 = len(group2)
            # Effect size (r) for Wilcoxon rank-sum test, Z / sqrt(N)
            z = statistic
            N = n1 + n2
            effect_size = z / np.sqrt(N)
            return effect_size

        except ValueError:
            return np.nan  # Return NaN if all numbers are identical in ranksums
    else:
        raise ValueError("Invalid test_type. Choose 'mannwhitneyu' or 'wilcoxon'.")



def analyze_candidate_criteria(candidate_detail_path, merged_data_path, output_file_path="analysis_results.txt"):
    """
    후보지 선정 조건이 수익성에 미치는 영향을 분석하고, 다중공선성 진단 및 해결을 추가합니다.

    Args:
        candidate_detail_path (str): candidate(detail).csv 파일 경로
        merged_data_path (str): merged_data.csv 파일 경로
        output_file_path (str, optional): 분석 결과를 저장할 TXT 파일 경로

    Returns:
        None
    """

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        try:
            candidate_df = pd.read_csv(candidate_detail_path)
            merged_df = pd.read_csv(merged_data_path)
        except FileNotFoundError as e:
            error_message = f"Error: File not found - {e}"
            print(error_message)
            output_file.write(error_message + "\n")
            return
        except pd.errors.EmptyDataError:
            error_message = "Error: One of the CSV files is empty."
            print(error_message)
            output_file.write(error_message + "\n")
            return
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(error_message)
            output_file.write(error_message + "\n")
            return

        if 'count' in candidate_df.columns:
            candidate_df = candidate_df.drop('count', axis=1)

        candidate_df['station_id'] = candidate_df.index
        merged_data = pd.merge(candidate_df, merged_df, on='station_id', how='left')
        merged_data = merged_data.fillna(0)

        target_variable = 'net_profit'
        independent_vars = ['OD', 'rest_area', 'interval', 'traffic', 'infra']

        output_file.write("\n--- Grouped Descriptive Statistics ---\n")
        print("\n--- Grouped Descriptive Statistics ---")
        for var in independent_vars:
            output_file.write(f"\n* Variable: {var}\n")
            print(f"\n* Variable: {var}")
            grouped_stats = merged_data.groupby(var)[target_variable].describe()
            print(grouped_stats)
            output_file.write(grouped_stats.to_string() + "\n")

        print("\n--- Box Plot Visualization ---")
        output_file.write("\n--- Box Plot Visualization ---\n")
        for var in independent_vars:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=var, y=target_variable, data=merged_data)
            title = f'{target_variable} vs. {var} (Condition Met vs. Not Met)'
            plt.title(title)
            plt.xlabel(f'{var} (0: Not Met, 1: Met)')
            plt.ylabel(target_variable)
            plot_filename = f"boxplot_{var}.png"
            plt.savefig(plot_filename)
            plt.close()
            output_file.write(f"Box plot for {title} saved as {plot_filename}.\n")

        output_file.write("\n--- Mann-Whitney U Test and Effect Size ---\n")
        print("\n--- Mann-Whitney U Test and Effect Size ---")
        for var in independent_vars:
            group0 = merged_data[merged_data[var] == 0][target_variable]
            group1 = merged_data[merged_data[var] == 1][target_variable]

            if not group0.empty and not group1.empty:
                statistic, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
                effect_size = calculate_effect_size(group0, group1, test_type='mannwhitneyu')

                output_file.write(f"\n* Variable: {var}\n")
                print(f"\n* Variable: {var}")
                output_file.write(f"  Mann-Whitney U statistic: {statistic:.4f}, p-value: {p_value:.4f}\n")
                print(f"  Mann-Whitney U statistic: {statistic:.4f}, p-value: {p_value:.4f}")

                if not np.isnan(effect_size):
                    output_file.write(f"  Effect Size (Rank-Biserial Correlation): {effect_size:.4f}\n")
                    print(f"  Effect Size (Rank-Biserial Correlation): {effect_size:.4f}")
                else:
                    output_file.write("  Effect Size: Not applicable (could not be calculated)\n")
                    print("  Effect Size: Not applicable (could not be calculated)")

                alpha = 0.05
                if p_value < alpha:
                    output_file.write(
                        f"  p-value ({p_value:.4f}) < significance level ({alpha:.2f}), **Reject Null Hypothesis**\n")
                    print(
                        f"  p-value ({p_value:.4f}) < significance level ({alpha:.2f}), **Reject Null Hypothesis**")
                    output_file.write(
                        f"  -> There is a **statistically significant difference** in median {target_variable} depending on whether the {var} condition is met.\n")
                    print(
                        f"  -> There is a **statistically significant difference** in median {target_variable} depending on whether the {var} condition is met.")
                else:
                    output_file.write(
                        f"  p-value ({p_value:.4f}) >= significance level ({alpha:.2f}), **Fail to Reject Null Hypothesis**\n")
                    print(
                        f"  p-value ({p_value:.4f}) >= significance level ({alpha:.2f}), **Fail to Reject Null Hypothesis**")
                    output_file.write(
                        f"  -> It is **difficult to conclude** that there is a statistically significant difference in median {target_variable} depending on whether the {var} condition is met.\n")
                    print(
                        f"  -> It is **difficult to conclude** that there is a statistically significant difference in median {target_variable} depending on whether the {var} condition is met.")
            else:
                output_file.write(f"\n* Variable: {var}\n")
                print(f"\n* Variable: {var}")
                output_file.write("  Mann-Whitney U Test: Not applicable (one or both groups are empty)\n")
                print("  Mann-Whitney U Test: Not applicable (one or both groups are empty)")

        # --- 다중공선성 진단 및 해결 ---
        output_file.write("\n--- Multicollinearity Diagnostics and Handling ---\n")
        print("\n--- Multicollinearity Diagnostics and Handling ---")

        # 1. 상관 행렬
        correlation_matrix = merged_data[independent_vars].corr()
        output_file.write("\n* Correlation Matrix:\n")
        output_file.write(correlation_matrix.to_string() + "\n")
        print("\n* Correlation Matrix:\n", correlation_matrix)

        # 상관 행렬 히트맵 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Independent Variables')
        corr_heatmap_filename = "correlation_heatmap.png"
        plt.savefig(corr_heatmap_filename)
        plt.close()
        output_file.write(f"Correlation heatmap saved as {corr_heatmap_filename}.\n")
        print(f"Correlation heatmap saved as {corr_heatmap_filename}.")


        # 2. VIF 계산
        X = merged_data[independent_vars].astype(float)  # VIF 계산을 위해 float로 변환
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        output_file.write("\n* Variance Inflation Factors (VIF):\n")
        output_file.write(vif_data.to_string() + "\n")
        print("\n* Variance Inflation Factors (VIF):\n", vif_data)

        # 3. 다중공선성 해결 (VIF가 높은 변수 제거 - 여기서는 예시로 'traffic' 제거)
        #   어떤 변수를 제거할지는 상관 행렬, VIF, 이론적 배경, 분석 목적 등을 종합적으로 고려
        vif_threshold = 5  # VIF 임계값 설정
        high_vif_vars = vif_data[vif_data['VIF'] > vif_threshold]['Variable'].tolist()

        if high_vif_vars:
            output_file.write(
                f"\n* Addressing Multicollinearity: Removing variables with VIF > {vif_threshold}.\n")
            print(f"\n* Addressing Multicollinearity: Removing variables with VIF > {vif_threshold}.")

            for var in high_vif_vars:
                if var in independent_vars:  # Ensure the variable is still in the list
                    independent_vars.remove(var)
                    output_file.write(f"  - Removed variable: {var}\n")
                    print(f"  - Removed variable: {var}")
        else:
             output_file.write("\n* No variables removed based on VIF threshold.\n")
             print("\n* No variables removed based on VIF threshold.")


        # --- 회귀 분석 (Linear Regression) - 다중공선성 처리 후 ---
        output_file.write("\n--- Regression Analysis (Linear Regression) - After Multicollinearity Handling ---\n")
        print("\n--- Regression Analysis (Linear Regression) - After Multicollinearity Handling ---")

        X = merged_data[independent_vars].astype(int)
        y = merged_data[target_variable]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()

        output_file.write(results.summary().as_text())
        print(results.summary())

        plt.figure(figsize=(8, 6))
        coef_df = pd.DataFrame({'Variable': results.params.index, 'Coefficient': results.params.values})
        coef_df = coef_df[coef_df['Variable'] != 'const']
        sns.barplot(x='Coefficient', y='Variable', data=coef_df, orient='h', palette='viridis')
        plt.title('Regression Coefficients')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Variable')
        coef_plot_filename = "regression_coefficients_after_mc.png"  # 파일 이름 변경
        plt.savefig(coef_plot_filename)
        plt.close()

        output_file.write(f"\nRegression coefficients plot (after multicollinearity handling) saved as {coef_plot_filename}.\n")
        print(f"\nRegression coefficients plot (after multicollinearity handling) saved as {coef_plot_filename}.")



        output_file.write("\n--- Interpretation ---\n")
        print("\n--- Interpretation ---")
        output_file.write(f"* Analyzed the difference in {target_variable} based on whether each selection condition ({independent_vars}) is met.\n")
        print(f"* Analyzed the difference in {target_variable} based on whether each selection condition ({independent_vars}) is met.")
        output_file.write(
            f"* Through group-specific descriptive statistics, box plots, Mann-Whitney U test results, effect sizes, and regression analysis,\n")
        print(
            f"* Through group-specific descriptive statistics, box plots, Mann-Whitney U test results, effect sizes, and regression analysis,")
        output_file.write(
            f"  you can understand the impact of each selection condition on profitability (net profit difference).\n")
        print(f"  you can understand the impact of each selection condition on profitability (net profit difference).")
        output_file.write(
            f"* In particular, if the p-value of the Mann-Whitney U test is less than the significance level (0.05),\n")
        print(f"* In particular, if the p-value of the Mann-Whitney U test is less than the significance level (0.05),")
        output_file.write(
            f"  it can be interpreted that whether the selection condition is met has a statistically significant effect on net profit.\n")
        print(
            f"  it can be interpreted that whether the selection condition is met has a statistically significant effect on net profit.")
        output_file.write(f"  The effect size (rank-biserial correlation) quantifies the magnitude of this difference.\n")
        print(f"  The effect size (rank-biserial correlation) quantifies the magnitude of this difference.")
        output_file.write(
            f"* Regression analysis provides insights into the relative importance of each condition in predicting net profit, after addressing multicollinearity.\n")
        print(
            f"* Regression analysis provides insights into the relative importance of each condition in predicting net profit, after addressing multicollinearity.")
        output_file.write(
            f"  The coefficients in the regression model indicate the change in net profit associated with each condition being met.\n")
        print(
            f"  The coefficients in the regression model indicate the change in net profit associated with each condition being met.")

        output_file.write(f"* Multicollinearity diagnostics (correlation matrix and VIF) were performed.\n")
        print(f"* Multicollinearity diagnostics (correlation matrix and VIF) were performed.")
        output_file.write(f"  Variables with high VIF (above a threshold) were removed to mitigate multicollinearity.\n")
        print(f"  Variables with high VIF (above a threshold) were removed to mitigate multicollinearity.")


        print(f"Analysis results are saved to '{output_file_path}' file.")
        output_file.write(f"Analysis results are saved to '{output_file_path}' file.\n")
# 파일 경로 설정 (실제 경로로 수정)
candidate_detail_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\candidate(detail).csv"  # 예시 경로
merged_data_path = r"C:\Users\wngud\Desktop\project\heavy_duty_truck_charging_infra\merged_data.csv"  # 예시 경로
output_file_path = "analysis_results.txt"

analyze_candidate_criteria(candidate_detail_path, merged_data_path, output_file_path)