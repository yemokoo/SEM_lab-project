from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_BASE_FOLDER = Path(
    r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\result_for_kori"
)
INPUT_CSV_PATH = RESULTS_BASE_FOLDER / "penetration_rate_vehicle_station_summary_seed42.csv"
OUTPUT_FOLDER = RESULTS_BASE_FOLDER / "report_figures_seed42"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV_PATH).sort_values("penetration_rate_percent").reset_index(drop=True)

    charged_share = pd.to_numeric(df["charged_vehicle_share_mean_percent"], errors="coerce") / 100.0
    charged_share = charged_share.replace(0, np.nan)
    charged_share_sd = pd.to_numeric(df["charged_vehicle_share_sd_percent"], errors="coerce") / 100.0

    charging_events_mean = pd.to_numeric(df["charging_events_per_vehicle_mean"], errors="coerce")
    charging_events_sd = pd.to_numeric(df["charging_events_per_vehicle_sd"], errors="coerce")
    charging_energy_mean = pd.to_numeric(df["charging_energy_per_vehicle_mean_kWh"], errors="coerce")
    charging_energy_sd = pd.to_numeric(df["charging_energy_per_vehicle_sd_kWh"], errors="coerce")

    df["charging_events_per_charged_vehicle_mean"] = charging_events_mean / charged_share
    df["charging_energy_per_charged_vehicle_mean_kWh"] = charging_energy_mean / charged_share

    # Delta-method approximation for ratio-derived error bars.
    df["charging_events_per_charged_vehicle_sd"] = np.sqrt(
        (charging_events_sd / charged_share) ** 2
        + ((charging_events_mean * charged_share_sd) / (charged_share ** 2)) ** 2
    )
    df["charging_energy_per_charged_vehicle_sd_kWh"] = np.sqrt(
        (charging_energy_sd / charged_share) ** 2
        + ((charging_energy_mean * charged_share_sd) / (charged_share ** 2)) ** 2
    )

    return df


def _x_positions_and_labels(df: pd.DataFrame):
    x_positions = np.arange(len(df))
    x_labels = [f"{int(v)}%" for v in df["penetration_rate_percent"]]
    return x_positions, x_labels


def create_total_demand_figure(df: pd.DataFrame) -> Path:
    x_positions, x_labels = _x_positions_and_labels(df)
    mean_mwh = pd.to_numeric(df["total_energy_demand_mean_kWh_per_day"], errors="coerce") / 1000.0
    sd_mwh = pd.to_numeric(df["total_energy_demand_sd_kWh_per_day"], errors="coerce") / 1000.0

    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.bar(
        x_positions,
        mean_mwh,
        yerr=sd_mwh,
        width=0.62,
        color="#4C78A8",
        edgecolor="black",
        linewidth=1.0,
        ecolor="#F58518",
        capsize=6,
        error_kw={"elinewidth": 1.8},
        alpha=0.9,
    )
    ax.set_xlabel("Electrification rate (%)", fontsize=14)
    ax.set_ylabel("Daily total charging demand\n(MWh/day)", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "figure_daily_total_actual_charging_demand.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_vehicle_consistency_figure(df: pd.DataFrame) -> Path:
    x_positions, x_labels = _x_positions_and_labels(df)

    metrics = [
        (
            "travel_distance_per_vehicle_mean_km",
            "travel_distance_per_vehicle_sd_km",
            "Travel distance\n(km/vehicle-day)",
            "#4C78A8",
        ),
        (
            "driving_energy_per_vehicle_mean_kWh",
            "driving_energy_per_vehicle_sd_kWh",
            "Driving energy\n(kWh/vehicle-day)",
            "#59A14F",
        ),
        (
            "charging_energy_per_vehicle_mean_kWh",
            "charging_energy_per_vehicle_sd_kWh",
            "Actual charging energy\n(kWh/vehicle-day)",
            "#F28E2B",
        ),
        (
            "charged_vehicle_share_mean_percent",
            "charged_vehicle_share_sd_percent",
            "Vehicles charged\n(%)",
            "#E15759",
        ),
        (
            "charging_events_per_charged_vehicle_mean",
            "charging_events_per_charged_vehicle_sd",
            "Charging events\n(events/charged vehicle-day)",
            "#B07AA1",
        ),
        (
            "charging_energy_per_charged_vehicle_mean_kWh",
            "charging_energy_per_charged_vehicle_sd_kWh",
            "Charging energy\n(kWh/charged vehicle-day)",
            "#76B7B2",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.3), sharex=True)
    axes = axes.flatten()

    for ax, (mean_col, sd_col, ylabel, color) in zip(axes, metrics):
        y = pd.to_numeric(df[mean_col], errors="coerce")
        ax.bar(
            x_positions,
            y,
            width=0.62,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Electrification rate (%)", fontsize=12)
        ax.set_ylim(bottom=0)
        if np.isfinite(y).any():
            ax.set_ylim(top=float(np.nanmax(y)) * 2.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=10.5)

    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "figure_vehicle_charging_consistency.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_vehicle_consistency_table(df: pd.DataFrame) -> Path:
    table_df = pd.DataFrame(
        {
            "전동화율(%)": pd.to_numeric(
                df["penetration_rate_percent"], errors="coerce"
            ),
            "차량당 주행거리(km/일)": pd.to_numeric(
                df["travel_distance_per_vehicle_mean_km"], errors="coerce"
            ),
            "차량당 주행에너지(kWh/일)": pd.to_numeric(
                df["driving_energy_per_vehicle_mean_kWh"], errors="coerce"
            ),
            "차량당 실제 충전에너지(kWh/일)": pd.to_numeric(
                df["charging_energy_per_vehicle_mean_kWh"], errors="coerce"
            ),
            "충전 경험 차량 비율(%)": pd.to_numeric(
                df["charged_vehicle_share_mean_percent"], errors="coerce"
            ),
            "충전차량당 충전횟수(회/일)": pd.to_numeric(
                df["charging_events_per_charged_vehicle_mean"], errors="coerce"
            ),
            "충전차량당 충전에너지(kWh/일)": pd.to_numeric(
                df["charging_energy_per_charged_vehicle_mean_kWh"], errors="coerce"
            ),
        }
    ).round(2)

    output_path = OUTPUT_FOLDER / "table_vehicle_level_charging_demand_characteristics_seed42.csv"
    table_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df = load_data()
    generated = [
        create_vehicle_consistency_figure(df),
        create_total_demand_figure(df),
        create_vehicle_consistency_table(df),
    ]
    print(f"Generated {len(generated)} report figures.")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
