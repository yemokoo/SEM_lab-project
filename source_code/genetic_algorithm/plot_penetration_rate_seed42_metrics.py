import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_BASE_FOLDER = Path(
    r"D:\연구실\연구\화물차 충전소 배치 최적화\Data\Processed_Data\simulator\result_for_kori"
)
COMBINED_CSV_PATH = RESULTS_BASE_FOLDER / "penetration_rate_vehicle_station_summary_seed42.csv"
PLOTS_OUTPUT_FOLDER = RESULTS_BASE_FOLDER / "seed42_metric_plots"
STYLE_OUTPUT_FOLDERS = {
    "option1_mean_only": PLOTS_OUTPUT_FOLDER / "option1_mean_only",
    "option2_mean_ci95": PLOTS_OUTPUT_FOLDER / "option2_mean_ci95",
    "option3_mean_and_sd": PLOTS_OUTPUT_FOLDER / "option3_mean_and_sd",
    "option4_mean_and_cv": PLOTS_OUTPUT_FOLDER / "option4_mean_and_cv",
}


METRIC_SPECS = [
    {
        "mean_col": "sample_size_mean",
        "sd_col": "sample_size_sd",
        "observation_col": "sample_size_observations",
        "label": "Sample Size",
        "ylabel": "Vehicles",
        "filename": "sample_size.png",
    },
    {
        "mean_col": "total_energy_demand_mean_kWh_per_day",
        "sd_col": "total_energy_demand_sd_kWh_per_day",
        "observation_col": "total_energy_demand_observations",
        "label": "Total Actual Charging Demand",
        "ylabel": "kWh/day",
        "filename": "total_actual_charging_demand.png",
    },
    {
        "mean_col": "travel_distance_per_vehicle_mean_km",
        "sd_col": "travel_distance_per_vehicle_sd_km",
        "observation_col": "travel_distance_vehicle_observations",
        "label": "Travel Distance per Vehicle",
        "ylabel": "km",
        "filename": "travel_distance_per_vehicle.png",
    },
    {
        "mean_col": "driving_energy_per_vehicle_mean_kWh",
        "sd_col": "driving_energy_per_vehicle_sd_kWh",
        "observation_col": "driving_energy_vehicle_observations",
        "label": "Driving Energy per Vehicle",
        "ylabel": "kWh",
        "filename": "driving_energy_per_vehicle.png",
    },
    {
        "mean_col": "charging_energy_per_vehicle_mean_kWh",
        "sd_col": "charging_energy_per_vehicle_sd_kWh",
        "observation_col": "charging_energy_vehicle_observations",
        "label": "Actual Charging Energy per Vehicle",
        "ylabel": "kWh",
        "filename": "actual_charging_energy_per_vehicle.png",
    },
    {
        "mean_col": "charging_events_per_vehicle_mean",
        "sd_col": "charging_events_per_vehicle_sd",
        "observation_col": "charging_events_vehicle_observations",
        "label": "Charging Events per Vehicle",
        "ylabel": "Count",
        "filename": "charging_events_per_vehicle.png",
    },
    {
        "mean_col": "charged_vehicle_share_mean_percent",
        "sd_col": "charged_vehicle_share_sd_percent",
        "observation_col": "charged_vehicle_share_observations",
        "label": "Charged Vehicle Share",
        "ylabel": "%",
        "filename": "charged_vehicle_share.png",
    },
    {
        "mean_col": "installed_power_per_station_mean_kW",
        "sd_col": "installed_power_per_station_sd_kW",
        "observation_col": "installed_power_station_observations",
        "label": "Installed Power per Station",
        "ylabel": "kW",
        "filename": "installed_power_per_station.png",
    },
    {
        "mean_col": "maximum_power_per_station_mean_kW",
        "sd_col": "maximum_power_per_station_sd_kW",
        "observation_col": "maximum_power_station_observations",
        "label": "Peak Observed Power per Station",
        "ylabel": "kW",
        "filename": "peak_observed_power_per_station.png",
    },
    {
        "mean_col": "peak_power_ratio_per_station_mean_percent",
        "sd_col": "peak_power_ratio_per_station_sd_percent",
        "observation_col": "peak_power_ratio_station_observations",
        "label": "Peak Power Ratio per Station",
        "ylabel": "%",
        "filename": "peak_power_ratio_per_station.png",
    },
    {
        "mean_col": "maximum_charging_vehicles_per_station_mean",
        "sd_col": "maximum_charging_vehicles_per_station_sd",
        "observation_col": "maximum_charging_vehicles_station_observations",
        "label": "Peak Simultaneous Charging Vehicles per Station",
        "ylabel": "Vehicles",
        "filename": "peak_simultaneous_charging_vehicles_per_station.png",
    },
]


def build_seed42_combined_csv(output_csv_path: Path) -> pd.DataFrame:
    summary_paths = sorted(
        p
        for p in RESULTS_BASE_FOLDER.rglob("penetration_rate_vehicle_station_summary.csv")
        if "seed=42" in str(p)
    )
    if not summary_paths:
        raise FileNotFoundError("No seed=42 penetration summary files were found.")

    frames = []
    for path in summary_paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.sort_values(
        by=["penetration_rate_percent", "experiment_name"]
    ).reset_index(drop=True)
    combined_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return combined_df


def load_seed42_summary() -> pd.DataFrame:
    if COMBINED_CSV_PATH.exists():
        return pd.read_csv(COMBINED_CSV_PATH)
    return build_seed42_combined_csv(COMBINED_CSV_PATH)


def prepare_plot_df(df: pd.DataFrame, metric_spec: dict) -> pd.DataFrame:
    mean_col = metric_spec["mean_col"]
    sd_col = metric_spec["sd_col"]
    observation_col = metric_spec["observation_col"]

    plot_df = df[["penetration_rate_percent", mean_col, sd_col, observation_col]].copy()
    plot_df = plot_df.sort_values(by="penetration_rate_percent")
    plot_df["penetration_rate_percent"] = pd.to_numeric(plot_df["penetration_rate_percent"], errors="coerce")
    plot_df["mean"] = pd.to_numeric(plot_df[mean_col], errors="coerce")
    plot_df["sd"] = pd.to_numeric(plot_df[sd_col], errors="coerce")
    plot_df["n"] = pd.to_numeric(plot_df[observation_col], errors="coerce")
    plot_df["ci95"] = np.where(
        plot_df["n"] > 0,
        1.96 * plot_df["sd"] / np.sqrt(plot_df["n"]),
        0.0,
    )
    plot_df["cv_percent"] = np.where(
        plot_df["mean"].abs() > 1e-12,
        (plot_df["sd"] / plot_df["mean"]) * 100.0,
        0.0,
    )
    plot_df["x_label"] = plot_df["penetration_rate_percent"].map(
        lambda v: f"{int(v)}%" if pd.notna(v) else ""
    )
    plot_df["x_position"] = range(len(plot_df))
    return plot_df


def _style_axis(ax, x_positions, x_labels, ylabel, title):
    ax.set_xlabel("Penetration Rate (%)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(list(x_labels))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_option1_mean_only(plot_df: pd.DataFrame, metric_spec: dict, output_folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        plot_df["x_position"],
        plot_df["mean"],
        width=0.65,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    _style_axis(ax, plot_df["x_position"], plot_df["x_label"], metric_spec["ylabel"], f"{metric_spec['label']} (Mean Only)")
    plt.tight_layout()

    output_path = output_folder / metric_spec["filename"]
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_option2_mean_ci95(plot_df: pd.DataFrame, metric_spec: dict, output_folder: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        plot_df["x_position"],
        plot_df["mean"],
        yerr=plot_df["ci95"],
        width=0.65,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        ecolor="tab:orange",
        capsize=6,
        error_kw={"elinewidth": 2},
        alpha=0.85,
    )
    _style_axis(ax, plot_df["x_position"], plot_df["x_label"], metric_spec["ylabel"], f"{metric_spec['label']} (Mean + 95% CI)")
    plt.tight_layout()

    output_path = output_folder / metric_spec["filename"]
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_option3_mean_and_sd(plot_df: pd.DataFrame, metric_spec: dict, output_folder: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].bar(
        plot_df["x_position"],
        plot_df["mean"],
        width=0.65,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    _style_axis(axes[0], plot_df["x_position"], plot_df["x_label"], metric_spec["ylabel"], f"{metric_spec['label']} (Mean)")

    axes[1].bar(
        plot_df["x_position"],
        plot_df["sd"],
        width=0.65,
        color="tab:orange",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    _style_axis(axes[1], plot_df["x_position"], plot_df["x_label"], metric_spec["ylabel"], f"{metric_spec['label']} (SD)")

    plt.tight_layout()
    output_path = output_folder / metric_spec["filename"]
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_option4_mean_and_cv(plot_df: pd.DataFrame, metric_spec: dict, output_folder: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].bar(
        plot_df["x_position"],
        plot_df["mean"],
        width=0.65,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    _style_axis(axes[0], plot_df["x_position"], plot_df["x_label"], metric_spec["ylabel"], f"{metric_spec['label']} (Mean)")

    axes[1].bar(
        plot_df["x_position"],
        plot_df["cv_percent"],
        width=0.65,
        color="tab:green",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    _style_axis(axes[1], plot_df["x_position"], plot_df["x_label"], "CV (%)", f"{metric_spec['label']} (CV)")

    plt.tight_layout()
    output_path = output_folder / metric_spec["filename"]
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


PLOT_BUILDERS = {
    "option1_mean_only": plot_option1_mean_only,
    "option2_mean_ci95": plot_option2_mean_ci95,
    "option3_mean_and_sd": plot_option3_mean_and_sd,
    "option4_mean_and_cv": plot_option4_mean_and_cv,
}


def generate_seed42_metric_plots() -> dict[str, list[Path]]:
    os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)
    for folder in STYLE_OUTPUT_FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

    summary_df = load_seed42_summary()
    generated_paths: dict[str, list[Path]] = {key: [] for key in STYLE_OUTPUT_FOLDERS}

    for metric_spec in METRIC_SPECS:
        required_cols = [metric_spec["mean_col"], metric_spec["sd_col"], metric_spec["observation_col"]]
        if any(col not in summary_df.columns for col in required_cols):
            continue
        plot_df = prepare_plot_df(summary_df, metric_spec)
        for style_name, plot_builder in PLOT_BUILDERS.items():
            generated_paths[style_name].append(
                plot_builder(plot_df, metric_spec, STYLE_OUTPUT_FOLDERS[style_name])
            )

    return generated_paths


if __name__ == "__main__":
    generated_files = generate_seed42_metric_plots()
    total_count = sum(len(paths) for paths in generated_files.values())
    print(f"Generated {total_count} plot files.")
    for style_name, paths in generated_files.items():
        print(f"[{style_name}] {len(paths)} files")
        for path in paths:
            print(path)
