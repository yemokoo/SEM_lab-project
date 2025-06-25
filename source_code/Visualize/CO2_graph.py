import matplotlib.pyplot as plt
import numpy as np

# Define the years (x-axis)
years = np.arange(2000, 2071, 1)

# Dummy data (approximations)
aviation = [
    0.8 + 0.04 * i if i < 20 else 0.8 + 0.04 * 20 + 0.01 * (i - 20) if i < 35 else 0.8 + 0.04 * 20 + 0.01 * 15 - 0.02 * (i - 35)
    for i in range(len(years))
]
shipping = [
    0.9 + 0.03 * i if i < 15 else 0.9 + 0.03 * 15 + 0.005 * (i - 15) if i < 30 else 0.9 + 0.03 * 15 + 0.005 * 15 - 0.015 * (i - 30)
    for i in range(len(years))
]
medium_heavy_trucks = [
    1.0 + 0.02 * i if i < 10 else 1.0 + 0.02 * 10 - 0.01 * (i - 10) if i < 45 else 1.0 + 0.02 * 10 - 0.01 * 35 - 0.005 * (i - 45)
    for i in range(len(years))
]
light_commercial_vehicles = [
    0.6 + 0.015 * i if i < 12 else 0.6 + 0.015 * 12 - 0.025 * (i - 12) if i < 50 else 0.6 + 0.015 * 12 - 0.025 * 38 - 0.002 * (i - 50)
    for i in range(len(years))
]
buses_minibuses = [
    0.1 + 0.005 * i if i < 8 else 0.1 + 0.005 * 8 - 0.008 * (i - 8) if i < 40 else 0.1 + 0.005 * 8 - 0.008 * 32
    for i in range(len(years))
]
passenger_cars = [
    1.8 + 0.05 * i if i < 10 else 1.8 + 0.05 * 10 - 0.04 * (i - 10) if i < 55 else 1.8 + 0.05 * 10 - 0.04 * 45 - 0.005 * (i - 55)
    for i in range(len(years))
]
two_three_wheelers = [0.2 - 0.002 * i if i < 45 else 0.2 - 0.002 * 45 for i in range(len(years))]
rail = [0.05 for _ in range(len(years))]


# Create the stacked area chart (3:1 aspect ratio)
plt.figure(figsize=(15, 5))
plt.stackplot(
    years,
    aviation,
    shipping,
    medium_heavy_trucks,
    light_commercial_vehicles,
    buses_minibuses,
    passenger_cars,
    two_three_wheelers,
    rail,
    labels=[
        "Aviation",
        "Shipping",
        "Medium- and heavy trucks",
        "Light-commercial vehicles",
        "Buses and minibuses",
        "Passenger cars",
        "Two/three-wheelers",
        "Rail",
    ],
    colors=[
        "#c8e6f2",  # Aviation - Even Lighter Blue
        "#d9eff7",  # Shipping - Very Light Blue
        "#458b74",  # Medium- and heavy trucks - Teal (Highlight)
        "#f9f0dd",  # Light-commercial vehicles - Very Light Orange
        "#faf3d3",  # Buses and minibuses - Very Light Orange
        "#f2d7c9",  # Passenger cars - Very Light Orange
        "#e2d1ea",  # Two/three-wheelers - Very Light Purple
        "#f2f3f4",  # Rail - Almost White
    ],
)


# Title and legend
plt.title("G1CO2 per year", loc="left", fontsize=10, pad=10)

# Get legend handles/labels, make "Medium- and heavy trucks" bold
handles, labels = plt.gca().get_legend_handles_labels()
index = labels.index("Medium- and heavy trucks")
labels[index] = r"$\bf{" + labels[index] + "}$"
plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)


# Remove spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# IEA License
plt.text(
    2070,
    -1.0,
    "IEA. Licence: CC BY 4.0",
    fontsize=8,
    ha="right",
    va="bottom",
    color="gray",
)

# Y-axis limits and ticks
plt.ylim(0, 9.5)
plt.yticks(np.arange(0, 10, 1))

# X-axis limits
plt.xlim(2000, 2070)

plt.tight_layout()
plt.show()