import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

sys.path.append("../../../")
sys.path.append("../../../modelparameter/functional")
from modelparameter.functional.climate import dummyWeather

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <weather_config.json> <plot.png>")
        sys.exit(1)

    config_file = sys.argv[1]
    plot_file = sys.argv[2]
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = dummyWeather.load_weather_config(config_file)

    times = np.linspace(0, 1, 25)
    RH_list, ea_list, es_list, TairC_list = [], [], [], []

    # Compute variables for each timestep
    for t in times:
        w = dummyWeather.weather_custom(t, config)
        RH_list.append(w['RH'])
        TairC_list.append(w['TairC'])
        ea_list.append(w['ea'])
        es_list.append(w['es'])

    # Convert lists to numpy arrays for plotting
    RH = np.array(RH_list)
    TairC = np.array(TairC_list)
    ea = np.array(ea_list)
    es = np.array(es_list)

    # Plotting
    plt.figure(figsize=(10, 14))

    plt.subplot(3, 1, 1)
    plt.plot(times * 24, es, label='Saturated vapor pressure ($e_s$)', color='midnightblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Saturated vapor pressure (hPa)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times * 24, ea, label='Actual vapour pressure ($e_a$)', color='cadetblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Actual vapour pressure (hPa)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    ax = plt.gca()  # Get current axis

    # Plot ea and es on the primary y-axis
    l1 = ax.plot(times * 24, es, label='Saturated vapor pressure ($e_s$)', color='midnightblue')
    l2 = ax.plot(times * 24, ea, label='Actual vapour pressure ($e_a$)', color='cadetblue')
    ax.set_ylabel('Vapor pressure (hPa)')
    ax.grid(True)

    # Combine legends from both axes
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right')
    ax.set_xlabel('Hour of Day')

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Plot saved as '{plot_file}'")


if __name__ == "__main__":
    main()