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
    TairC_list, Qlight_list, RH_list, cs_list = [], [], [], []

    # Compute variables for each timestep
    for t in times:
        w = dummyWeather.weather_custom(t, config)
        TairC_list.append(w['TairC'])
        Qlight_list.append(w['Qlight'])
        RH_list.append(w['RH'])
        cs_list.append(w['cs'])

    # Convert lists to numpy arrays for plotting
    TairC = np.array(TairC_list)
    Qlight = np.array(Qlight_list)
    RH = np.array(RH_list)
    cs = np.array(cs_list)

    # Plotting
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(times * 24, TairC, label='Air Temperature ($T_{air}$)', color='red')
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(times * 24, Qlight, label='Photosynthetically active radiation (PAR or $Q_{light}$)', color='orange')
    plt.xlabel('Hour of Day')
    plt.ylabel('$Q_{light}$ (mmol photons/cm² day)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(times * 24, cs, label='CO2 molar fraction ($c_{bl}$ or $c_s$)', color='green')
    plt.xlabel('Hour of Day')
    plt.ylabel('CO2 molar fraction on the leaf surface (mol/mol)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(times * 24, RH, label='Relative humidity (RH)', color='purple')
    plt.xlabel('Hour of Day')
    plt.ylabel('Relative air humidity (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Plot saved as '{plot_file}'")


if __name__ == "__main__":
    main()