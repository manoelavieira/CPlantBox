import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the dataset
    file_path = "../../../modelparameter/functional/climate/Selhausen_weather_data.txt"
    df = pd.read_csv(file_path, sep="\t", quotechar='"')

    # Convert time strings to hour values
    df["Hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour + pd.to_datetime(df["time"], format="%H:%M:%S").dt.minute / 60

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot PAR, CO2, RH, Tair on subplots
    plt.subplot(2, 2, 1)
    plt.plot(df["Hour"], df["Tair"], label='Air temperature (°C)', color='red')
    plt.xlabel("Hour of Day")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df["Hour"], df["PAR"], label='Photosynthetically active radiation (PAR)', color='orange')
    plt.xlabel("Hour of Day")
    plt.ylabel("PAR")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(df["Hour"], df["co2"] * 1e6, label='CO2 molar fraction ($c_{bl}$ or $c_s$)', color='green')
    plt.xlabel("Hour of Day")
    plt.ylabel("CO2 molar fraction")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(df["Hour"], df["RH"] * 100, label='Relative humidity (RH)', color='purple')
    plt.xlabel("Hour of Day")
    plt.ylabel("Relative air humidity (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/Selhausen_weather_data.png")
    print(f"Plot saved as figures/Selhausen_weather_data.png")


if __name__ == "__main__":
    main()