import os
import sys

sys.path.append("../../")
sys.path.append("../../src/")
sys.path.append("../../modelparameter/")

import plantbox as pb
import numpy as np
import pandas as pd
import visualisation.vtk_plot as vp
import matplotlib.pyplot as plt
import argparse

from functional.xylem_flux import XylemFluxPython
from functional.phloem_flux import PhloemFluxPython
from functional.PlantHydraulicParameters import PlantHydraulicParameters
from modelparameter.functional.plant_photosynthesis.wheat_FcVB_Giraud2023adapted import *
from modelparameter.functional.plant_hydraulics.wheat_Giraud2023adapted import *
from modelparameter.functional.plant_sucrose.wheat_phloem_Giraud2023adapted import *
from datetime import datetime
from matplotlib.dates import DateFormatter, HourLocator
from modelparameter.functional.climate import dummyWeather

def getWeatherData(sim_time):
    diffDt = abs(pd.to_timedelta(weatherData['time']) - pd.to_timedelta(sim_time%1, unit='d'))
    line_data = np.where(diffDt == min(diffDt))[0][0]
    return weatherData.iloc[line_data]  # get the weather data for the current time step

# Command-line arguments (defaults preserve previous behavior)
parser = argparse.ArgumentParser(description="Simulate phloem flow")
parser.add_argument('--save-image', action='store_true', dest='save_image',
                    help='Save plant images each step (default: False)')
parser.add_argument('--image-dir', type=str, default='images', dest='image_dir',
                    help='Directory to save images (default: images)')
parser.add_argument('--phloem-dir', type=str, default='data/sim', dest='phloem_dir',
                    help='Directory to save phloem output (default: data/tmp)')
parser.add_argument('--weather-file', type=str, default='climate/baseline.json', dest='weather_file',
                    help='Weather configuration file (default: climate/baseline.json)')
args = parser.parse_args()

save_image = args.save_image
image_dir = args.image_dir
phloem_dir = args.phloem_dir
weather_config = args.weather_file

""" Parameters and variables """
plant_age = 5 # [day] init simtime
sim_time = 5 # [day]
dt = 1./24.
N = int(sim_time/dt)
depth = 60
p_mean = -600 # mean soil water potential [cm]

""" Weather data """
path = "../../modelparameter/functional/climate/"
weatherData = pd.read_csv(path + 'Selhausen_weather_data.txt', delimiter="\t")
config = dummyWeather.load_weather_config(weather_config)
weatherInit = dummyWeather.weather_custom(plant_age, config)

""" Plant """
plant = pb.MappedPlant(seednum=2)
# plant.disableExtraNode()
path = "../../modelparameter/structural/plant/"
name = "Triticum_aestivum_test_2021" # "Triticum_aestivum_adapted_2023"
plant.readParameters(path + name + ".xml")

sdf = pb.SDF_PlantBox(np.inf, np.inf, depth)
plant.setGeometry(sdf) # creates soil space to stop roots from growing out of the soil

verbose = False
plant.initialize(verbose)
plant.simulate(plant_age, verbose)

""" Soil """
p_bot = p_mean + depth/2
p_top = p_mean - depth/2
sx = np.linspace(p_top, p_bot, depth) # soil water potential per voxel
picker = lambda x,y,z : max(int(np.floor(-z)), -1)
plant.setSoilGrid(picker)

""" Plant functional properties """
params = PlantHydraulicParameters()
params.read_parameters("../../modelparameter/functional/plant_hydraulics/wheat_Giraud2023adapted")

hm = PhloemFluxPython(plant, params, psiXylInit=min(sx), ciInit=weatherInit['co2']*0.5)
hm.wilting_point = -10000

path = '../../modelparameter/functional/'
hm.read_photosynthesis_parameters(filename=path+"plant_photosynthesis/photosynthesis_parameters2025")
hm.read_phloem_parameters(filename=path+"plant_sucrose/phloem_parameters2025")
# list_data = hm.get_phloem_data_list() # option of data that can be obtained from the phloem model
# hm.write_phloem_parameters(filename='phloem_parameters')

time = []
cumulAssimilation = 0.
cumulTranspiration = 0.
Q_Rm_is, Q_Gr_is, Q_Exud_is, Q_Water_is = [], [], [], []

print("Entering simulation loop...")
""" Simulation loop """
for i in range(N):
    # Create output directory if saving is enabled
    if save_image:
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{image_dir}/plant_{i:02d}.png"
    else:
        filename = None

    vp.plot_plant(plant, "organType", render=save_image,
                  interactiveImage=False, save_path=filename)

    """ Weather variables """
    weatherData_i = dummyWeather.weather_custom(plant_age, config)
    weatherData_i['PAR'] = weatherData_i['Qlight'] * (24*3600) / 1e4

    """ Plant growth """
    plant_age += dt
    plant.simulate(dt, False)

    """ Plant transpiration and photosynthesis """
    hm.pCO2 = weatherData_i['co2']
    es = hm.get_es(weatherData_i['Tair'])
    ea = es * weatherData_i['RH']

    print("Solve: Plant transpiration and photosynthesis...")
    hm.solve(sim_time=plant_age, rsx=sx, cells=True,
            ea=ea, es=es,
            PAR=weatherData_i['PAR'],
            TairC=weatherData_i['Tair'],
            verbose=0)

    """ Plant inner carbon balance """
    print("Solve: Phloem flow...")
    os.makedirs(phloem_dir, exist_ok=True)
    phloem_file = f"{phloem_dir}/phloem_{i:02d}.txt"

    hm.solve_phloem_flow(plant_age, dt, weatherData_i['Tair'], unit=1, outputfile=phloem_file)

    # Save all simulation data in HDF5 format
    hm.save_simulation_data(
        step=i,
        sim_time=sim_time,
        plant_age=plant_age,
        dt=dt,
        weather_data=weatherData_i,
        outdir=phloem_dir,
        save_params=(i==0)  # Only save parameters on first step
    )

    """ Post processing """
    print("Post processing...")
    cumulAssimilation  += np.sum(hm.get_net_assimilation())  * dt
    cumulTranspiration += np.sum(hm.get_transpiration()) * dt

    C_ST = hm.get_phloem_data(data="sieve tube concentration")

    # Cumulative
    Q_Rm = hm.get_phloem_data(data="maintenance respiration", doSum=True)
    Q_Exud = hm.get_phloem_data(data="exudation", doSum=True)
    Q_Gr = hm.get_phloem_data(data="growth", doSum=True)
    Q_out = Q_Rm + Q_Exud + Q_Gr

    # Last time step
    Q_Rm_i = hm.get_phloem_data(data="maintenance respiration", last=True, doSum=True)
    Q_Exud_i = hm.get_phloem_data(data="exudation", last=True, doSum=True)
    Q_Gr_i = hm.get_phloem_data(data="growth", last=True, doSum=True)
    Q_out_i = Q_Rm_i + Q_Exud_i + Q_Gr_i

    n = round(float(i) / float(N-1) * 100.)
    print("\n[" + ''.join(["*"]) * n + ''.join([" "]) * (100 - n) + "]")
    print("\t\tat ", int(np.floor(plant_age)),"d", int((plant_age%1)*24),"h, PAR:",  round(weatherData_i['PAR']*1e6),"mumol m-2 s-1")
    print("cumulative: transpiration {:5.2e} [cm3]\tnet assimilation {:5.2e} [mol]".format(cumulTranspiration, cumulAssimilation))
    print("sucrose concentration in sieve tube (mol ml-1):\n\tmean {:.2e}\tmin  {:5.2e}\tmax  {:5.2e}".format(np.mean(C_ST), min(C_ST), max(C_ST)))
    print("aggregated sink repartition at last time step (%) :\n\tRm   {:5.1f}\tGr   {:5.1f}\tExud {:5.1f}".format(Q_Rm_i/Q_out_i*100,  Q_Gr_i/Q_out_i*100,Q_Exud_i/Q_out_i*100))
    print("total aggregated sink repartition (%) :\n\tRm   {:5.1f}\tGr   {:5.1f}\tExud {:5.1f}".format(Q_Rm/Q_out*100, Q_Gr/Q_out*100, Q_Exud/Q_out*100))

    # Convert plant_age (in days) to time of day for plotting
    time_of_day = plant_age % 1  # Get fractional part (time of day)
    hours = int(time_of_day * 24)
    minutes = int((time_of_day * 24 - hours) * 60)
    seconds = int(((time_of_day * 24 - hours) * 60 - minutes) * 60)
    time.append(datetime.strptime(f'{hours:02d}:{minutes:02d}:{seconds:02d}', '%H:%M:%S'))
    Q_Rm_is.append(Q_Rm_i/dt)
    Q_Exud_is.append(Q_Exud_i/dt)
    Q_Gr_is.append(Q_Gr_i/dt)
    Q_Water_is.append(np.sum(hm.get_transpiration()))

# """ Plot results """
# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(time, Q_Rm_is)
# axs[0,0].set(xlabel="Time [hh:mm]", ylabel='Total respiration rate (mol/day)')
# axs[1,0].plot(time, Q_Gr_is, 'tab:red')
# axs[1,0].set(xlabel="Time [hh:mm]", ylabel='Total growth rate (mol/day)')
# axs[0,1].plot(time, Q_Exud_is , 'tab:brown')
# axs[0,1].set(xlabel="Time [hh:mm]", ylabel='Total exudation\nrate (mol/day)')
# axs[1,1].plot(time, Q_Water_is , 'tab:brown')
# axs[1,1].set(xlabel="Time [hh:mm]", ylabel='Total transpiration\nrate (cm3/day)')

# for ax in axs.flatten():
#     ax.xaxis.set_major_locator(HourLocator(range(0, 25, 1)))
#     ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#     ax.fmt_xdata = DateFormatter('%H:%M')
#     fig.autofmt_xdate()

# fig.tight_layout()
# plt.show()
