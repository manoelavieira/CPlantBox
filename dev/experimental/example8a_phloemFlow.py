import os
import sys
sys.path.append("../../")
sys.path.append("../../src/")
sys.path.append("../../modelparameter/")

# # Coupled carbon and water flow in CPlantBox (with a static soil)
#
# # Simulation of water and carbon movement
#
# In the following we will show how to compute the coupled water and carbon flow in the plant.
#
# We consider a dynamic plant and a static soil.
# To compute the carbon flux, we use the code developped by Lacointe et al. (2019).
#
# **Reference**
# A Lacointe and P. Minchin. A mechanistic model to predict distribution of carbon among multiple sinks. *Methods in molecular biology* (Clifton, N.J.) vol. 2014, 2019.
#
# The sucrose flow depends on several plant, soil and atmospheric variables. For clarity, the basic functions defining those variables were moved to the file "parametersSucroseFlow".

import plantbox as pb
import numpy as np
import matplotlib.pyplot as plt
import visualisation.vtk_plot as vp # for quick 3d vizualisations
import time
import pandas as pd

from functional.xylem_flux import XylemFluxPython  # Python hybrid solver
from functional.phloem_flux import PhloemFluxPython
from modelparameter.functional.plant_photosynthesis.wheat_FcVB_Giraud2023adapted import *
from modelparameter.functional.plant_hydraulics.wheat_Giraud2023adapted import *
from modelparameter.functional.plant_sucrose.wheat_phloem_Giraud2023adapted import *
from modelparameter.functional.climate import dummyWeather

# --------------------------------
# 0. Configurations
# --------------------------------
# Define constants
RUN_SIM_FLAG = True

# Visualization configuration
save_image = True # set to True to save image to file
show_image = False # set to True to open the image after saving
image_dir = "images/vtk" # folder where images will be saved
phloem_dir = "phloem"

one_weather = True
config_phase1_file = None
config_phase2_file = None

# Parse parameters
if len(sys.argv) not in [2, 3, 4]:
    print("Usage: python script.py <results_file.csv> <weather_config1.json> <weather_config2.json>")
    sys.exit(1)

results_file = sys.argv[1]
config_phase1_file = sys.argv[2]
if not os.path.exists(config_phase1_file):
    print(f"Error: Configuration file '{config_phase1_file}' not found.")
    sys.exit(1)

if len(sys.argv) == 4:
    one_weather = False
    config_phase2_file = sys.argv[3]
    if not os.path.exists(config_phase2_file):
        print(f"Error: Configuration file '{config_phase2_file}' not found.")
        sys.exit(1)


# --------------------------------
# 1. Define initial conditions
# --------------------------------
# We start with a small plant to have a lower computation time
simInit = 3 # [day] init simtime
simMax = 6
weatherDuration = 0
dt = 1/24
depth = 60

# Load separate weather configs
# config_phase1 = dummyWeather.load_weather_config(config_phase1_file)
# config_phase2 = dummyWeather.load_weather_config(config_phase2_file) if config_phase2_file else None
# weatherInit = dummyWeather.weather_custom(simInit, config_phase1)

# config = dummyWeather.load_weather_config("config/normal_weather.json")
# weatherInit = dummyWeather.weather_custom(simInit, config)
weatherInit = dummyWeather.weather(simInit)
simDuration = simInit

# Plant system
pl = pb.MappedPlant(seednum = 2) # seednum: gives the option of setting a random seed to make the simulations replicable
path = "../../modelparameter/structural/plant/"
name = "Triticum_aestivum_adapted_2023"
pl.readParameters(path + name + ".xml")

sdf = pb.SDF_PlantBox(np.Inf, np.Inf, depth )
pl.setGeometry(sdf) # creates soil space to stop roots from growing out of the soil

verbose = False
pl.initialize(verbose)
pl.simulate(simInit, verbose)

# For post-processing
Q_out = 0 # sucrose lost by the plant
AnSum = 0 # assimilation
# phloem_filename = "phloemoutputs.txt"

Q_Rmbu = np.array([0.])
Q_Grbu = np.array([0.])
Q_Exudbu = np.array([0.])
Q_STbu = np.array([0.])

Q_Rmall = np.array([])
Q_Grall = np.array([])
Q_Exudall = np.array([])
lengthTotall = np.array([])
sim_time = np.array([])
lengthTotInit = sum(pl.segLength())
lengthTotBU = sum(pl.segLength())

df_records = []


# --------------------------------
# 2. Define static soil
# --------------------------------
min_b = [-3./2, -12./2, -61.] # distance between wheat plants
max_b = [3./2, 12./2, 0.]
cell_number = [6, 24, 61] # soil resolution
layers = depth; soilvolume = (depth / layers) * 3 * 12
k_soil = [] # conductivity of soil when in contact with roots
p_mean = weatherInit['p_mean'] # mean soil water potential
p_bot = p_mean + depth/2
p_top = p_mean - depth/2
sx = np.linspace(p_top, p_bot, depth) # soil water potential per voxel

picker = lambda x,y,z : max(int(np.floor(-z)),-1)
pl.setSoilGrid(picker)  # maps segment


# ----------------------------------------------------------
# 3. Create object to compute carbon and water flux
# ----------------------------------------------------------
# The PhloemFluxPython class containes the functionalities of PhotosynthesisPython as well as the sucrose-related functions.

# Give initial guess of leaf water potential and internal CO2 partial pressure (to start computation loop)
r = PhloemFluxPython(pl, psiXylInit=min(sx), ciInit=weatherInit["cs"]*0.5)


# ----------------------------------------------------------
# 4. Set other parameters and initial variable
# ----------------------------------------------------------
# We present bellow some of the main sucrose-related parameters.

r = setPhotosynthesisParameters(r, weatherInit)

r = setKrKx_phloem(r) # conductivity of the sieve tube
r.setKrm2([[2e-5]]) # effect of the sucrose content on maintenance respiration
r.setKrm1([[10e-2]]) # effect of structural sucrose content on maintenance respiration
r.setRhoSucrose([[0.51],[0.65],[0.56]]) # sucrose density per organ type (mmol/cm3)
r.setRmax_st([[14.4,9.0,6.0,14.4],[5.,5.],[15.]]) # maximum growth rate when water and carbon limitation is activated
r.KMfu = 0.11 # michaelis menten coefficient for usage of sucrose
r.beta_loading = 0.6 # feedback effect of sieve tube concentraiton on loading from mesophyll
r.Vmaxloading = 0.05 # mmol/d, max loading rate from mesophyll
r.Mloading = 0.2 # michaelis menten coefficient for loading of sucrose
r.Gr_Y = 0.8 # efficiency of sucrose usage for growth. if <1, we have growth respiration
r.CSTimin = 0.4 # minimum sucrose concentration below which no sucrose usage occures
r.Csoil = 1e-4 # mean soil concentration in sucrose

r.update_viscosity = True # update sucrose viscosity according to concentraiton ?
r.atol = 1e-12 # max absolute error for sucrose flow solver
r.rtol = 1e-8 # max relative error for sucrose flow solver


# --------------------------------
# 5. Launch simulation
# --------------------------------
# In this simulation, we use the same time step for all the modules. The first time steps tend to require longer computation time.
# Increasing the maximum errors allowed for the sucrose computation (r.atol, r.rtol) and the minium and maximum plant segment length (dxMin, dx) can help decrease the computaiton time.

# Start of the entire simulation
sim_start = time.time()
print(f"\n====== Simulation started at: {time.strftime('%H:%M:%S')} ======")


# print("segments:")
# for seg in r.plant.segments:
#     print(f"\t({seg.x}, {seg.y})")

i = 0
while (simDuration <= simMax) and RUN_SIM_FLAG:
    loop_start = time.time()
    print(f"\n====== Loop {i} started at: {time.strftime('%H:%M:%S')} ======")

    print([attr for attr in dir(r) if not attr.startswith("_")])

    # Create output directory if saving is enabled
    if save_image:
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{image_dir}/plant_{i:02d}.png"
    else:
        filename = None

    vp.plot_plant(pl, "subType", render=True, interactiveImage=False, filename=filename, show=show_image)

    Nt = len(r.plant.nodes)
    print(f"simDuration: {simDuration}")
    print(f"number of plant nodes: {Nt}\n")

    # Update weather variables
    if one_weather:
        weatherX = dummyWeather.weather_custom(simDuration, config_phase1)
        print("Using weather config phase 1")
    else:
        if config_phase2 and (simDuration >= simInit + weatherDuration):
            weatherX = dummyWeather.weather_custom(simDuration, config_phase2)
            print("Using weather config phase 2")
        else:
            weatherX = dummyWeather.weather_custom(simDuration, config_phase1)
            print("Using weather config phase 1")

    print(f"weatherX: 'p_mean': {weatherX.get("p_mean")}\n")
    # weatherX = dummyWeather.weather(simDuration) # update weather variables

    r.Qlight = weatherX["Qlight"]
    r = setKrKx_xylem(weatherX["TairC"], weatherX["RH"], r) # update xylem conductivity data

    # Compute plant water flow
    r.solve_photosynthesis(ea_=weatherX["ea"], es_=weatherX["es"],
                           sim_time_=simDuration, sxx_=sx, cells_=True,
                           verbose_=False, doLog_=False, TairC_=weatherX["TairC"])


    AnSum += np.sum(r.Ag4Phloem)*dt # total cumulative carbon assimilaiton
    errLeuning = sum(r.outputFlux) # should be 0 : no storage of water in the plant
    fluxes = np.array(r.outputFlux)

    # returns a map: soil cell → total water flux into that cell
    # hash map with cell indices as keys and fluxes as values [cm3/day]
    fluxesSoil = r.soilFluxes(simDuration, r.psiXyl, sx, approx=False) # root water flux per soil voxel

    # Simulation of phloem flow
    startphloem = simDuration
    endphloem = startphloem + dt
    stepphloem = 1

    os.makedirs(phloem_dir, exist_ok=True)
    phloem_filename = f"{phloem_dir}/phloemoutputs_{i:02d}.txt"
    r.startPM(startphloem, endphloem, stepphloem, (weatherX["TairC"] + 273.15), True, phloem_filename)

    # Get ouput of sucrose flow computation
    Q_ST = np.array(r.Q_out[0:Nt]) # sieve tube sucrose content
    Q_meso = np.array(r.Q_out[Nt:(Nt*2)]) # mesophyll sucrose content
    Q_Rm = np.array(r.Q_out[(Nt*2):(Nt*3)]) # sucrose used for maintenance respiration
    Q_Exud = np.array(r.Q_out[(Nt*3):(Nt*4)]) # sucrose used for exudation
    Q_Gr = np.array(r.Q_out[(Nt*4):(Nt*5)]) # sucrose used for growth and growth respiration

    C_ST = np.array(r.C_ST) # sieve tube sucrose concentraiton
    volST = np.array(r.vol_ST) # sieve tube volume
    volMeso = np.array(r.vol_Meso) # mesophyll volume
    C_meso = Q_meso / volMeso # sucrose concentration in mesophyll
    Q_out = Q_Rm + Q_Exud + Q_Gr # total sucrose lost/used by the plant
    error = sum(Q_ST + Q_meso + Q_out ) - AnSum # balance residual (error)

    lengthTot = sum(r.plant.segLength()) # total plant length

    # Variation of sucrose content at the last time step (mmol)
    Q_ST_i = Q_ST - Q_STbu # in the sieve tubes
    Q_Rm_i = Q_Rm - Q_Rmbu # for maintenance
    Q_Gr_i = Q_Gr - Q_Grbu # for growth
    Q_Exud_i = Q_Exud - Q_Exudbu # for exudation
    Q_out_i = Q_Rm_i + Q_Exud_i + Q_Gr_i # total usage

    # Print some outputs
    print("\nSimulation outputs at", int(np.floor(simDuration)), "d", int((simDuration%1)*24), "h,\n\tPAR:", round(r.Qlight*1e6),"mumol m-2 s-1")
    print("Error in sucrose balance:\n\tabs (mmol) {:5.2e}\trel (-) {:5.2e}".format(error, dummyWeather.div0f(error, AnSum, 1.)))
    print("Error in water balance:\n\tabs (cm3/day) {:5.2e}".format(errLeuning))
    print("Water fluxes (cm3/day):\n\ttranspiration {:5.2e}".format(sum(fluxesSoil.values())))
    print("Assimilated sucrose (cm):\n\tAn {:5.2e}".format(AnSum))
    print("Sucrose concentration in sieve tube (mmol ml-1):\n\tmean {:.2e}\tmin  {:5.2e} at {:d} segs \tmax  {:5.2e}".format(np.mean(C_ST), min(C_ST), len(np.where(C_ST == min(C_ST) )[0]), max(C_ST)))
    print('Acumulated\n\tRm   {:.2e}\tGr   {:.2e}\tExud {:5.2e}'.format(sum(Q_Rm), sum(Q_Gr), sum(Q_Exud)))
    print("Aggregated sink repartition at last time step (%):\n\tRm   {:5.1f}\tGr   {:5.1f}\tExud {:5.1f}".format(sum(Q_Rm_i)/sum(Q_out_i)*100,
        sum(Q_Gr_i)/sum(Q_out_i)*100,sum(Q_Exud_i)/sum(Q_out_i)*100))
    print("Total aggregated sink repartition (%):\n\tRm   {:5.1f}\tGr   {:5.1f}\tExud {:5.1f}".format(sum(Q_Rm)/sum(Q_out)*100,
        sum(Q_Gr)/sum(Q_out)*100,sum(Q_Exud)/sum(Q_out)*100))
    print("Growth rate (cm/day):\n\ttotal {:5.2e}\tlast time step {:5.2e}".format(lengthTot - lengthTotInit, lengthTot - lengthTotBU))


    print("\nDatasetLogger")
    print("Vmaxloading: {:5.1f}".format(r.Vmaxloading))
    print("JW_ST:", np.array2string(np.array(r.JW_ST), precision=2, separator=", "))
    print("C_amont:", np.array2string(np.array(r.C_amont), precision=2, separator=", "))

    # df_records.append({
    #     "day": int(np.floor(simDuration)),
    #     "hour": int((simDuration % 1) * 24),
    #     "simTime": simDuration,
    #     "Qlight": r.Qlight * 1e6,
    #     "Temp": weatherX["TairC"],
    #     "ea": weatherX["ea"],
    #     # "Ag4Phloem": np.sum(r.Ag4Phloem),
    #     # "psiXyl4Phloem": np.sum(r.psiXyl4Phloem),
    #     "AnSum": AnSum,
    #     "ErrorSucrose": error,
    #     "ErrorWater": errLeuning,
    #     "Transpiration": sum(fluxesSoil.values()),
    #     "C_ST (mean)": np.mean(C_ST),
    #     "C_ST (min)": np.min(C_ST),
    #     "C_ST (max)": np.max(C_ST),
    #     "Rm": sum(Q_Rm_i)/sum(Q_out_i)*100, # Aggregated sink repartition at last time step (%)
    #     "Gr": sum(Q_Gr_i)/sum(Q_out_i)*100, # Aggregated sink repartition at last time step (%)
    #     "Exud": sum(Q_Exud_i)/sum(Q_out_i)*100, # Aggregated sink repartition at last time step (%)
    #     "RmTotal": sum(Q_Rm), # Acumulated
    #     "GrTotal": sum(Q_Gr), # Acumulated
    #     "ExudTotal": sum(Q_Exud), # Acumulated
    #     "TotalPlantLength": lengthTot - lengthTotInit, # Growth rate total (since simulation start)
    #     "GrowthRate": lengthTot - lengthTotBU, # Growth rate last time step
    # })

    # print(f"\nAg4Phloem\n\tlength: {len(r.Ag4Phloem)}\n\tvalues: {r.Ag4Phloem}\n")
    # print(f"psiXyl4Phloem\n\tlength: {len(r.psiXyl4Phloem)}\n\tvalues: {r.psiXyl4Phloem}\n")

    # nodes = r.plant.nodes  # list of node objects
    # positions = np.array([[n.x, n.y, n.z] for n in nodes])  # shape: (N, 3)
    # n_nodes = len(nodes)
    # # print(r.plant.printNodes())

    # t_array = np.full((n_nodes, 1), simDuration)
    # psi_array = np.array(r.psiXyl4Phloem).reshape(-1, 1)  # shape: (N, 1)
    # sucrose_array = C_ST.reshape(-1, 1)  # shape: (N, 1)
    # data_timestep = np.hstack([positions, t_array, psi_array, sucrose_array])  # shape: (N, 6)

    # # Save to CSV or append to a master list
    # columns = ["x", "y", "z", "time", "psi_t_x", "s_st"]
    # df_timestep = pd.DataFrame(data_timestep, columns=columns)
    # # df_timestep.to_csv(f"outputs/nodes/test_timestep_{i:03d}.csv", index=False)


    # psi_xylem = np.array(r.psiXyl4Phloem)

    # fig = plt.figure(figsize=(14, 8))
    # axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    # labels = ["s_st (mmol/ml)", "psi_xylem (MPa)"]
    # values = [C_ST, psi_xylem]
    # vmins = [0, -2000]
    # vmaxs = [1, -1000]

    # for ax, val, label, vmin, vmax in zip(axs, values, labels, vmins, vmaxs):
    #     for seg in r.plant.segments:
    #         start_idx = seg.x
    #         end_idx = seg.y
    #         p1 = positions[start_idx]
    #         p2 = positions[end_idx]
    #         xs, ys, zs = zip(p1, p2)
    #         ax.plot(xs, ys, zs, color='gray', linewidth=1)

    #     sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
    #                     c=val, cmap='viridis', s=10, vmin=vmin, vmax=vmax)
    #     fig.colorbar(sc, ax=ax, shrink=0.6, label=label)
    #     ax.set_title(f"{label}")
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")

    # plt.tight_layout()
    # plt.savefig(f"images/nodes/plant_{i:02d}.png")


    # Plant growth based on Gr * Gr_Y
    r.plant.simulate(dt, verbose)
    simDuration += dt

    # For post processing
    Ntbu = Nt
    Nt = len(r.plant.nodes)
    lengthTotBU = lengthTot
    Q_STbu = np.concatenate((Q_ST, np.full(Nt - Ntbu, 0.)))
    Q_Rmbu = np.concatenate((Q_Rm, np.full(Nt - Ntbu, 0.)))
    Q_Grbu = np.concatenate((Q_Gr, np.full(Nt - Ntbu, 0.)))
    Q_Exudbu = np.concatenate((Q_Exud, np.full(Nt - Ntbu, 0.)))

    Q_Rmall = np.append(Q_Rmall, sum(Q_Rm_i))
    Q_Grall = np.append(Q_Grall, sum(Q_Gr_i))
    Q_Exudall = np.append(Q_Exudall, sum(Q_Exud_i))
    lengthTotall = np.append(lengthTotall,lengthTot)
    sim_time = np.append(time, simDuration)

    loop_end = time.time()
    loop_duration = loop_end - loop_start
    print(f"====== Loop {i} ended at: {time.strftime('%H:%M:%S')} (Duration: {loop_duration:.2f} seconds) ======")
    i+=1

# End of the entire simulation
simulation_end = time.time()
total_duration = simulation_end - sim_start
print(f"\n====== Simulation ended at: {time.strftime('%H:%M:%S')} ======")
print(f"====== Total simulation duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes) ======")


# --------------------------------
# 8. Save and plot some results
# --------------------------------
# Convert list of dicts to dataframe and save it to csv
# df = pd.DataFrame(df_records)
# df.to_csv(results_file, index=False)

# print(f"\nSimulation data saved to: {results_file}")

# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(time, Q_Rmall/dt)
# axs[0,0].set(xlabel='day of growth', ylabel='total Rm rate (mmol/day)')
# axs[1,0].plot(time, Q_Grall/dt, 'tab:red')
# axs[1,0].set(xlabel='day of growth', ylabel='total Gr rate (mmol/day)')
# axs[0,1].plot(time, Q_Exudall/dt , 'tab:brown')
# axs[0,1].set(xlabel='day of growth', ylabel='total exudation\nrate (mmol/day)')
# axs[1,1].plot(time, lengthTotall , 'tab:green')
# axs[1,1].set(xlabel='day of growth', ylabel='total plant\nlength (cm)')
# fig.tight_layout()
# plt.show()