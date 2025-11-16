import os
import sys

sys.path.append("../..")
sys.path.append("../../src/")
sys.path.append("../../modelparameter/functional")

import plantbox as pb
import numpy as np
import matplotlib.pyplot as plt
import visualisation.vtk_plot as vp

plant = pb.MappedPlant(seednum=0)
path = "../../modelparameter/structural/plant/"
name = "example1e"

depth = 60

plant.readParameters(path + name + ".xml")

# print(f"pb.seed")
# for p in plant.getOrganRandomParameter(pb.seed):
#     print(f"{p.subType}: {p.name}")

# print(f"pb.root")
# for p in plant.getOrganRandomParameter(pb.root):
#     print(f"{p.subType}: {p.name}")

print(f"pb.stem")
for p in plant.getOrganRandomParameter(pb.stem):
    print(f"{p.subType}: {p.name}")
    if (p.subType >= 3):
        print(p)

# print(f"pb.leaf")
# for p in plant.getOrganRandomParameter(pb.leaf):
#     print(f"{p.subType}: {p.name}")
#     # if (p.subType >= 2):
#     #     print(p)

# sdf = pb.SDF_PlantBox(np.Inf, np.Inf, depth) # creates a rectangular box (length, width, height)
# plant.setGeometry(sdf) # creates soil space to stop roots from growing out of the soil

verbose = False
plant.initialize(verbose)
plant.simulate(30, verbose)

# organs = plant.getOrgans(-1, True)
# print("Total organs:", len(organs))

# for organ in organs:
#     print("Organ Type:", pb.Organism.organTypeName(organ.organType()))
#     print("SubType:", organ.getParameter("subType"))
#     print("Length:", organ.getParameter("length"))

vp.plot_plant(plant, "subType")