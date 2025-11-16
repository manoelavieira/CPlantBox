#!/bin/bash

set -e  # Exit on any error

echo "Starting heat wave experiment (1/3) ..."
python3 example8a_phloemFlow.py outputs/heat_wave_15d.csv config/baseline.json config/heat_wave.json
echo -e "Heat wave experiment successfully completed!\n"

echo "Starting drought air experiment (2/3) ..."
python3 example8a_phloemFlow.py outputs/drought_air_15d.csv config/baseline.json config/drought_air.json
echo -e "Drought air experiment successfully completed!\n"

echo "Starting nuclear winter experiment (3/3) ..."
python3 example8a_phloemFlow.py outputs/nuclear_winter_15d.csv config/baseline.json config/nuclear_winter.json
echo -e "Nuclear winter experiment successfully completed!\n"