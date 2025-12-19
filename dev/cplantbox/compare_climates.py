#!/usr/bin/env python3
"""
Climate Scenario Comparison Tool

Visualize and compare different climate configurations for CPlantBox simulations.
Usage: python compare_climates.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants for calculations
def theta2H(vg, theta):
    """Convert volumetric water content to pressure head (cm)"""
    thetar = vg[0]
    thetas = vg[1]
    alpha = vg[2]
    n = vg[3]
    nrev = 1/(1-1/n)
    H = -(((( (thetas - thetar)/(theta - thetar))**nrev) - 1)**(1/n))/alpha
    return H

def calc_vpd(T, RH):
    """Calculate vapor pressure deficit (kPa)"""
    # Saturation vapor pressure (kPa)
    es = 0.6112 * np.exp((17.67 * T)/(T + 243.5))
    # Actual vapor pressure
    ea = es * RH
    # VPD
    vpd = es - ea
    return vpd

def load_climate_files(climate_dir='climate'):
    """Load all climate JSON files"""
    climate_dir = Path(climate_dir)
    climates = {}

    for file in sorted(climate_dir.glob('*.json')):
        if file.name == 'README.md':
            continue
        with open(file, 'r') as f:
            data = json.load(f)
            scenario_name = file.stem
            climates[scenario_name] = data

    return climates

def calculate_derived_parameters(climates):
    """Calculate derived parameters for each climate scenario"""
    derived = {}

    for name, climate in climates.items():
        # VPD calculations
        vpd_day = calc_vpd(climate['Tday'], climate['RHday'])
        vpd_night = calc_vpd(climate['Tnight'], climate['RHnight'])

        # Soil water potential
        p_soil = theta2H(climate['vgSoil'], climate['thetaInit'])

        # Temperature range
        temp_range = climate['Tday'] - climate['Tnight']

        # Light intensity (convert to µmol/m²/s)
        light_umol = climate['Qday'] * 1e6

        derived[name] = {
            'vpd_day': vpd_day,
            'vpd_night': vpd_night,
            'vpd_mean': (vpd_day + vpd_night) / 2,
            'p_soil': p_soil,
            'temp_range': temp_range,
            'temp_mean': (climate['Tday'] + climate['Tnight']) / 2,
            'light_umol': light_umol,
            'co2_ppm': climate['co2'] * 1e6 / 0.00085 * 400  # Convert to ppm
        }

    return derived

def create_comparison_plot(climates, derived, output_file='climate_comparison.png'):
    """Create comprehensive comparison visualization"""

    scenario_names = list(climates.keys())
    n_scenarios = len(scenario_names)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Climate Scenario Comparison for CPlantBox Simulations', fontsize=16, fontweight='bold')

    # Colors for different scenarios
    colors = plt.cm.tab20(np.linspace(0, 1, n_scenarios))

    # 1. Temperature ranges
    ax = axes[0, 0]
    for i, name in enumerate(scenario_names):
        climate = climates[name]
        ax.barh(i, climate['Tday'] - climate['Tnight'],
                left=climate['Tnight'], color=colors[i], alpha=0.7)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Temperature (°C)')
    ax.set_title('Temperature Range (Night-Day)')
    ax.grid(axis='x', alpha=0.3)

    # 2. Light intensity
    ax = axes[0, 1]
    light_values = [derived[name]['light_umol'] for name in scenario_names]
    ax.barh(range(n_scenarios), light_values, color=colors, alpha=0.7)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('PPFD (µmol m⁻² s⁻¹)')
    ax.set_title('Daytime Light Intensity')
    ax.grid(axis='x', alpha=0.3)

    # 3. VPD
    ax = axes[0, 2]
    vpd_values = [derived[name]['vpd_day'] for name in scenario_names]
    ax.barh(range(n_scenarios), vpd_values, color=colors, alpha=0.7)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('VPD (kPa)')
    ax.set_title('Daytime Vapor Pressure Deficit')
    ax.grid(axis='x', alpha=0.3)

    # 4. Soil water content
    ax = axes[1, 0]
    theta_values = [climates[name]['thetaInit'] * 100 for name in scenario_names]
    ax.barh(range(n_scenarios), theta_values, color=colors, alpha=0.7)
    ax.axvline(x=5.9, color='red', linestyle='--', label='Residual (θr)')
    ax.axvline(x=45, color='blue', linestyle='--', label='Saturated (θs)')
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Soil Water Content (%)')
    ax.set_title('Initial Soil Moisture')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 5. Soil water potential
    ax = axes[1, 1]
    p_soil_values = [derived[name]['p_soil'] for name in scenario_names]
    ax.barh(range(n_scenarios), p_soil_values, color=colors, alpha=0.7)
    ax.axvline(x=-15000, color='red', linestyle='--', label='Wilting point')
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Soil Water Potential (cm)')
    ax.set_title('Soil Water Status')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 6. Relative Humidity
    ax = axes[1, 2]
    rh_day = [climates[name]['RHday'] * 100 for name in scenario_names]
    rh_night = [climates[name]['RHnight'] * 100 for name in scenario_names]
    x_pos = np.arange(n_scenarios)
    ax.barh(x_pos - 0.2, rh_day, 0.4, label='Day', color=colors, alpha=0.7)
    ax.barh(x_pos + 0.2, rh_night, 0.4, label='Night', color=colors, alpha=0.4)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_title('RH Day vs Night')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 7. CO2 concentration
    ax = axes[2, 0]
    co2_values = [derived[name]['co2_ppm'] for name in scenario_names]
    bars = ax.barh(range(n_scenarios), co2_values, color=colors, alpha=0.7)
    ax.axvline(x=400, color='green', linestyle='--', label='Ambient (~400 ppm)')
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('CO₂ (ppm)')
    ax.set_title('CO₂ Concentration')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 8. Atmospheric pressure
    ax = axes[2, 1]
    pressure_values = [climates[name]['Pair'] for name in scenario_names]
    ax.barh(range(n_scenarios), pressure_values, color=colors, alpha=0.7)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Pressure (hPa)')
    ax.set_title('Atmospheric Pressure')
    ax.grid(axis='x', alpha=0.3)

    # 9. Stress Index (composite)
    ax = axes[2, 2]
    # Simple stress index: normalized sum of stressors
    stress_scores = []
    for name in scenario_names:
        climate = climates[name]
        der = derived[name]

        # Temperature stress (optimal ~18°C)
        temp_stress = abs(der['temp_mean'] - 18) / 15  # normalized

        # Water stress
        if der['p_soil'] < -15000:
            water_stress = 1.0
        else:
            water_stress = max(0, (-der['p_soil'] - 100) / 14900)

        # VPD stress (optimal ~1 kPa)
        vpd_stress = max(0, (der['vpd_day'] - 1.0) / 2.0)

        # Light stress (optimal 800-1200)
        if der['light_umol'] < 800:
            light_stress = (800 - der['light_umol']) / 800
        elif der['light_umol'] > 1200:
            light_stress = (der['light_umol'] - 1200) / 800
        else:
            light_stress = 0

        # Combined stress (0 = no stress, 1 = extreme stress)
        total_stress = min(1.0, (temp_stress + water_stress + vpd_stress + light_stress) / 4)
        stress_scores.append(total_stress)

    bars = ax.barh(range(n_scenarios), stress_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_names, fontsize=8)
    ax.set_xlabel('Stress Index (0-1)')
    ax.set_title('Composite Stress Index')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    # Color bars based on stress level
    for i, (bar, stress) in enumerate(zip(bars, stress_scores)):
        if stress < 0.3:
            bar.set_color('green')
        elif stress < 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

    return fig

def print_summary_table(climates, derived):
    """Print a summary table of all scenarios"""
    print("\n" + "="*100)
    print("CLIMATE SCENARIO SUMMARY TABLE")
    print("="*100)

    # Header
    header = f"{'Scenario':<20} {'Temp(°C)':<12} {'Light':<10} {'θ(%)':<8} {'ψ(cm)':<10} {'VPD(kPa)':<10} {'CO2(ppm)':<10}"
    print(header)
    print("-"*100)

    # Data rows
    for name in sorted(climates.keys()):
        climate = climates[name]
        der = derived[name]

        temp_str = f"{climate['Tnight']:.1f}-{climate['Tday']:.1f}"
        light_str = f"{der['light_umol']:.0f}"
        theta_str = f"{climate['thetaInit']*100:.1f}"
        psi_str = f"{der['p_soil']:.0f}"
        vpd_str = f"{der['vpd_day']:.2f}"
        co2_str = f"{der['co2_ppm']:.0f}"

        row = f"{name:<20} {temp_str:<12} {light_str:<10} {theta_str:<8} {psi_str:<10} {vpd_str:<10} {co2_str:<10}"
        print(row)

    print("="*100 + "\n")

def main():
    """Main function"""
    print("CPlantBox Climate Scenario Comparison Tool")
    print("="*50)

    # Load climate files
    script_dir = Path(__file__).parent
    climate_dir = script_dir / 'climate'

    if not climate_dir.exists():
        climate_dir = Path('climate')

    print(f"Loading climate files from: {climate_dir}")
    climates = load_climate_files(climate_dir)
    print(f"Found {len(climates)} climate scenarios\n")

    # Calculate derived parameters
    derived = calculate_derived_parameters(climates)

    # Print summary table
    print_summary_table(climates, derived)

    # Create comparison plot
    output_file = script_dir / 'climate_comparison.png' if script_dir != Path('.') else 'climate_comparison.png'
    create_comparison_plot(climates, derived, output_file)

    print("\nComparison complete!")
    print("Use these scenarios with sim_phloem_flow.py:")
    print("  python sim_phloem_flow.py --weather-file climate/<scenario>.json")

if __name__ == '__main__':
    main()
