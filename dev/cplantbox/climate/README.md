# Climate Configuration Files for CPlantBox Simulations

This directory contains climate configuration files for CPlantBox phloem flow simulations. Each file represents a distinct environmental scenario designed to test different aspects of plant physiology and stress responses.

---

## Available Climate Scenarios

### 1. baseline.json ✅
**Reference conditions** - Based on uqr7to14wet.py
- Temperature: 15.8-22°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 30% | VPD: ~1.3 kPa
- **Use for**: Control simulations, standard growing conditions

### 2. heat_wave.json 🔥
**Extreme heat + moderate drought** - Based on uqr21to28dry.py
- Temperature: 20.7-30.3°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 20% | VPD: ~2.8 kPa
- **Use for**: Heat stress, combined stress, climate change scenarios

### 3. drought_soil.json 💧
**Severe soil water limitation**
- Temperature: 16-23°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 15% | VPD: ~1.4 kPa
- **Use for**: Hydraulic stress, drought adaptation, root function

### 4. drought_air.json 🌬️
**Atmospheric drought (high VPD)**
- Temperature: 18-25°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 30% | VPD: ~2.5 kPa
- **Use for**: Stomatal response, transpiration stress

### 5. combined_stress.json ⚠️
**Multiple stressors**
- Temperature: 22-32°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 18% | VPD: ~3.6 kPa
- **Use for**: Extreme conditions, stress interaction effects

### 6. elevated_co2.json 🌍
**Future climate scenario**
- Temperature: 17-24°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 28% | CO₂: 550 ppm
- **Use for**: Climate change, CO₂ fertilization, water use efficiency

### 7. waterlogged.json 🌊
**Hypoxia from soil saturation**
- Temperature: 15-20°C | Light: 960 µmol m⁻²s⁻¹ | Soil θ: 43% | VPD: ~0.5 kPa
- **Use for**: Flooding tolerance, root oxygen limitation

### 8. nuclear_winter.json ❄️
**Extreme low light + cold**
- Temperature: 8-12°C | Light: 500 µmol m⁻²s⁻¹ | Soil θ: 35% | VPD: ~0.4 kPa
- **Use for**: Light limitation, cold stress, survival conditions

---

## Quick Reference

### Parameter Ranges Across Scenarios

| Parameter | Optimal | Moderate Stress | Severe Stress |
|-----------|---------|-----------------|---------------|
| Temp (°C) | 14-22 | 10-14 or 22-28 | <10 or >28 |
| Soil θ (%) | 30-35 | 20-25 | <20 or >40 |
| VPD (kPa) | 0.5-1.5 | 1.5-2.5 | >2.5 |
| Light (µmol m⁻²s⁻¹) | 800-1200 | 500-800 | <500 |

### Expected Responses

| Scenario | Assimilation | Transpiration | Stress Level |
|----------|-------------|---------------|--------------|
| baseline | High | Moderate | Low |
| heat_wave | Reduced | High | High |
| drought_soil | Low | Very Low | Very High |
| drought_air | Moderate | High | Moderate |
| combined_stress | Very Low | Low | Extreme |
| elevated_co2 | Enhanced | Reduced | Low |
| waterlogged | Reduced | Moderate | Moderate-High |
| nuclear_winter | Very Low | Very Low | High |

---

## File Format

Each JSON file contains:

- **vgSoil**: Van Genuchten soil parameters [θr, θs, α, n, Ks]
- **Qnight/Qday**: Light intensity (mol m⁻² s⁻¹)
- **Tnight/Tday**: Temperature (°C)
- **RHnight/RHday**: Relative humidity (fraction 0-1)
- **Pair**: Atmospheric pressure (hPa)
- **thetaInit**: Initial soil water content (fraction)
- **co2**: CO₂ concentration (mol mol⁻¹, default ~400 ppm)

---

## Usage

### Basic Usage
```bash
# Run single scenario
python sim_phloem_flow.py --weather-file climate/baseline.json --phloem-dir data/baseline

# Run heat stress
python sim_phloem_flow.py --weather-file climate/heat_wave.json --phloem-dir data/heat
```

### Comparative Studies
```bash
# Single factor: Water stress
python sim_phloem_flow.py --weather-file climate/baseline.json --phloem-dir data/control
python sim_phloem_flow.py --weather-file climate/drought_soil.json --phloem-dir data/drought

# Single factor: Temperature stress
python sim_phloem_flow.py --weather-file climate/baseline.json --phloem-dir data/control
python sim_phloem_flow.py --weather-file climate/heat_wave.json --phloem-dir data/heat

# Stress interactions
python sim_phloem_flow.py --weather-file climate/drought_soil.json --phloem-dir data/drought_only
python sim_phloem_flow.py --weather-file climate/heat_wave.json --phloem-dir data/heat_only
python sim_phloem_flow.py --weather-file climate/combined_stress.json --phloem-dir data/combined
```

---

## Reference Code Mappings

### uqr7to14wet.py → baseline.json
```python
# Reference Python code parameters:
Tmin/Tmax = 15.8/22°C
Qmax = 960e-6 mol m⁻²s⁻¹
θ_init = 30%
specificHumidity = 0.0097 kg/kg
Pair = 1010 hPa
```

### uqr21to28dry.py → heat_wave.json
```python
# Reference Python code parameters:
Tmin/Tmax = 20.7/30.27°C
Qmax = 960e-6 mol m⁻²s⁻¹
θ_init = 20%
specificHumidity = 0.0111 kg/kg
Pair = 1070 hPa
```

### Soil Water Potential Calculation
```python
# Van Genuchten equation (ψ in cm):
def theta2H(vg, theta):
    thetar, thetas, alpha, n = vg[0], vg[1], vg[2], vg[3]
    nrev = 1/(1-1/n)
    H = -(((((thetas - thetar)/(theta - thetar))**nrev) - 1)**(1/n))/alpha
    return H

# Examples:
# baseline (θ=0.30): ψ ≈ -187 cm (well-watered)
# heat_wave (θ=0.20): ψ ≈ -1040 cm (moderate drought)
# drought_soil (θ=0.15): ψ ≈ -3500 cm (severe drought)
```

---

## Physiological Thresholds

### Temperature (°C)
- **Optimal:** 15-25°C
- **Heat stress:** >28°C
- **Cold stress:** <10°C

### VPD (kPa)
- **Optimal:** 0.5-1.5 kPa
- **Moderate stress:** 1.5-2.5 kPa
- **High stress:** >2.5 kPa

### Soil Water Potential (cm)
- **Field capacity:** ~-100 cm
- **Moderate stress:** -500 to -1000 cm
- **Severe stress:** -1000 to -5000 cm
- **Wilting point:** -15000 cm

### Light (µmol m⁻²s⁻¹)
- **Light compensation:** ~50-100
- **Light saturation (wheat):** ~1200-1500
- **Full sun:** ~2000

---

## Recommended Combinations for GNN Training

### Core 4 Scenarios (Maximum Coverage)
1. **baseline.json** - Reference conditions
2. **heat_wave.json** - High temperature + moderate drought
3. **drought_soil.json** - Severe water stress
4. **nuclear_winter.json** - Low light + cold stress

### Extended 5 Scenarios (Include Future Climate)
Add: **elevated_co2.json** - Climate change scenario

These provide orthogonal coverage of:
- Temperature: 8-30°C range
- Water: 15-35% soil moisture
- Light: 500-960 µmol m⁻²s⁻¹
- CO₂: 400-550 ppm

---

## Notes

- All scenarios use consistent Van Genuchten soil parameters
- Day/night cycles are sinusoidal in simulation
- Parameters validated against wheat physiology (Giraud et al. 2023)
- Reference implementations: uqr7to14wet.py, uqr21to28dry.py

---

## Citation

When using these climate scenarios:
```
Climate configurations based on:
- CPlantBox framework (Schnepf et al.)
- Reference codes: uqr7to14wet.py, uqr21to28dry.py
- Physiological parameters: Giraud et al. (2023)
```
