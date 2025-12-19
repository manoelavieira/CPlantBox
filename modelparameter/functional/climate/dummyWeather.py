import numpy as np
import matplotlib.pyplot as plt
import json

"""avoid division per 0 during post processing"""
def div0(a, b, c):
    return np.divide(a, b, out=np.full(len(a), c), where=b!=0)

def div0f(a, b, c):
    if b != 0:
        return a/b
    else:
        return a/c

"""compute soil water potential from water content and VanGenouchten parameters"""
def theta2H(vg, theta): # (-) to cm
    thetar = vg[0]
    thetas = vg[1]
    alpha = vg[2]
    n = vg[3]
    nrev = 1/(1-1/n)
    H = -(((( (thetas - thetar)/(theta - thetar))**nrev) - 1)**(1/n))/alpha
    return(H) # cm

def sinusoidal(t):
    return (np.sin(np.pi*t*2)+1)/2

"""get environmental conditions"""
def weather(simDuration):
    vgSoil = [0.059, 0.45, 0.00644, 1.503, 1] # van Gemuchten parameters
    Qnight = 0  # night light [mol m-2 s-1]
    Qday = 960e-6  # peak daytime light [mol m-2 s-1], matching typical PAR measurements
    Tnight = 15.8  # night temperature [°C]
    Tday = 22  # day temperature [°C]
    RHnight = 0.8  # night relative humidity [-]
    RHday = 0.5  # day relative humidity [-]
    Pair = 1010.00  # atmospheric pressure [hPa]
    thetaInit = 30/100  # initial soil water content [-]

    coefhours = sinusoidal(simDuration)  # sinusoidal coefficient (0-1) for daily variation
    RH = RHnight + (RHday - RHnight) * coefhours
    Tair = Tnight + (Tday - Tnight) * coefhours
    Q_ = Qnight + (Qday - Qnight) * coefhours  # light follows sinusoidal pattern
    co2 = 850e-6  # co2 partial pressure at leaf surface [mol mol-1]
    es = 6.112 * np.exp((17.67 * Tair)/(Tair + 243.5))  # saturation vapor pressure [hPa]
    ea = es * RH  # actual vapor pressure [hPa]

    pmean = theta2H(vgSoil, thetaInit)  # average soil water potential [cm]

    weatherVar = {'Tair': Tair, 'Qlight': Q_, 'ea': ea, 'es': es,
                  'co2': co2, 'RH': RH, 'p_mean': pmean, 'vg': vgSoil}

    return weatherVar

def required_param(config, key):
    if key not in config:
        raise KeyError(f"Missing required weather configuration parameter: '{key}'")
    return config[key]

"""get environmental conditions"""
def weather_custom(simDuration, config):
    vgSoil = required_param(config, "vgSoil")
    Qnight = required_param(config, "Qnight")  # night light [mol m-2 s-1]
    Qday = required_param(config, "Qday")  # peak daytime light [mol m-2 s-1]
    Tnight = required_param(config, "Tnight")  # night temperature [°C]
    Tday = required_param(config, "Tday")  # day temperature [°C]
    RHnight = required_param(config, "RHnight")  # night relative humidity [-]
    RHday = required_param(config, "RHday")  # day relative humidity [-]
    Pair = required_param(config, "Pair")  # atmospheric pressure [hPa]
    thetaInit = required_param(config, "thetaInit")  # soil water content [-]
    co2 = required_param(config, "co2")  # co2 partial pressure at leaf surface [mol mol-1]

    coefhours = sinusoidal(simDuration)  # sinusoidal coefficient (0-1) for daily variation
    RH = RHnight + (RHday - RHnight) * coefhours
    Tair = Tnight + (Tday - Tnight) * coefhours
    Q_ = Qnight + (Qday - Qnight) * coefhours  # light follows sinusoidal pattern
    es = 6.112 * np.exp((17.67 * Tair) / (Tair + 243.5))  # saturation vapor pressure [hPa]
    ea = es * RH  # actual vapor pressure [hPa]

    pmean = theta2H(vgSoil, thetaInit)  # average soil water potential (pressure head, in cm)

    weatherVar = {
        'Tair': Tair,
        'Qlight': Q_,
        'ea': ea,
        'es': es,
        'co2': co2,
        'RH': RH,
        'p_mean': pmean,
        'vg': vgSoil
    }

    return weatherVar

"""utility to load config from JSON file"""
def load_weather_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)