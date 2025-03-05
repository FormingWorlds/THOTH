"""Physical constants and configuration values for rocky surface analysis."""

import numpy as np

# Physical constants (SI units)
C = 2.99792458e8      # Speed of light (m/s)
H = 6.6260693e-34     # Planck's constant (J s)
K = 1.3806503e-23     # Boltzmann constant (J/K)
R = 8.3144598         # Ideal gas constant (J/mol/K)
G = 6.67408e-11       # Gravitational constant (m^3/kg/s^2)

# Astronomical constants
RSUN = 695508e3       # Solar radius (m)
MSUN = 1.989e30       # Solar mass (kg)
RJUP = 71492e3        # Jupiter radius (m)
MJUP = 1.8986e27      # Jupiter mass (kg)
RE = 6.3781e6         # Earth radius (m)
ME = 5.972e24         # Earth mass (kg)
AU = 1.4959787066e11  # Astronomical Unit (m)
D2S = 86400           # Days to seconds conversion

# Configuration for spectral processing
WAVE_MIN = 300.0      # Minimum wavelength (nm)
WAVE_MAX = 1e5        # Maximum wavelength (nm)
WAVE_BINS = 300       # Number of wavelength bins

# Data processing configuration
SW_FOOTER = 20        # Footer lines to skip in shortwave data
LW_FOOTER = 6         # Footer lines to skip in longwave data

# File paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "output/processed"
FIGURE_DIR = "output/figures" 