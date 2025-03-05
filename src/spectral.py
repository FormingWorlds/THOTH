"""Spectral processing and calculation functions."""

import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Tuple, Optional, Dict
import warnings
import logging

from .constants import (H, C, K, WAVE_MIN, WAVE_MAX, WAVE_BINS)
from .data_loader import SurfaceData

class SpectralCalculator:
    """Handles spectral calculations and transformations."""
    
    def __init__(self, 
                 wave_min: float = WAVE_MIN,
                 wave_max: float = WAVE_MAX,
                 wave_bins: int = WAVE_BINS):
        """
        Initialize spectral calculator.
        
        Args:
            wave_min: Minimum wavelength in nm
            wave_max: Maximum wavelength in nm
            wave_bins: Number of wavelength bins
        """
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.wave_bins = wave_bins
        self.setup_wavelength_grid()
        
    def setup_wavelength_grid(self):
        """Initialize wavelength grid for calculations."""
        obs_pts = np.logspace(np.log10(self.wave_min),
                            np.log10(self.wave_max),
                            self.wave_bins)
        self.obs_bc = (obs_pts[1:] + obs_pts[:-1]) * 0.5  # bin centers
        self.obs_bw = obs_pts[1:] - obs_pts[:-1]          # bin widths
        
    @staticmethod
    def planck(wav: float, tmp: float) -> float:
        """Evaluate the Planck function at given wavelength and temperature.
        
        Args:
            wav: Wavelength in nm
            tmp: Temperature in K
            
        Returns:
            Spectral radiance in W/m²/nm
        """
        # Constants (SI units)
        h_pl = 6.62607015e-34  # Planck constant
        k_B = 1.380649e-23     # Boltzmann constant
        c_vac = 299792458.0    # Speed of light
        
        # Convert wavelength to meters
        wav_m = wav * 1.0e-9
        
        # Calculate Planck function [W m⁻² sr⁻¹ m⁻¹]
        numerator = 2.0 * h_pl * c_vac * (c_vac / wav_m**5.0)
        denominator = np.exp(h_pl * c_vac / (wav_m * k_B * tmp)) - 1.0
        flux = numerator / denominator
        
        # Integrate over hemisphere and convert to nm⁻¹
        flux = flux * np.pi * 1.0e-9  # [W m⁻² nm⁻¹]
        
        return flux

    def process_surface_data(self, 
                           surface: SurfaceData,
                           temperature: float) -> Dict[str, np.ndarray]:
        """
        Process surface data to calculate emission and reflection.
        
        Args:
            surface: SurfaceData object containing spectral data
            temperature: Surface temperature in K
            
        Returns:
            Dictionary containing processed spectral components
        """
        # Basic input validation
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        # Calculate single-scattering albedo with safety checks
        w = np.clip(1.0 - surface.reflectance**2, 0, 1)
        
        # Calculate thermal emission with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            bb_emission = self.planck(surface.wavelength, temperature)
        
        thermal_emission = (1.0 - w) * bb_emission
        
        # Replace any remaining invalid values with zeros
        thermal_emission = np.nan_to_num(thermal_emission, 0.0)
        w = np.nan_to_num(w, 0.0)
        
        # Get wavelength range for interpolation
        valid_range = (surface.wavelength >= self.wave_min) & (surface.wavelength <= self.wave_max)
        if not np.any(valid_range):
            logging.warning("No data points in target wavelength range. Extrapolating...")
        
        # Ensure strictly increasing wavelength for interpolation
        sort_idx = np.argsort(surface.wavelength)
        wavelength_sorted = surface.wavelength[sort_idx]
        emission_sorted = thermal_emission[sort_idx]
        w_sorted = w[sort_idx]
        
        # Remove any duplicate wavelength values
        unique_idx = np.concatenate(([True], np.diff(wavelength_sorted) > 0))
        wavelength_unique = wavelength_sorted[unique_idx]
        emission_unique = emission_sorted[unique_idx]
        w_unique = w_sorted[unique_idx]
        
        try:
            # Interpolate to observation wavelength grid
            emission_interp = PchipInterpolator(wavelength_unique, emission_unique)
            albedo_interp = PchipInterpolator(wavelength_unique, w_unique)
            
            # Calculate on observation grid
            emission_obs = emission_interp(self.obs_bc)
            albedo_obs = albedo_interp(self.obs_bc)
            
            # Replace any remaining invalid values
            emission_obs = np.nan_to_num(emission_obs, 0.0)
            albedo_obs = np.nan_to_num(albedo_obs, 0.0)
            
        except ValueError as e:
            logging.error(f"Interpolation failed: {str(e)}")
            # Fall back to simple nearest-neighbor interpolation
            logging.warning("Falling back to nearest-neighbor interpolation")
            emission_obs = np.interp(self.obs_bc, wavelength_unique, emission_unique)
            albedo_obs = np.interp(self.obs_bc, wavelength_unique, w_unique)
        
        return {
            'wavelength': self.obs_bc,
            'emission': emission_obs,
            'albedo': albedo_obs,
            'bin_widths': self.obs_bw
        } 