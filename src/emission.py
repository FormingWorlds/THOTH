"""Planetary emission calculation module."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import astropy.units as u
import logging
from scipy.optimize import root_scalar
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from .constants import (RSUN, AU, RE)
from .spectral import SpectralCalculator
from .data_loader import SurfaceData

@dataclass
class PlanetarySystem:
    """Container for planetary system parameters."""
    star_temp: float           # Stellar temperature (K)
    star_radius: float        # Stellar radius (solar radii)
    planet_radius: float      # Planet radius (earth radii)
    semi_major_axis: float    # Semi-major axis (AU)
    
    @property
    def instellation(self) -> float:
        """Calculate incident stellar flux in W/m².
        
        Uses Stefan-Boltzmann law and geometric dilution factor:
        F = σT⁴ * (R_star/a)²
        where:
        - σ is Stefan-Boltzmann constant
        - T is stellar temperature
        - R_star is stellar radius (converted from solar radii)
        - a is semi-major axis (converted from AU)
        """
        sigma = 5.670374419e-8  # Stefan-Boltzmann constant
        
        # Convert units:
        # - star_radius from solar radii to meters
        # - semi_major_axis from AU to meters
        r_star_m = self.star_radius * RSUN
        a_m = self.semi_major_axis * AU
        
        # Calculate instellation
        return sigma * (self.star_temp**4) * (r_star_m / a_m)**2
    
    @property
    def equilibrium_temp(self) -> float:
        """Calculate equilibrium temperature in K.
        
        T_eq = (F*(1-A)/(4σ))^(1/4)
        where:
        - F is instellation
        - A is albedo (assumed 0.3 for now)
        - σ is Stefan-Boltzmann constant
        """
        sigma = 5.670374419e-8
        albedo = 0.3  # Typical planetary albedo
        return (self.instellation * (1 - albedo) / (4 * sigma))**0.25

class EmissionCalculator:
    """Calculate planetary emission spectra."""
    
    def __init__(self, system: PlanetarySystem, spectral_calc: 'SpectralCalculator'):
        """Initialize calculator with system parameters.
        
        Args:
            system: PlanetarySystem object with stellar and planetary parameters
            spectral_calc: SpectralCalculator object for spectral calculations
        """
        self.system = system
        self.spectral_calc = spectral_calc
        self.setup_grids()
        self.load_stellar_spectrum()
        
    def setup_grids(self, grid_points: int = 6):
        """Setup calculation grids for dayside hemisphere.
        
        Args:
            grid_points: Number of points in latitude/longitude grid
        """
        self.longitudes = np.linspace(-np.pi/2, np.pi/2, grid_points)
        self.latitudes = np.linspace(-np.pi/2, np.pi/2, grid_points)
        
        # Store old names for compatibility
        self.day_lons = self.longitudes
        self.day_lats = self.latitudes
        
    def load_stellar_spectrum(self):
        """Load stellar spectrum (currently using blackbody approximation)."""
        # Calculate stellar flux at star surface
        self.stellar_flux = np.array([
            self.spectral_calc.planck(wav, self.system.star_temp)
            for wav in self.spectral_calc.obs_bc
        ])
        
        # Scale to flux at planet's orbit using proper unit conversions
        r_star_m = self.system.star_radius * RSUN
        a_m = self.system.semi_major_axis * AU
        self.stellar_flux_at_planet = self.stellar_flux * (r_star_m / a_m)**2
        
    def calculate_fluxes(self, wav: float, wid: float, tmp: float, 
                        instellation_scale: float) -> Dict[str, float]:
        """Calculate spectral fluxes at a surface point.
        
        Args:
            wav: Wavelength in nm
            wid: Wavelength bin width in nm
            tmp: Surface temperature in K
            instellation_scale: cos(solar zenith angle) = mu0
        """
        # Get reflectance and calculate gamma
        reflectance = np.interp(wav, 
                               self.spectral_calc.wavelength,
                               self.spectral_calc.reflectance)
        gamma = np.sqrt(1 - reflectance)  # sqrt(1 - albedo)
        mu0 = max(instellation_scale, 0.0)  # Ensure non-negative
        
        # Calculate incident stellar flux first (scaled to planet distance)
        stellar = (mu0 * 
                  np.interp(wav, self.spectral_calc.obs_bc, self.stellar_flux_at_planet) * 
                  wid)
        
        # Calculate directional-hemispheric reflectance (for reflection)
        r_h = (1 - gamma) / (1 + 2 * gamma * mu0) if mu0 > 0 else 0.0
        
        # Calculate reflected flux - cannot exceed incident flux
        reflected = min(stellar * r_h, stellar)
        
        # Calculate spherical reflectance (for emission)
        r_0 = (1 - gamma) / (1 + gamma)
        r_s = r_0 * (1 - (1./3.) * (gamma / (1 + gamma)))
        
        # Calculate emissivity from spherical reflectance
        emissivity = 1.0 - r_s
        
        # Calculate thermal emission in all directions (hemisphere-integrated)
        thermal_total = self.spectral_calc.planck(wav, tmp) * wid * emissivity * np.pi
        
        # Calculate thermal emission in viewing direction (no mu0 here, will be applied through dA_proj)
        thermal_observed = self.spectral_calc.planck(wav, tmp) * wid * emissivity
        
        return {
            "LW_UP_TOTAL": thermal_total,  # Total thermal emission over hemisphere
            "LW_UP_OBSERVED": thermal_observed,  # Base thermal emission (viewing angle applied through dA_proj)
            "LW_DN": 0.0,  # No downward thermal radiation
            "SW_DN": stellar * (1 - r_h),  # Absorbed using dir-hem reflectance
            "SW_UP": reflected,  # Reflected light (capped at incident flux)
            "SW_IN": stellar  # Incident stellar flux
        }
        
    def solve_radiative_equilibrium(self, instellation_scale: float) -> float:
        """Solve for temperature satisfying radiative equilibrium.
        
        Args:
            instellation_scale: cos(solar zenith angle)
        """
        def residual(tmp: float) -> float:
            """Calculate net flux for given temperature."""
            up = down = 0.0
            try:
                # Calculate fluxes at each wavelength
                for bc, bw in zip(self.spectral_calc.obs_bc, self.spectral_calc.obs_bw):
                    fluxes = self.calculate_fluxes(bc, bw, tmp, instellation_scale)
                    up += fluxes["LW_UP_TOTAL"]  # Only upward thermal emission
                    down += fluxes["SW_DN"]  # Only absorbed stellar flux
                
                # Print diagnostic info
                logging.debug(f"T={tmp:.1f}K: up={up:.2e}, down={down:.2e}, net={down-up:.2e}")
                return down - up
                
            except RuntimeWarning as e:
                logging.warning(f"Runtime warning in residual at T={tmp:.1f}K: {str(e)}")
                return np.inf
            except Exception as e:
                logging.error(f"Error in residual at T={tmp:.1f}K: {str(e)}")
                return np.inf
            
        # Start with equilibrium temperature as initial guess
        x0 = self.system.equilibrium_temp
        
        # For points with very low instellation, use a lower temperature guess
        if instellation_scale < 0.1:
            x0 *= instellation_scale**0.25  # T ∝ F^(1/4)
        
        eps = 1e-4
        p1 = x0 * (1 + eps)
        p1 += eps if p1 >= 0 else -eps
        
        try:
            # Use secant method with initial points close to equilibrium temp
            sol = root_scalar(residual, x0=x0, x1=p1, 
                            method='secant',
                            xtol=1e-2,  # Looser tolerance
                            maxiter=50)  # More iterations
            
            temp = float(sol.root)
            
            # Sanity check on temperature
            if temp > 2 * self.system.equilibrium_temp:
                logging.warning(f"Temperature {temp:.1f}K seems too high! Using equilibrium temp.")
                return self.system.equilibrium_temp
                
            logging.debug(f"Found equilibrium T = {temp:.1f}K for cos(sza) = {instellation_scale:.3f}")
            return temp
            
        except ValueError as e:
            logging.warning(f"Root finding failed: {str(e)}")
            logging.warning(f"Using equilibrium temperature scaled by instellation...")
            return x0
        except Exception as e:
            logging.error(f"Unexpected error in temperature solution: {str(e)}")
            return x0
            
    def calculate_emission(self, surface: SurfaceData, 
                         include_reflection: bool = True, show_plot: bool = False) -> Dict[str, np.ndarray]:
        """Calculate emission spectrum.
        
        Args:
            surface: SurfaceData object containing spectral data
            include_reflection: Whether to include reflected light
            
        Returns:
            Dictionary containing wavelengths and emission components
        """
        # Store surface data in spectral calculator
        self.spectral_calc.wavelength = surface.wavelength
        self.spectral_calc.reflectance = surface.reflectance
        
        # Calculate mean emissivity and reflectivity
        reflectance = self.spectral_calc.reflectance
        gamma = np.sqrt(1 - reflectance)
        r_0 = (1 - gamma) / (1 + gamma)
        r_s = r_0 * (1 - (1./3.) * (gamma / (1 + gamma)))
        emissivity = 1.0 - r_s
        
        mean_reflectance = np.mean(reflectance)
        mean_emissivity = np.mean(emissivity)
        
        logging.info(f"Mean surface reflectivity: {mean_reflectance:.3f}")
        logging.info(f"Mean surface emissivity: {mean_emissivity:.3f}")
        
        # Initialize separate arrays for total and observed fluxes
        thermal_flux_total = np.zeros_like(self.spectral_calc.obs_bc)  # Total thermal over hemisphere
        thermal_flux_observed = np.zeros_like(self.spectral_calc.obs_bc)  # Thermal in viewing direction
        reflected_flux = np.zeros_like(self.spectral_calc.obs_bc)
        incident_flux = np.zeros_like(self.spectral_calc.obs_bc)
        
        # Initialize area counters
        total_area = 0.0
        projected_area = 0.0
        
        # Initialize integrated flux counters
        total_thermal_total = 0.0
        total_thermal_observed = 0.0
        total_reflected = 0.0
        total_incident = 0.0
        
        # Store temperatures for mapping
        self.temperatures = np.zeros((len(self.longitudes), len(self.latitudes)))
        
        # Calculate emission across visible hemisphere
        for i, lon in enumerate(self.longitudes):
            for j, lat in enumerate(self.latitudes):
                cos_sza = np.cos(lon) * np.cos(lat)
                
                if cos_sza > 0:  # Only calculate for dayside
                    # Calculate local temperature
                    temp = self.solve_radiative_equilibrium(cos_sza)
                    self.temperatures[i,j] = temp
                    
                    # Calculate area elements
                    dA = np.cos(lat)  # Area element on unit sphere
                    dA_proj = dA * cos_sza  # Projected area for incoming/viewing
                    
                    # Accumulate areas
                    total_area += dA
                    projected_area += dA_proj
                    
                    # Calculate local fluxes
                    local_thermal_total = 0.0
                    local_thermal_observed = 0.0
                    local_absorbed = 0.0
                    local_reflected = 0.0
                    local_incident = 0.0
                    
                    # Calculate fluxes at each wavelength
                    for k, (bc, bw) in enumerate(zip(self.spectral_calc.obs_bc, self.spectral_calc.obs_bw)):
                        fluxes = self.calculate_fluxes(bc, bw, temp, cos_sza)
                        
                        # Add to spectral arrays with proper area weighting
                        thermal_flux_total[k] += fluxes["LW_UP_TOTAL"] * dA / bw  # Use full area for total
                        thermal_flux_observed[k] += fluxes["LW_UP_OBSERVED"] * dA_proj / bw  # Use projected for observed
                        if include_reflection:
                            reflected_flux[k] += fluxes["SW_UP"] * dA_proj / bw  # Use projected for reflection
                        incident_flux[k] += fluxes["SW_IN"] * dA_proj / bw  # Use projected for incident
                        
                        # Accumulate total fluxes
                        local_thermal_total += fluxes["LW_UP_TOTAL"]
                        local_thermal_observed += fluxes["LW_UP_OBSERVED"]
                        local_absorbed += fluxes["SW_DN"]
                        local_reflected += fluxes["SW_UP"]
                        local_incident += (fluxes["SW_DN"] + fluxes["SW_UP"])
                    
                    # Add to integrated totals (with proper area elements)
                    total_thermal_total += local_absorbed * dA_proj  # Use absorbed flux instead of thermal emission
                    total_thermal_observed += local_thermal_observed * dA_proj  # Use projected area for observed thermal
                    total_reflected += local_reflected * dA_proj  # Reflected uses projected area
                    total_incident += local_incident * dA_proj  # Incident uses projected area
                    
                    # Print local energy balance
                    logging.info(f"\nPoint ({np.rad2deg(lon):.1f}°, {np.rad2deg(lat):.1f}°):")
                    logging.info(f"Temperature: {temp:.1f}K")
                    logging.info(f"cos(sza): {cos_sza:.3f}")
                    logging.info(f"Local incident flux: {local_incident:.2e} W/m²")
                    logging.info(f"Local absorbed flux: {local_absorbed:.2e} W/m²")
                    logging.info(f"Local thermal emission (total): {local_thermal_total:.2e} W/m²")
                    logging.info(f"Local thermal emission (observed): {local_thermal_observed:.2e} W/m²")
                    logging.info(f"Local reflected flux: {local_reflected:.2e} W/m²")
                    logging.info(f"Local energy balance ratio: {local_thermal_total/local_absorbed:.3f}")
                
                else:
                    self.temperatures[i,j] = 0.0  # Night side
        
        # Normalize fluxes by appropriate areas
        thermal_flux_total /= total_area  # Total thermal per unit area
        thermal_flux_observed /= projected_area  # Observed thermal per projected area
        reflected_flux /= projected_area  # Reflected per projected area
        incident_flux /= projected_area  # Incident per projected area
        
        # Calculate total fluxes (all per projected area)
        thermal_total = total_thermal_total / projected_area
        thermal_observed = total_thermal_observed / projected_area
        reflected_total = total_reflected / projected_area
        incident_total = total_incident / projected_area
        
        # Total outgoing flux (per projected area)
        total_outgoing = thermal_observed + reflected_total  # Use observed thermal, not total
        
        # Add more detailed logging for debugging
        logging.info("\nSurface Properties:")
        logging.info(f"Mean surface reflectivity: {mean_reflectance:.3f}")
        logging.info(f"Mean surface emissivity: {mean_emissivity:.3f}")
        
        logging.info("\nArea Diagnostics:")
        logging.info(f"Total area: {total_area:.3f}")
        logging.info(f"Projected area: {projected_area:.3f}")
        logging.info(f"Area ratio (total/projected): {total_area/projected_area:.3f}")
        
        logging.info("\nGlobal Energy Balance:")
        logging.info(f"Total incoming stellar flux: {incident_total:.2e} W/m² (per projected area)")
        logging.info(f"Total absorbed flux: {thermal_total:.2e} W/m² (per projected area)")
        logging.info(f"Total reflected flux: {reflected_total:.2e} W/m² (per projected area)")
        logging.info(f"Sum (absorbed + reflected): {thermal_total + reflected_total:.2e} W/m²")
        logging.info(f"Energy balance (absorbed + reflected)/incoming: {(thermal_total + reflected_total)/incident_total:.3f}")
        logging.info(f"Effective albedo (reflected/incoming): {reflected_total/incident_total:.3f}")
        
        logging.info("\nViewing Direction Fluxes:")
        logging.info(f"Total thermal emission: {thermal_total:.2e} W/m² (integrated over hemisphere)")
        logging.info(f"Expected observed thermal: {thermal_total/np.pi:.2e} W/m² (total/π)")
        logging.info(f"Actual observed thermal: {thermal_observed:.2e} W/m² (in viewing direction)")
        logging.info(f"Observed reflected: {reflected_total:.2e} W/m² (in viewing direction)")
        logging.info(f"Total observed flux: {total_outgoing:.2e} W/m² (thermal + reflected)")
        logging.info(f"Viewing ratio (observed/incoming): {total_outgoing/incident_total:.3f}")
        
        # Check if energy balance is significantly violated
        absorbed_plus_reflected = thermal_total + reflected_total
        if abs(absorbed_plus_reflected/incident_total - 1.0) > 0.1:  # Allow 10% margin for numerical error
            logging.warning("\nEnergy conservation violated! Total absorbed + reflected differs from incoming by more than 10%")
        
        # Check if viewing direction flux makes sense
        expected_thermal_observed = thermal_total / np.pi  # Expect roughly total/pi for hemisphere average
        if abs(thermal_observed/expected_thermal_observed - 1.0) > 0.2:  # Allow 20% margin for discretization
            logging.warning("\nObserved thermal emission differs significantly from expected hemisphere average")
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Temperature map
        lon_deg = np.rad2deg(self.longitudes)
        lat_deg = np.rad2deg(self.latitudes)
        lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)
        im = axes[0,0].pcolormesh(lon_grid, lat_grid, self.temperatures.T, 
                           shading='auto', cmap='RdYlBu_r')
        plt.colorbar(im, ax=axes[0,0], label='Temperature (K)')
        axes[0,0].set_xlabel('Longitude (degrees)')
        axes[0,0].set_ylabel('Latitude (degrees)')
        axes[0,0].set_title('Surface Temperature Distribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Local energy balance components
        wavelengths = self.spectral_calc.obs_bc / 1000  # Convert to microns
        axes[0,1].plot(wavelengths, incident_flux, 'g-', label='Incident')
        axes[0,1].plot(wavelengths, thermal_flux_total, 'r-', label='Thermal (All Directions)')
        if include_reflection:
            axes[0,1].plot(wavelengths, reflected_flux, 'b-', label='Reflected')
        axes[0,1].set_xlabel('Wavelength (μm)')
        axes[0,1].set_ylabel('Spectral Flux (W/m²/nm)')
        axes[0,1].set_title('Local Energy Balance Components')
        axes[0,1].set_xlim(0, 20)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Observed emission components
        axes[1,0].plot(wavelengths, incident_flux, 'g-', label='Incoming Stellar')
        axes[1,0].plot(wavelengths, thermal_flux_observed, 'r-', label='Observed Thermal')
        if include_reflection:
            axes[1,0].plot(wavelengths, reflected_flux, 'b-', label='Observed Reflected')
        axes[1,0].plot(wavelengths, thermal_flux_observed + reflected_flux, 'k-', label='Total Observed')
        axes[1,0].set_xlabel('Wavelength (μm)')
        axes[1,0].set_ylabel('Observed Flux (W/m²/nm)')
        axes[1,0].set_title('Observed Emission Components')
        axes[1,0].set_xlim(0, 20)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Planet-star contrast (convert radii to meters first)
        planet_radius_m = self.system.planet_radius * RE
        star_radius_m = self.system.star_radius * RSUN
        radius_ratio_squared = (planet_radius_m / star_radius_m)**2
        
        # Use original stellar flux (not diluted by distance) for contrast
        fp_fs = 1e6 * radius_ratio_squared * \
               (thermal_flux_observed + reflected_flux) / self.stellar_flux
        thermal_contrast = 1e6 * radius_ratio_squared * \
                         thermal_flux_observed / self.stellar_flux
        reflected_contrast = 1e6 * radius_ratio_squared * \
                           reflected_flux / self.stellar_flux
        
        axes[1,1].plot(wavelengths, fp_fs, 'k-', label='Total', linewidth=2)
        axes[1,1].plot(wavelengths, thermal_contrast, 'r--', label='Thermal')
        if include_reflection:
            axes[1,1].plot(wavelengths, reflected_contrast, 'b--', label='Reflected')
        axes[1,1].set_xlabel('Wavelength (μm)')
        axes[1,1].set_ylabel('Planet/Star Contrast (ppm)')
        axes[1,1].set_title('Planet-Star Contrast')
        axes[1,1].set_xlim(0, 20)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # Add text showing max contrast
        max_contrast = np.max(fp_fs)
        max_wavelength = wavelengths[np.argmax(fp_fs)]
        axes[1,1].text(0.05, 0.95, f'Max contrast: {max_contrast:.1f} ppm\nat λ = {max_wavelength:.1f} μm', 
                      transform=axes[1,1].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        # Return observed quantities for contrast calculation
        return {
            'wavelength': self.spectral_calc.obs_bc,
            'thermal_emission': thermal_flux_observed,  # Use observed thermal
            'reflected_light': reflected_flux,
            'total_emission': thermal_flux_observed + reflected_flux,
            'fp_fs': fp_fs
        }

def calculate_planet_star_contrast(emission_result: Dict[str, np.ndarray],
                                 system: PlanetarySystem) -> np.ndarray:
    """Calculate planet-to-star flux ratio."""
    logging.info("Calculating planet-star contrast")
    
    # Return pre-calculated contrast if available
    if 'fp_fs' in emission_result:
        return emission_result['fp_fs']
    
    # Otherwise calculate it
    planet_flux = emission_result['total_emission']
    wavelengths = emission_result['wavelength']
    
    # Calculate stellar surface flux (not diluted by distance)
    stellar_flux = np.array([SpectralCalculator.planck(wl, system.star_temp) 
                            for wl in wavelengths])
    
    # Convert radii to meters for proper ratio
    planet_radius_m = system.planet_radius * RE
    star_radius_m = system.star_radius * RSUN
    radius_ratio_squared = (planet_radius_m / star_radius_m)**2
    
    # Calculate contrast in ppm
    return 1e6 * radius_ratio_squared * planet_flux / stellar_flux 