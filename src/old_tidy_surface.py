# Standard library imports
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, NamedTuple

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import PchipInterpolator
import astropy.units as u

# Local imports
from miri_functions_tidy import estimate_planet_filter_snr, readin_miri_filters

# Configure environment and suppress warnings
os.environ["pandeia_refdata"] = "pandeia_data-3.0rc3"
os.environ["PYSYN_CDBS"] = "grp/redcat/trds"
warnings.filterwarnings('ignore')

class PhysicalConstants:
    """Physical constants in SI units"""
    AU = 1.495979e11
    SIGMA = 5.670374419e-8
    H_PLANCK = 6.62607015e-34
    K_BOLTZMANN = 1.380649e-23
    C_VACUUM = 299792458.0
    R_GAS = 8.3144598
    G = 6.67408e-11
    
    # Astronomical constants
    R_SUN = 695508e3
    M_SUN = 1.989e30
    R_JUPITER = 71492e3
    M_JUPITER = 1.8986e27
    R_EARTH = 6.3781e6
    M_EARTH = 5.972e24
    
    # Conversion factors
    DAYS_TO_SECONDS = 86400

class EmissionResult(NamedTuple):
    """Container for emission calculation results"""
    filter_flux: float
    filter_flux_error: float
    filter_15um: float
    filter_15um_error: float
    filter_other: float
    filter_other_error: float
    wavelengths: np.ndarray
    fp_fs: np.ndarray

class SpectralCalculator:
    """Handles spectral calculations for planetary emission"""
    
    def __init__(self, wave_min: float = 300.0, wave_max: float = 1e5, wave_bins: int = 300):
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.wave_bins = wave_bins
        self.setup_wavelength_grid()
        
    def setup_wavelength_grid(self):
        """Initialize wavelength grid for calculations"""
        obs_pts = np.logspace(np.log10(self.wave_min), np.log10(self.wave_max), self.wave_bins)
        self.obs_bc = (obs_pts[1:] + obs_pts[:-1]) * 0.5  # bin centers
        self.obs_bw = obs_pts[1:] - obs_pts[:-1]          # bin widths
        
    @staticmethod
    def evaluate_planck(wav:float, tmp:float):
        ''' 
        Evaluate the planck function at a given wavelength and temperature
        '''
        # Constants  (SI units)
        AU    = 1.495979e+11  
        sigma = 5.670374419e-8
        h_pl  = 6.62607015e-34
        k_B   = 1.380649e-23  
        c_vac = 299792458.0   

        # Output value 
        flx = 0.0
    
        # Convert nm to m
        wav = wav * 1.0e-9
    
        # Calculate planck function value [W m-2 sr-1 m-1]
        # http://spiff.rit.edu/classes/phys317/lectures/planck.html
        flx = 2.0 * h_pl * c_vac * (c_vac / wav**5.0) / ( np.exp(h_pl * c_vac / (wav * k_B * tmp)) - 1.0)
    
        # Integrate solid angle (hemisphere), convert units
        flx = flx * np.pi * 1.0e-9 # [W m-2 nm-1]
    
        return flx

class PlanetarySystem:
    """Represents a planetary system with its star and planet parameters"""
    
    def __init__(self, planet_name: str, df: pd.DataFrame):
        self.planet_name = planet_name
        self._load_system_parameters(df)
        self.calculate_derived_parameters()
        
    def _load_system_parameters(self, df: pd.DataFrame):
        """Load basic system parameters from dataframe"""
        planet_data = df[df['pl_name'] == self.planet_name].iloc[0]
        
        self.period = planet_data['pl_orbper']
        self.semimajor_axis = planet_data['pl_orbsmax']
        self.stellar_mass = planet_data['st_mass']
        self.stellar_radius = planet_data['st_rad']
        self.stellar_temp = planet_data['st_teff']
        self.stellar_logg = planet_data['st_logg']
        self.stellar_metallicity = planet_data['st_met']
        self.planet_radius = planet_data['pl_radj']
        
    def calculate_derived_parameters(self):
        """Calculate derived system parameters"""
        if not pd.isna(self.semimajor_axis):
            self.a_rs = (self.semimajor_axis * PhysicalConstants.AU / PhysicalConstants.R_SUN) / self.stellar_radius
        else:
            st_mass_kg = self.stellar_mass * PhysicalConstants.M_SUN
            a_m = (((self.period * 86400)**2 * st_mass_kg * PhysicalConstants.G / (4 * np.pi**2)))**(1/3)
            self.a_rs = (a_m / PhysicalConstants.R_SUN) / self.stellar_radius
            
        self.rp_rs = (self.planet_radius * PhysicalConstants.R_JUPITER / PhysicalConstants.R_SUN) / self.stellar_radius
        self.instellation = (PhysicalConstants.SIGMA * (self.stellar_temp**4)) / (self.a_rs**2)
        self.Teq_instellation = (self.instellation / 5.6e-8)**0.25
        self.Teq_redist_instellation = (0.5 * self.instellation / 5.6e-8)**0.25

class EmissionCalculator:
    """Handles emission calculations for a planetary system"""
    
    def __init__(self, system: PlanetarySystem, spectral_calc: SpectralCalculator):
        self.system = system
        self.spectral_calc = spectral_calc
        self.setup_grids()
        
    def setup_grids(self, grid_points: int = 7):
        """Setup calculation grids"""
        self.day_lons = np.linspace(-np.pi/2, np.pi/2, grid_points)
        self.day_lats = np.linspace(-np.pi/2, np.pi/2, grid_points)
        
    def load_stellar_spectrum(self, Ts: float, logg: float, logZ: float) -> None:
        """Load and process stellar spectrum"""
        Ts_grid = np.arange(2000, 4000 + 1e-10, 200)
        logg_grid = np.arange(4.0, 5.25 + 1e-10, 0.25)
        logZ_grid = np.arange(-1, 1 + 1e-10, 0.25)
        CtoO = 0.5

        # Find nearest grid points
        sphinx_Ts = str(np.round(Ts_grid[np.argmin(abs(Ts_grid - Ts))], 1))
        sphinx_logg = str(np.round(logg_grid[np.argmin(abs(logg_grid - logg))], 2))
        sphinx_logZ = str(np.round(logZ_grid[np.argmin(abs(logZ_grid - logZ))], 2))
        sphinx_CtoO = str(np.round(CtoO, 2))

        # Construct file path
        if float(sphinx_logZ) >= 0:
            star_path = f"data/SPHINX_V4/Teff_{sphinx_Ts}_logg_{sphinx_logg}_logZ_+{sphinx_logZ}_CtoO_{sphinx_CtoO}.txt"
        else:
            star_path = f"data/SPHINX_V4/Teff_{sphinx_Ts}_logg_{sphinx_logg}_logZ_{sphinx_logZ}_CtoO_{sphinx_CtoO}.txt"

        # Load and process stellar spectrum
        star_data = np.loadtxt(star_path).T
        star_wl = 1e3 * star_data[0]  # Convert to nm
        star_fl = star_data[1]
        
        # Calculate total stellar flux and scale to match instellation
        total_stellar_flux = np.trapz(star_fl, star_wl)
        scale_factor = PhysicalConstants.SIGMA * (Ts**4) / total_stellar_flux
        star_fl = star_fl * scale_factor
        
        self.star_interpolator = PchipInterpolator(star_wl, star_fl)
        
    def load_albedo_data(self, albedo_file: str) -> Tuple[float, float]:
        """Load and process albedo data"""
        w_data = np.loadtxt(f"data/our_suite_albedo/{albedo_file}").T
        w_wl = list(w_data[0])
        w_w = list(w_data[1])
        
        # Extend wavelength range
        min_alb_wl = w_wl[0]
        max_alb_wl = w_wl[-1]
        
        w_wl = [0.0] + w_wl + [1e9]
        w_w = [w_w[0]] + w_w + [w_w[-1]]
        
        self.gamma = np.sqrt(1 - np.array(w_w))
        self.gamma_wl = np.array(w_wl)
        
        return min_alb_wl, max_alb_wl

    def calculate_reflection_functions(self, sza: float) -> Tuple[PchipInterpolator, PchipInterpolator]:
        """Calculate reflection functions for given solar zenith angle"""
        mu0 = np.cos(sza)
        
        # Directional-hemispheric reflectance
        r_h = (1 - self.gamma) / (1 + 2 * self.gamma * mu0)
        alb_interpolator = PchipInterpolator(self.gamma_wl, r_h)
        
        # Spherical reflectance
        r_0 = (1 - self.gamma) / (1 + self.gamma)
        r_s = r_0 * (1 - (1./3.) * (self.gamma / (1 + self.gamma)))
        r_s_interpolator = PchipInterpolator(self.gamma_wl, r_s)
        
        return alb_interpolator, r_s_interpolator

    def calculate_fluxes(self, wav: float, wid: float, tmp: float, 
                        instellation_scale: float, alb_itp: PchipInterpolator, 
                        r_s_itp: PchipInterpolator) -> Dict[str, float]:
        """Calculate component fluxes at surface"""
        # Get albedo and emissivity at this wavelength
        alb_s = alb_itp(wav)
        eps_s = 1 - r_s_itp(wav)
        
        # Calculate thermal emission (multiply by emissivity)
        thermal = self.spectral_calc.evaluate_planck(wav, tmp) * wid * eps_s
        
        # Calculate scaled stellar flux - use instellation_scale (cos(sza)) directly
        try:
            stellar = (instellation_scale * self.star_interpolator(wav) * wid) / (self.system.a_rs**2)
        except ValueError:
            stellar = 0.0
            
            
        return {
            "LW_UP": thermal,
            "LW_DN": 0.0,
            "SW_DN": stellar * (1 - alb_s),
            "SW_UP": stellar * alb_s
        }

    def solve_radiative_equilibrium(self, instellation_scale: float, 
                                  alb_itp: PchipInterpolator, 
                                  r_s_itp: PchipInterpolator) -> float:
        """Solve for temperature satisfying radiative equilibrium"""
        def residual(tmp: float) -> float:
            up = down = 0.0
            for bc, bw in zip(self.spectral_calc.obs_bc, self.spectral_calc.obs_bw):
                fluxes = self.calculate_fluxes(bc, bw, tmp, instellation_scale, alb_itp, r_s_itp)
                up += fluxes["LW_UP"]
                down += fluxes["LW_DN"] + fluxes["SW_DN"]
            return down - up

        # Start with equilibrium temperature as initial guess
        x0 = self.system.Teq_instellation  # Changed from 1000.0 to Teq
        eps = 1e-4
        p1 = x0 * (1 + eps)
        p1 += eps if p1 >= 0 else -eps
        
        # Add bounds to ensure physical temperatures
        sol = root_scalar(residual, x0=x0, x1=p1) 
        
        temp = float(sol.root)
            
        return temp

    def calculate_emission_spectrum(self, tmp: float, instellation_scale: float,
                                  alb_itp: PchipInterpolator, 
                                  r_s_itp: PchipInterpolator) -> Dict[str, np.ndarray]:
        """Calculate emission spectrum components"""
        components = {
            "LW_UP": [],
            "SW_UP": [],
            "SW_DN": [],
            "UP": []
        }
        
        # Print diagnostic information
        # print(f"Calculating spectrum for T = {tmp:.1f}K")
        
        for bc, bw in zip(self.spectral_calc.obs_bc, self.spectral_calc.obs_bw):
            fluxes = self.calculate_fluxes(bc, bw, tmp, instellation_scale, alb_itp, r_s_itp)
            
            components["LW_UP"].append(fluxes["LW_UP"] / bw)
            components["SW_UP"].append(fluxes["SW_UP"] / bw)
            components["SW_DN"].append(fluxes["SW_DN"] / bw)
            components["UP"].append((fluxes["LW_UP"] + fluxes["SW_UP"]) / bw)
        
        return {k: np.array(v, dtype=float) for k, v in components.items()}
    
    def setup_grids(self, grid_points: int = 5):
        """Setup calculation grids
        
        Note: Grid now spans -pi/2 to pi/2 in both longitude and latitude
        to properly capture the dayside hemisphere
        """
        self.day_lons = np.linspace(-np.pi/2, np.pi/2, grid_points)  # Changed from -np.pi to np.pi
        self.day_lats = np.linspace(-np.pi/2, np.pi/2, grid_points)
        

    def calculate_emission(self, albedo_file: str) -> EmissionResult:
        """Calculate full planetary emission"""
        # Load stellar spectrum and albedo data
        self.load_stellar_spectrum(self.system.stellar_temp, 
                                 self.system.stellar_logg,
                                 self.system.stellar_metallicity)
        min_alb_wl, max_alb_wl = self.load_albedo_data(albedo_file)
        
        # Initialize day-side flux array
        day_Fp = np.zeros([len(self.day_lons), len(self.day_lats), len(self.spectral_calc.obs_bc)])
        
        # Calculate emission across planetary surface
        for i, lon in enumerate(self.day_lons):
            for j, lat in enumerate(self.day_lats):
                # Calculate solar zenith angle
                # Only consider illuminated portions where cos(sza) > 0
                cos_sza = np.cos(lon) * np.cos(lat)
                
                if cos_sza > 0:  # Only calculate for dayside
                    # Get reflection functions
                    alb_itp, r_s_itp = self.calculate_reflection_functions(np.arccos(cos_sza))
                    
                    # Solve for temperature
                    tmp = self.solve_radiative_equilibrium(cos_sza, alb_itp, r_s_itp)
                    

                    # Calculate emission spectrum
                    emission = self.calculate_emission_spectrum(tmp, cos_sza, alb_itp, r_s_itp)
                    day_Fp[i, j, :] = emission['UP']
                else:
                    day_Fp[i, j, :] = 0.0  # No emission from nightside in this simple model

        
        # Calculate total emission
        total_day_area = (np.cos(self.day_lats[:, None])**2) * (np.cos(self.day_lons[None, :])) * \
                            np.gradient(self.day_lats[:, None], axis=0)
        total_day_area = np.sum(np.sum(total_day_area, axis=0), axis=0)
        
        # Calculate total planetary flux
        total_day_Fp = day_Fp * (np.cos(self.day_lats[:, None, None])**2) * \
                      (np.cos(self.day_lons[None, :, None])) * \
                      np.gradient(self.day_lons[None, :, None], axis=1) * \
                      np.gradient(self.day_lats[:, None, None], axis=0)
        
        total_day_Fp = np.sum(np.sum(total_day_Fp, axis=0), axis=0) / total_day_area
        
        # Calculate stellar flux
        Fs = np.array([self.star_interpolator(bc) for bc in self.spectral_calc.obs_bc])
        # Fs *= (self.system.a_rs**2)
        
        # Calculate planet/star contrast
        fp_fs = 1e6 * (self.system.rp_rs**2) * total_day_Fp / Fs
        
        # Calculate filter values using miri_functions
        eclipse_snr_list, names_list, filter_15um_list, filter_15um_error_list, filter_list, filter_error_list = \
            estimate_planet_filter_snr(Fs, total_day_Fp, self.spectral_calc.obs_bc * 1e-3, self.system.Teq_instellation, self.system.planet_name)
        
        return EmissionResult(
            filter_flux=filter_list[:],
            filter_flux_error=filter_error_list[:],
            filter_15um=filter_list[4],
            filter_15um_error=filter_error_list[4],
            filter_other=filter_list[5],
            filter_other_error=filter_error_list[5],
            wavelengths=self.spectral_calc.obs_bc,
            fp_fs=fp_fs
        )

def plot_spectral_results(results: Dict, save_path: Optional[str] = None) -> None:
    """Plot spectral results with MIRI filter data."""
    wl_filt, dwl_filt, tputs, fnames = readin_miri_filters()
    tputs_modified = tputs[:, :-2]
    wl_filt_modified = wl_filt

    # Calculate filter properties
    widths = get_filter_widths(wl_filt_modified, tputs_modified)[:-2]
    central_wl = [(wl_filt_modified[tputs_modified[:,i] >= 0.05 * np.max(tputs_modified[:,i])]).mean() 
                  for i in range(tputs_modified.shape[1])]

    plt.figure(figsize=(10, 6))
    plt.xlim(5, 21)
    plt.ylim(0, 120)

    for i, (case, result) in enumerate(results.items()):
        color = f'C{i}'
        # Plot continuous spectrum
        plt.plot(result.wavelengths*1e-3, result.fp_fs, 
                color=color, label=case)

        # Plot filter points with error bars
        plt.errorbar(central_wl, result.filter_flux, 
                    xerr=widths/2, yerr=result.filter_flux_error,
                    fmt='o', capsize=2, color=color)

    plt.legend()
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Fp/Fs (ppm)')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    plt.close()

def get_filter_widths(wl_filt: np.ndarray, tputs: np.ndarray) -> np.ndarray:
    
    """Calculate filter widths based on throughput threshold."""
    wl_filt, dwl_filt, tputs, fnames = readin_miri_filters()
    filter_widths = []
    
    for i in range(tputs.shape[1]):
        tput = tputs[:, i]
        threshold = 0.05 * np.max(tput)
        mask = tput >= threshold
        wl_above_thresh = wl_filt[mask]
        width = wl_above_thresh[-1] - wl_above_thresh[0]
        filter_widths.append(width)
    
    return np.array(filter_widths)


"""Main execution function"""
# Load planet data
planet_data = pd.read_csv("exoplanet_table_comp_200824_csv.csv")

# Initialize calculators
spectral_calc = SpectralCalculator()
plotter = Plotter()

# Process planets
for planet_name in ['GJ 357 b']:
    print(f"\nProcessing {planet_name}...")

    # Initialize system
    system = PlanetarySystem(planet_name, planet_data)
    emission_calc = EmissionCalculator(system, spectral_calc)

    # Store results for all albedo models
    results = {}

    # First process blackbody case
    print("Calculating blackbody case...")
    bb_result = emission_calc.calculate_emission('bb_w.txt')
    results['bb_w.txt'] = bb_result

    # Process other albedo models
    for albedo_file in os.listdir("data/our_suite_albedo"):
        if albedo_file.endswith('w.txt') and albedo_file.startswith('9'):
            print(f"Processing {albedo_file}...")
            results[albedo_file] = emission_calc.calculate_emission(albedo_file)

    # Plot comparison of all albedo models
    # plotter.plot_albedo_comparison(system, results)
    
    plot_spectral_results(results)

    # Print summary
    print(f"\nResults for {planet_name}:")
    print(f"Equilibrium Temperature: {system.Teq_instellation:.1f} K")
    print(f"Blackbody 15μm filter value: {bb_result.filter_15um:.2f} ± {bb_result.filter_15um_error:.2f} ppm")
    print(f"Blackbody 12μm filter value: {bb_result.filter_other:.2f} ± {bb_result.filter_other_error:.2f} ppm")

    
