"""Example script demonstrating the usage of the rocky surfaces analysis codebase."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
import traceback
import pandas as pd
from typing import Optional, Dict

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_raw_surface, get_surface_info
from src.spectral import SpectralCalculator
from src.emission import PlanetarySystem, EmissionCalculator, calculate_planet_star_contrast
from src.plotting import plot_raw_spectra, plot_emission_spectrum, plot_temperature_map
from src.constants import FIGURE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_surface(surface_id: str = '97_', 
                   frankenspectrum: bool = False,
                   planet_name: str = 'TRAPPIST-1 b') -> Optional[dict]:
    """Process a surface and calculate its emission properties.
    
    Args:
        surface_id: ID of the surface to process
        frankenspectrum: Whether to use frankenspectrum
        planet_name: Name of the planet in the database
    """
    try:
        # Load planet database from data folder
        csv_path = Path("data") / "exoplanet_table_comp_200824_csv.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Planet database not found at {csv_path}")
            
        planet_data = pd.read_csv(csv_path)
        
        # Get planet parameters from database
        planet_matches = planet_data[planet_data['pl_name'] == planet_name]
        if len(planet_matches) == 0:
            raise ValueError(f"Planet {planet_name} not found in database")
            
        planet_info = planet_matches.iloc[0]
        
        # 1. Load surface data
        logging.info(f"Loading surface data for {surface_id}...")
        surface = load_raw_surface(surface_id, frankenspectrum)
        
        # Get surface info for full filename
        surface_info = get_surface_info(surface_id)
        if surface_info['is_frankenspectrum'] or frankenspectrum:
            full_name = surface_info['data_name']
        else:
            full_name = f"{surface_info['sw_name']}_{surface_info['lw_name']}"
        
        # 2. Setup planetary system with database parameters
        logging.info(f"Setting up planetary system for {planet_name}...")
        system = PlanetarySystem(
            star_temp=planet_info['st_teff'],
            star_radius=planet_info['st_rad'],  # Already in solar radii
            planet_radius=planet_info['pl_radj'] * 11.2,  # Convert Jupiter radii to Earth radii
            semi_major_axis=planet_info['pl_orbsmax']  # Already in AU
        )
        
        logging.info(f"System parameters:")
        logging.info(f"- Star temperature: {system.star_temp:.1f} K")
        logging.info(f"- Star radius: {system.star_radius:.3f} RSUN")
        logging.info(f"- Planet radius: {system.planet_radius:.3f} RE")
        logging.info(f"- Semi-major axis: {system.semi_major_axis:.3f} AU")
        logging.info(f"- Equilibrium temperature: {system.equilibrium_temp:.1f} K")
        logging.info(f"- Instellation: {system.instellation:.2e} W/m²")
        
        # 3. Calculate emission
        logging.info("Calculating emission spectrum...")
        spectral_calc = SpectralCalculator()
        emission_calc = EmissionCalculator(system, spectral_calc)
        
        try:
            logging.info("Starting emission calculation...")
            emission_result = emission_calc.calculate_emission(
                surface, 
                include_reflection=True
            )
            logging.info("Emission calculation completed successfully")
        except Exception as e:
            logging.error("Error in emission calculation:")
            logging.error(traceback.format_exc())
            raise
        
        # Calculate contrast
        contrast = calculate_planet_star_contrast(emission_result, system)
        
        # 4. Generate plots
        logging.info("Generating plots...")
        
        # Create figures directory if it doesn't exist
        Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Raw spectrum
        raw_spectrum_path = Path(FIGURE_DIR) / f'{full_name}_raw_spectrum.png'
        plot_raw_spectra(surface, save_path=str(raw_spectrum_path))
        logging.info(f"Raw spectrum plot saved to {raw_spectrum_path}")
        
        # Emission spectrum and contrast
        emission_path = Path(FIGURE_DIR) / f'{full_name}_emission_spectrum.png'
        plot_emission_spectrum(emission_result, contrast, save_path=str(emission_path))
        logging.info(f"Emission spectrum plot saved to {emission_path}")
        
        # Temperature map
        temp_map_path = Path(FIGURE_DIR) / f'{full_name}_temperature_map.png'
        plot_temperature_map(
            emission_calc.longitudes,
            emission_calc.latitudes,
            emission_calc.temperatures,
            save_path=str(temp_map_path)
        )
        logging.info(f"Temperature map plot saved to {temp_map_path}")
        
        return {
            'emission_result': emission_result,
            'contrast': contrast,
            'temperatures': emission_calc.temperatures
        }
        
    except Exception as e:
        logging.error(f"Error processing surface {surface_id}:")
        logging.error(traceback.format_exc())
        return None

def main():
    """Main execution function."""
    try:
        results = process_surface()
        if results is None:
            sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error:")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 