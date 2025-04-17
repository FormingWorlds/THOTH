"""Visualization functions for spectral data and results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .constants import FIGURE_DIR
from .data_loader import SurfaceData
from .spectral import SpectralCalculator

def plot_raw_spectra(surface: SurfaceData,
                    ax: Optional[plt.Axes] = None,
                    show: bool = True,
                    save_path: Optional[str] = None) -> plt.Axes:
    """Plot raw spectral data."""
    if ax is None:
        # Check if we have raw SW/LW data
        if 'sw_data' in surface.metadata and 'lw_data' in surface.metadata:
            # Create two subplots: one for raw data, one for combined
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            ax = ax2  # Use second subplot for combined data
            
            # Plot raw SW/LW data on first subplot
            sw_data = surface.metadata['sw_data']
            lw_data = surface.metadata['lw_data']
            
            logging.info(f"Plotting raw SW data: {sw_data.shape[0]} points")
            sw_wl = sw_data[:,0]  # Keep in original units
            sw_ref = sw_data[:,1]
            ax1.plot(sw_wl, sw_ref, 'b.', markersize=2, label='Shortwave')
            
            logging.info(f"Plotting raw LW data: {lw_data.shape[0]} points")
            lw_wl = lw_data[:,0]  # Keep in original units
            lw_ref = lw_data[:,1]
            ax1.plot(lw_wl, lw_ref, 'r.', markersize=2, label='Longwave')
            
            ax1.set_xlabel('Wavelength (original units)')
            ax1.set_ylabel('Reflectance')
            ax1.set_title(f'Raw SW/LW Data: {surface.surface_id}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            # Single plot for frankenspectrum
            fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spectral data
    # For frankenspectrum, wavelengths are in microns, so multiply by 1e3 to get nm
    wavelength_microns = surface.wavelength  # Already in microns for frankenspectrum
    
    # Plot the data
    ax.plot(wavelength_microns, surface.reflectance, 'k-', label='Spectrum', linewidth=1)
    ax.plot(wavelength_microns, surface.reflectance, 'k.', markersize=1, alpha=0.3)
    
    # Customize plot
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Reflectance')
    ax.set_title(f'Spectral Data: {surface.surface_id}')
    ax.set_xlim(0, 20)  # Set wavelength range to 0-20 microns
    ax.set_ylim(0, 1)   # Set reflectance range to 0-1
    ax.grid(True, alpha=0.3)
    
    # Add text with data statistics
    stats_text = (f'Points: {len(surface.wavelength)}\n'
                 f'λ range: {wavelength_microns.min():.2f} - {wavelength_microns.max():.2f} μm\n'
                 f'Mean reflectance: {np.mean(surface.reflectance):.3f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=8, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logging.info(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return ax

def plot_emission_spectrum(emission_result: Dict[str, np.ndarray],
                         contrast: np.ndarray,
                         ax: Optional[plt.Axes] = None,
                         show: bool = True,
                         save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot emission spectrum and planet/star contrast.
    
    Args:
        emission_result: Result dictionary from EmissionCalculator
        contrast: Planet/star contrast array
        ax: Optional matplotlib axes for plotting
        show: Whether to display the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2])
        
        # Hide x-axis labels on top plot (ax1) only
        plt.setp(ax1.get_xticklabels(), visible=False)

    else:
        ax1, ax2, ax3 = ax
    
    # Get wavelength in microns
    wavelength = emission_result['wavelength'] * 1e-3
    
    # Plot emission components
    ax1.plot(wavelength, emission_result['thermal_emission'], 
             label='Thermal Emission', color='red')
    
    if 'reflected_light' in emission_result:
        ax1.plot(wavelength, emission_result['reflected_light'],
                label='Reflected Light', color='blue')
        ax1.plot(wavelength, emission_result['total_emission'],
                label='Total', color='black', ls='--')
    
    # Add reference blackbody at 400K with proper scaling
    bb_400K = np.array([SpectralCalculator.planck(wl, 400.0) 
                        for wl in emission_result['wavelength']])
    # Use average emissivity of 0.9 and same planet area as main calculation
    planet_area = np.max(emission_result['thermal_emission']) / np.max(bb_400K)
    ax1.plot(wavelength, bb_400K * planet_area * 0.9,
             label='400K Blackbody (ε=0.9)', color='orange', ls=':')
    
    # Add simple reflection with albedo 0.5
    if 'reflected_light' in emission_result:
        simple_reflection = 0.5 * np.array([
            SpectralCalculator.planck(wl, emission_result.get('star_temp', 3000.0))
            for wl in emission_result['wavelength']
        ])
        # Use same geometric factors as main calculation
        scale_factor = np.max(emission_result['reflected_light']) / np.max(simple_reflection)
        ax1.plot(wavelength, simple_reflection * scale_factor,
                label='Simple Reflection (A=0.5)', color='cyan', ls=':')
    
    ax1.set_ylabel('Spectral Flux (W/m²/nm)')
    ax1.set_xlim(1, 20)  # Set wavelength range to 1-20 microns
    # ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot contrast
    ax2.plot(wavelength, contrast, 'k-')
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Fp/F* (ppm)')
    ax2.set_xlim(1, 20)  # Set wavelength range to 1-20 microns
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot contrast zoomed-in (to compare with Manfield 2024)
    ax3.plot(wavelength, contrast, 'k-')
    ax3.set_xlabel('Wavelength (μm)')
    ax3.set_ylabel('Fp/F* (ppm)')
    ax3.set_xlim(5, 12)  # Set wavelength range to 1-20 microns
    ax3.set_ylim(0, 400)  # Set wavelength range to 1-20 microns
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return (ax1, ax2, ax3)

def plot_temperature_map(longitudes: np.ndarray,
                        latitudes: np.ndarray,
                        temperatures: np.ndarray,
                        ax: Optional[plt.Axes] = None,
                        show: bool = True,
                        save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot planetary surface temperature map.
    
    Args:
        longitudes: Array of longitude values (radians)
        latitudes: Array of latitude values (radians)
        temperatures: 2D array of temperatures
        ax: Optional matplotlib axes for plotting
        show: Whether to display the plot
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to degrees for plotting
    lon_deg = np.rad2deg(longitudes)
    lat_deg = np.rad2deg(latitudes)
    
    # Create mesh for plotting
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)
    
    # Create temperature map
    im = ax.pcolormesh(lon_grid, lat_grid, temperatures.T,
                      shading='auto', cmap='RdYlBu_r')
    plt.colorbar(im, ax=ax, label='Temperature (K)')
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Surface Temperature Map')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()
        
    return ax 