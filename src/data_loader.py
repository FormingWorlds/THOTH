"""Functions for loading and processing raw surface spectral data."""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

from .constants import RAW_DATA_DIR, SW_FOOTER, LW_FOOTER

class SurfaceData:
    """Container for surface spectral data."""
    def __init__(self, 
                 wavelength: np.ndarray,
                 reflectance: np.ndarray,
                 surface_id: str,
                 metadata: Optional[Dict] = None):
        self.wavelength = wavelength
        self.reflectance = reflectance
        self.surface_id = surface_id
        self.metadata = metadata or {}

def clean_spectral_data(wavelength: np.ndarray, 
                       reflectance: np.ndarray,
                       surface_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean spectral data by removing NaN values and invalid entries.
    
    Args:
        wavelength: Array of wavelength values
        reflectance: Array of reflectance values
        surface_id: Surface identifier for logging
        
    Returns:
        Cleaned wavelength and reflectance arrays
    """
    # Find valid data points
    valid_mask = np.logical_and(
        np.isfinite(wavelength),
        np.isfinite(reflectance)
    )
    
    # Log information about removed points
    n_total = len(wavelength)
    n_valid = np.sum(valid_mask)
    if n_valid < n_total:
        logging.warning(f"Surface {surface_id}: Removed {n_total - n_valid} invalid points")
        
    # Keep only valid points
    wavelength_clean = wavelength[valid_mask]
    reflectance_clean = reflectance[valid_mask]
    
    # Ensure positive wavelengths
    positive_mask = wavelength_clean > 0
    if not np.all(positive_mask):
        logging.warning(f"Surface {surface_id}: Removed {np.sum(~positive_mask)} non-positive wavelengths")
        wavelength_clean = wavelength_clean[positive_mask]
        reflectance_clean = reflectance_clean[positive_mask]
    
    # Clip reflectance to [0, 1]
    invalid_reflectance = np.logical_or(reflectance_clean < 0, reflectance_clean > 1)
    if np.any(invalid_reflectance):
        logging.warning(f"Surface {surface_id}: Clipping {np.sum(invalid_reflectance)} reflectance values to [0, 1]")
        reflectance_clean = np.clip(reflectance_clean, 0, 1)
    
    return wavelength_clean, reflectance_clean

def get_surface_info(surface_id: str) -> Dict[str, str]:
    """
    Get file names and metadata for a given surface ID by searching the data directory.
    
    Args:
        surface_id: First 3 characters of the surface file name (e.g., '96_')
        
    Returns:
        Dictionary containing file info and whether it's a frankenspectrum
    """
    # Special case for blackbody
    if surface_id == 'bb_':
        return {
            'data_name': '15b_albitedust_fs',  # Use this as template
            'is_frankenspectrum': True,
            'is_blackbody': True
        }
    
    # Get list of all files in RAW_DATA_DIR
    data_dir = Path(RAW_DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")
    
    # Search for files matching the surface ID prefix
    all_files = [f.stem for f in data_dir.glob("*.tab")]
    matching_files = [f for f in all_files if f.startswith(surface_id)]
    
    if not matching_files:
        raise ValueError(f"No files found starting with {surface_id}")
    
    # Find SW and LW files
    sw_files = [f for f in matching_files if '_sw_' in f]
    lw_files = [f for f in matching_files if '_lw_' in f]
    
    # If we have both SW and LW files, use those
    if sw_files and lw_files:
        surface_info = {
            'sw_name': sw_files[0],
            'lw_name': lw_files[0],
            'is_frankenspectrum': False,
            'is_blackbody': False
        }
        logging.info(f"Found separate SW/LW files for {surface_id}:")
        logging.info(f"- SW: {surface_info['sw_name']}")
        logging.info(f"- LW: {surface_info['lw_name']}")
    # Otherwise treat as frankenspectrum and use the first matching file
    else:
        surface_info = {
            'data_name': matching_files[0],
            'is_frankenspectrum': True,
            'is_blackbody': False
        }
        logging.info(f"Using single file as frankenspectrum: {surface_info['data_name']}")
    
    return surface_info

def load_raw_surface(surface_id: str, 
                    frankenspectrum: bool = False) -> SurfaceData:
    """
    Load raw surface spectral data from files.
    
    Args:
        surface_id: First 3 characters of the surface file name (e.g., '96_')
        frankenspectrum: Whether to use frankenspectrum (ignored if only one file exists)
        
    Returns:
        SurfaceData object containing wavelength and reflectance data
    """
    # Define file names based on surface ID
    surface_info = get_surface_info(surface_id)
    metadata = {}
    
    try:
        # If it's a frankenspectrum or we only have one file
        if surface_info['is_frankenspectrum'] or frankenspectrum:
            # Load single file with comma delimiter
            data_path = Path(RAW_DATA_DIR) / f"{surface_info['data_name']}.tab"
            
            # Try loading with different options
            try:
                surface = np.genfromtxt(data_path, delimiter=',')
            except:
                # If comma delimiter fails, try space delimiter
                surface = np.genfromtxt(data_path)
            
            logging.info(f"Loaded data shape: {surface.shape}")
            logging.info(f"Data type: {surface.dtype}")
            
            # Reshape if needed
            if surface.ndim == 1:
                # If we have a flat array, try to reshape it
                if len(surface) % 2 == 0:  # If even number of elements
                    surface = surface.reshape(-1, 2)
                    logging.info(f"Reshaped to: {surface.shape}")
                else:
                    raise ValueError(f"Unexpected data format in {data_path}")
            
            wavelength = surface[:,0]  # Already in microns
            
            # For blackbody, set reflectance to zero
            if surface_info.get('is_blackbody', False):
                reflectance = np.zeros_like(wavelength)
                logging.info("Creating blackbody surface (zero reflectance)")
            else:
                reflectance = surface[:,1]
            
            logging.info(f"Wavelength range: {wavelength.min():.2f} to {wavelength.max():.2f}")
            logging.info(f"Reflectance range: {reflectance.min():.2f} to {reflectance.max():.2f}")
            
        else:
            # Load separate SW and LW spectra
            sw_path = Path(RAW_DATA_DIR) / f"{surface_info['sw_name']}.tab"
            lw_path = Path(RAW_DATA_DIR) / f"{surface_info['lw_name']}.tab"
            
            # Load longwave data
            surface_lw = np.genfromtxt(lw_path, 
                                     skip_header=1,
                                     skip_footer=LW_FOOTER)
            logging.info(f"Loaded LW data: {surface_lw.shape[0]} points")
            
            # Load shortwave data
            surface_sw = np.genfromtxt(sw_path,
                                     skip_header=1,
                                     skip_footer=SW_FOOTER)
            logging.info(f"Loaded SW data: {surface_sw.shape[0]} points")
            
            # Store raw data in metadata
            metadata['sw_data'] = surface_sw
            metadata['lw_data'] = surface_lw
            
            # Find joining point between SW and LW
            join_wl = surface_sw[:,0][-1]  # Use last point of SW as join point
            join_lw_wl_idx = np.nanargmin(abs(surface_lw[:,0] - join_wl)) + 1
            join_sw_wl_idx = np.nanargmin(abs(surface_sw[:,0] - join_wl))
            
            # Scale LW data to match at joining point
            surface_lw_r_scaled = (surface_sw[:,1][join_sw_wl_idx] * 
                                 surface_lw[:,1] / 
                                 surface_lw[:,1][join_lw_wl_idx])
            
            # Combine the data using only relevant portions
            wavelength = np.concatenate([surface_sw[:,0], 
                                      surface_lw[:,0][join_lw_wl_idx:]])
            reflectance = np.concatenate([surface_sw[:,1], 
                                        surface_lw_r_scaled[join_lw_wl_idx:]])
            
            # Sort by wavelength
            sort_idx = np.argsort(wavelength)
            wavelength = wavelength[sort_idx]
            reflectance = reflectance[sort_idx]
        
        # Clean the data
        wavelength_clean, reflectance_clean = clean_spectral_data(
            wavelength, reflectance, surface_id)
        
        if len(wavelength_clean) == 0:
            raise ValueError("No valid data points after cleaning")
            
        return SurfaceData(wavelength_clean, reflectance_clean, surface_id, metadata)
        
    except Exception as e:
        logging.error(f"Error loading surface {surface_id}: {str(e)}")
        raise 