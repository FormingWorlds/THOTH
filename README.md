# Rocky Surfaces Analysis

This codebase provides tools for analyzing spectral data of rocky surfaces, calculating their emission properties, and visualizing the results. It is designed to model and analyze the thermal emission and reflection properties of rocky exoplanet surfaces.

## Project Structure

```
rocky_surfaces/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── data/              # Data directory
│   └── raw/           # Raw surface spectral data
├── src/               # Source code
│   ├── constants.py   # Physical constants and configuration
│   ├── data_loader.py # Data loading functions
│   ├── spectral.py    # Spectral processing
│   ├── emission.py    # Emission calculations
│   ├── plotting.py    # Visualization functions
│   └── process_surface.py # Main processing script
└── output/           # Generated outputs
    ├── processed/    # Processed data
    └── figures/      # Generated plots
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place raw surface spectral data files in `data/raw/`
2. Ensure the planet database file (`exoplanet_table_comp_200824_csv.csv`) is in the `data/` directory
3. Run the main processing script:
```bash
python src/process_surface.py
```

The script will:
- Load and process surface spectral data
- Calculate thermal emission and reflection properties
- Generate temperature maps and emission spectra
- Save results to the output directories

## Data Format

Raw surface data should be provided in tab-delimited files with the following naming convention:
- Shortwave data: `{ID}_sw_{details}.tab`
- Longwave data: `{ID}_lw_{details}.tab`
- Combined data: `{ID}_combine_{details}.tab`

Example: 
- `B19_lherzolite_sw_cpfb08.tab`
- `B19_lherzolite_lw_nafb8p.tab`
- `B19_lherzolite_combine_cpfb08p_nafb8p.tab`

## Code Components

- `constants.py`: Physical constants and configuration values
- `data_loader.py`: Functions for loading and processing raw spectral data
- `spectral.py`: Spectral calculations and processing
- `emission.py`: Planetary emission calculations and temperature modeling
- `plotting.py`: Visualization functions for results
- `process_surface.py`: Main script coordinating the analysis pipeline 