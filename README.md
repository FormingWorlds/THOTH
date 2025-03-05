# Rocky Surfaces Analysis

This codebase provides tools for analyzing spectral data of rocky surfaces, calculating their emission properties, and visualizing the results.

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
│   └── plotting.py    # Visualization functions
├── notebooks/         # Jupyter notebooks
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
2. Use the provided notebooks in `notebooks/` to:
   - Process raw surface data
   - Calculate emission properties
   - Generate visualizations

## Data Format

Raw surface data should be provided in tab-delimited files with the following naming convention:
- Shortwave data: `{ID}_sw_{details}.tab`
- Longwave data: `{ID}_lw_{details}.tab`
- Combined data: `{ID}_combine_{details}.tab`

Example: 
- `B19_lherzolite_sw_cpfb08.tab`
- `B19_lherzolite_lw_nafb8p.tab`
- `B19_lherzolite_combine_cpfb08p_nafb8p.tab` 