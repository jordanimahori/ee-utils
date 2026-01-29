# GEE Imagery Exporter

Utilities for sampling from Google Earth Engine imagery.

## Installation

```bash
python -m pip install -e .
```

## Package layout

- `src/ee_utils/geo.py`: CRS helpers.
- `src/ee_utils/sampling.py`: Patch and covariate extraction.
- `src/ee_utils/viz.py`: Pre-export imagery previews and neighbourhood plotting for extracted patches.
- `src/ee_utils/imagery/`: Data accessors for Landsat, Sentinel-2, and spatial covariates.
