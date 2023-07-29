[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guiwitz/roidynamics/master?urlpath=lab)
[![PyPI](https://img.shields.io/pypi/v/roidynamics.svg?color=green)](https://pypi.org/project/roidynamics)
[![CI](https://github.com/guiwitz/roidynamics/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/guiwitz/roidynamics/actions/workflows/test_and_deploy.yml)

# roidynamics

Small Python package providing tools to analyze the dynamics of intensity in time-lapse microscopy images split into regions of specific geometries.  

## Installation

You can install this package directly using: 

```
pip install roidynamics
```

**Note**: If you encounter errors when trying to use the ND2 format because of errors related to unusual ROIs (non-square), you can try to install an alternative version with:

```
pip install git+https://github.com/guiwitz/nd2reader.git@master#egg=nd2reader -U
```

## Authors

This package has been created by Guillaume Witz, Data Science Lab, University of Bern in collaboration with Jakobus van Unen, Pertz lab, Institute of Cell Biology, University of Bern.
