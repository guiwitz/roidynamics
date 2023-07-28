[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guiwitz/splitmask/master?urlpath=lab)
[![PyPI](https://img.shields.io/pypi/v/splitmask.svg?color=green)](https://pypi.org/project/splitmask)
[![CI](https://github.com/guiwitz/splitmask/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/guiwitz/splitmask/actions/workflows/test_and_deploy.yml)

# splitmask

Small Python package providing tools to analyze the dynamics of intensity in time-lapse microscopy images split into regions of specific geometries.  

## Installation

You can install this package directly using: 

```
pip install splitmask
```

**Note**: If you encounter errors when trying to use the ND2 format because of errors related to unusual ROIs (non-square), you can try to install an alternative version with:

```
pip install git+https://github.com/guiwitz/nd2reader.git@master#egg=nd2reader -U
```

## Authors

This package has been created by Guillaume Witz, Data Science Lab, University of Bern in collaboration with Jakobus van Unen, Pertz lab, Institute of Cell Biology, University of Bern.
