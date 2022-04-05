# OpenAP Trajectory Optimizer

This repository contains the flight trajectory optimizer module of [OpenAP](https://github.com/junzis/openap)

## Install

Install the development branch from GitHub:

```sh
pip install --upgrade git+https://github.com/junzis/openap-opti
```

Install the latest stable release from pypi:

```sh
pip install --upgrade openap-opti
```

## Quick start

Examples:

```python
from openap.opti import CruiseOptimizer
cro = CruiseOptimizer("A320", "EHAM", "LEMD", takeoff_mass_factor=0.8)
df = cro.trajectory(objective="ci:10")
```