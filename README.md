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
from openap import opti

optimizer = opti.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)

flight = optimizer.trajectory(objective="fuel")
flight = optimizer.trajectory(objective="ci:30")
flight = optimizer.trajectory(objective="gwp100")
flight = optimizer.trajectory(objective="gtp100")
```