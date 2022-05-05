# OpenAP Trajectory Optimizer

This repository contains the flight trajectory optimizer module of [OpenAP](https://github.com/junzis/openap)

## Install

Install the development branch from GitHub:

```sh
pip install --upgrade git+https://github.com/junzis/openap-top
```

Install the latest stable release from pypi:

```sh
pip install --upgrade openap-top
```

## Quick start

Examples:

```python
from openap import top

optimizer = top.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)

flight = optimizer.trajectory(objective="fuel")
flight = optimizer.trajectory(objective="ci:30")
flight = optimizer.trajectory(objective="gwp100")
flight = optimizer.trajectory(objective="gtp100")
```