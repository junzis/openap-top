# OpenAP Trajectory Optimizer

This repository contains the flight trajectory optimizer module of [OpenAP](https://github.com/junzis/openap)

## Install

Install the development branch from GitHub:

```sh
pip install --upgrade git+https://github.com/junzis/openap-otop
```

Install the latest stable release from pypi:

```sh
pip install --upgrade openap-top
```

## Quick start

Examples:

```python
import openap.top as otop

optimizer = otop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)

flight = optimizer.trajectory(objective="fuel")
flight = optimizer.trajectory(objective="ci:30")
flight = optimizer.trajectory(objective="gwp100")
flight = optimizer.trajectory(objective="gtp100")
```

An optimal flight example:

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)