# OpenAP Trajectory Optimizer

This repository contains the flight trajectory optimizer module of [OpenAP](https://github.com/junzis/openap)

## Install

Install the development branch from GitHub:

```sh
pip install --upgrade git+https://github.com/junzis/openap-top
```

Install the stable release from pypi:

```sh
pip install --upgrade openap-top
```

## Quick start

Example code to generate a fuel optimal flight between two airports:

```python
import openap.top as otop

optimizer = otop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)

flight = optimizer.trajectory(objective="fuel")
```

You can specify different objective functions as:

```python
flight = optimizer.trajectory(objective="ci:30")
flight = optimizer.trajectory(objective="gwp100")
flight = optimizer.trajectory(objective="gtp100")
```

The final `flight` object is a pandas DataFrame.

## Use wind data

To enable wind in your optimizer, you must first download meteorological data in `grib` format from ECMWF, for example, the ERA5 data at https://doi.org/10.24381/cds.bd0915c6. 

Then enable the wind for the defined optimizer. 

Example code:

```python
import openap.top as otop

optimizer = otop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)

fgrib = "path_to_the_wind_data.grib"
windfield = top.wind.read_grib(fgrib)
op.enable_wind(windfield)

flight = optimizer.trajectory(objective="fuel")
```

If your grib file includes multiple timestamps, make sure to filter the correct time in the previous `windfield` object (pandas DataFrame).


### Example of an optimal flight:

Visualization of the previous `flight` using `matplotlib` and `cartopy`:

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)