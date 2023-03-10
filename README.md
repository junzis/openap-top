# OpenAP Trajectory Optimizer

This repository contains the flight trajectory optimizer module of [OpenAP](https://github.com/junzis/openap).

A more detailed user guide can be found on: https://openap.dev/top.


## Install

OpenAP.top uses `cfgrib` for integrating wind data, `cartopy` for plotting, and a few other libraries. I recommend using `conda` to install these dependencies. Following is an example how I set it up on my computer for testing.

1. Create a new conda environmental (`openap`), which avoids messing up the base conda environment:
```sh
conda create -n openap python=3.10 -c conda-forge
```
2. Use the `openap` environment
```sh
conda activate openap
```
3. Install dependent libraries:
```sh
conda install -c conda-forge cfgrib cartopy
```
4. Install the most recent version of `openap`:
```sh
pip install --upgrade git+https://github.com/junzis/openap
```
5. Install the most recent version of `openap-top`:
```sh
pip install --upgrade git+https://github.com/junzis/openap-top
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
windfield = otop.wind.read_grib(fgrib)

optimizer.enable_wind(windfield)

flight = optimizer.trajectory(objective="fuel")
```

If your grib file includes multiple timestamps, make sure to filter the correct time in the previous `windfield` object (pandas DataFrame).


### Example of an optimal flight:

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)
