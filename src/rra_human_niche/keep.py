# ruff: noqa
# mypy: ignore-errors
import itertools
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import rasterio
import tqdm

MSFT_BLUE_BLACK = "#091F2C"


def load_variable(
    variable: str, year: str, scenario: str, scale: float, offset: float
) -> np.ndarray:
    path = Path("KEN") / f"CHELSA_{variable}_{year}_composite_{scenario}_V.2.1.tif"
    with rasterio.open(path) as src:
        data = scale * src.read(1).flatten() + offset
    return data


def convert_to_index(
    data: np.ndarray, var_min: float, var_max: float, dvar: float
) -> np.ndarray:
    def to_raw_idx(x):
        return (x - var_min) / dvar

    idx_min = to_raw_idx(var_min)
    idx_max = to_raw_idx(var_max)
    raw_idx = np.ceil(to_raw_idx(data))
    return np.minimum(np.maximum(raw_idx, idx_min), idx_max).astype("uint16")


@numba.njit
def calculate_availability_and_occupancy(temperature_index, rainfall_index, population):
    availability = np.zeros((100, 100))
    occupancy = np.zeros((100, 100))
    pixel_occupancy = np.zeros((100, 100))
    for i in range(len(temperature_index)):
        row = temperature_index[i]
        col = rainfall_index[i]
        p = population[i]
        availability[row, col] += 1
        occupancy[row, col] += p
        if p > 10:
            pixel_occupancy[row, col] += 1
    return availability, occupancy, pixel_occupancy


def make_availability_and_occupancy(
    population: np.ndarray, mask: np.ndarray, year: str, scenario: str
):
    temp_data = load_variable("bio1", year, scenario, scale=0.1, offset=-273.15)[mask]
    temp_idx = convert_to_index(temp_data, -9.5, 40, 0.5)
    rain_data = load_variable("bio12", year, scenario, scale=0.1, offset=0.0)[mask]
    rain_idx = convert_to_index(rain_data, 40, 4000, 40)
    return calculate_availability_and_occupancy(temp_idx, rain_idx, population)


availability = {}
occupancy = {}
pixel_occupancy = {}
for year, scenario in tqdm.tqdm(list(itertools.product(YEARS, SCENARIOS))):
    key = (year, scenario)
    av, oc, p_oc = make_availability_and_occupancy(pop, mask, year, scenario)
    availability[key] = av
    occupancy[key] = oc
    pixel_occupancy[key] = p_oc

temperature = np.arange(-10, 40, 0.5)
rainfall = np.arange(40, 4040, 40)


def framify(data, y, sc, measure):
    df = pd.DataFrame(data, columns=rainfall.astype(str), index=temperature)
    df.index.name = "Temperature (\u00b0C)"
    df.columns.name = "Precipitation (mm)"
    df["year"] = y
    df["scenario"] = sc
    df["measure"] = measure
    df = df.reset_index().set_index(
        ["year", "scenario", "measure", "Temperature (\u00b0C)"]
    )
    return df


dfs = []
for year, scenario in tqdm.tqdm(availability):
    dfs.append(framify(availability[(year, scenario)], year, scenario, "availability"))
    dfs.append(framify(occupancy[(year, scenario)], year, scenario, "occupancy"))
    dfs.append(
        framify(pixel_occupancy[(year, scenario)], year, scenario, "pixel_occupancy")
    )
df = pd.concat(dfs)

df.to_parquet("data10.parquet", engine="fastparquet")


@numba.njit
def invert_niche(temperature_index, rainfall_index, niche_mask):
    niche_map = np.zeros_like(temperature_index, dtype="bool")
    for row in range(niche_map.shape[0]):
        for col in range(niche_map.shape[1]):
            t_idx = temperature_index[row, col]
            r_idx = rainfall_index[row, col]
            niche_map[row, col] = niche_mask[t_idx, r_idx]
    return niche_map
