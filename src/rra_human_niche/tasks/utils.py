from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

MSFT_BLUE_BLACK = "#091F2C"

temperature = np.arange(-10, 40, 0.5)
rainfall = np.arange(40, 4040, 40)


def framify(
    data: list[tuple[Any, ...]], y: str | int, sc: str, measure: str
) -> pd.DataFrame:
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


@numba.njit  # type: ignore[misc]
def invert_niche(
    temperature_index: npt.NDArray[np.int_],
    rainfall_index: npt.NDArray[np.int_],
    niche_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool]:
    niche_map = np.zeros_like(temperature_index, dtype="bool")
    for row in range(niche_map.shape[0]):
        for col in range(niche_map.shape[1]):
            t_idx = temperature_index[row, col]
            r_idx = rainfall_index[row, col]
            niche_map[row, col] = niche_mask[t_idx, r_idx]
    return niche_map
