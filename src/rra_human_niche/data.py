from pathlib import Path

import xarray as xr
from rra_tools.shell_tools import mkdir

from rra_population_pipelines.shared.data import RRA_POP

DEFAULT_MODEL_ROOT = RRA_POP.modeling_root / "human_niche"


class HumanNicheData:
    """Data loader for the human niche model."""

    def __init__(self, root: str | Path = DEFAULT_MODEL_ROOT):
        self.root = Path(root)
        self._create_model_root()

    def _create_model_root(self) -> None:
        mkdir(self.root, exist_ok=True)

    @property
    def niche_data(self) -> Path:
        return Path(self.root, "niche_data")

    @property
    def diagnostics(self) -> Path:
        return Path(self.root, "diagnostics")

    def save_niche_data(
        self,
        iso3: str,
        year: int | str,
        scenario: str,
        niche_data: xr.Dataset,
    ) -> None:
        out_root = self.niche_data / iso3
        mkdir(out_root, exist_ok=True)
        out_path = out_root / f"{scenario}_{year}.nc"
        niche_data.to_netcdf(out_path)

    def load_niche_data(
        self,
        iso3: str,
        year: int | str,
        scenario: str,
        temperature_variable: str,
        *,
        lazy: bool = True,
    ) -> xr.Dataset:
        in_path = self.niche_data / iso3 / f"{scenario}_{year}.nc"

        if lazy:  # noqa: SIM108
            ds = xr.open_dataset(in_path)
        else:
            ds = xr.load_dataset(in_path)

        ds = ds.sel(scenario=scenario, year=int(year)).drop(["scenario", "year"])
        collapse_var = "dat" if temperature_variable == "temperature" else "tas"
        return ds.sum(collapse_var)
