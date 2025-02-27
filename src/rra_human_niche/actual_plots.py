# ruff: noqa
# mypy: ignore-errors
import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import numpy as np
import rasterra as rt
import tqdm
import xarray as xr

from rra_population_pipelines.shared.data import RRA_POP, RRAPopulationData
from rra_population_pipelines.shared.plot_utils import strip_axes


def get_location_id(iso3: str) -> int:
    if iso3 == "KEN":
        return 180
    return -111


def load_ihme_admin1_shapes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame()


def make_single_scenario_climate_gif(iso3: str) -> None:
    """Plot temperature, days over thirty, and precipitation through time as a gif."""
    location_id = get_location_id(iso3)
    admins = load_ihme_admin1_shapes()
    admin0 = admins[admins.loc_id == location_id]

    scenario, stitle = "ssp245", "SSP2/RCP4.5"
    in_path = ()
    dat = xr.open_dataset(
        RRA_POP.projected_climate_data / iso3 / f"days_over_thirty_{scenario}.nc"
    ).rename({"tas": "value"})
    pr = xr.open_dataset(
        RRA_POP.projected_climate_data / iso3 / f"precipitation_{scenario}.nc"
    ).rename({"pr": "value"})
    tas = xr.open_dataset(
        RRA_POP.projected_climate_data / iso3 / f"temperature_{scenario}.nc"
    ).rename({"tas": "value"})

    for year in tqdm.trange(2017, 2101):
        fig, axes = plt.subplots(figsize=(5, 10), nrows=3)
        data = [
            (dat, "Days over 30\u2103", -1, 0, 365, "plasma"),
            (pr, "Precipitation (mm)", np.nan, 0, 5000, "summer_r"),
            (tas, "Temperature (\u2103)", np.nan, 15, 30, "plasma"),
        ]

        for ax, (ds, title, nodata, vmin, vmax, cmap) in zip(axes, data, strict=False):
            raster = rt.RasterArray(ds.sel(year=year)["value"], nodata).mask(
                admin0.unary_union
            )

            raster.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_ylabel(title, fontsize=14)
            strip_axes(ax)

        fig.suptitle(f"{stitle} - {year}", fontsize=16)
        fig.tight_layout()
        out_path = f"diagnostics/climate/{scenario}_{year}.png"
        fig.savefig(out_path)
        plt.close(fig)

    images = [
        imageio.imread(f"diagnostics/climate/{scenario}_{year}.png")
        for year in tqdm.trange(2017, 2101)
    ]
    imageio.mimsave(f"diagnostics/climate_{scenario}.gif", images, duration=0.2)


def make_all_scenarios_climate_gif(
    pop_data: RRAPopulationData, climate_path_template: str
) -> None:
    """Same as above for all scenarios"""
    admin0 = pop_data.load_admin_boundaries("KEN", 0)
    for year in tqdm.trange(2015, 2101):
        fig, axes = plt.subplots(figsize=(13, 8), ncols=4, nrows=3)

        for col, scenario in enumerate(["ssp126", "ssp245", "ssp370", "ssp585"]):
            dat = xr.open_dataset(
                climate_path_template.format(
                    measure="days_over_thirty", scenario=scenario
                )
            ).rename({"tas": "value"})
            pr = xr.open_dataset(
                climate_path_template.format(measure="precipitation", scenario=scenario)
            ).rename({"pr": "value"})
            tas = xr.open_dataset(
                climate_path_template.format(measure="temperature", scenario=scenario)
            ).rename({"tas": "value"})

            data = [
                (dat, "Days over 30C", -1, 0, 200, "plasma"),
                (pr, "Precipitation", np.nan, 0, 5000, "summer"),
                (tas, "Temperature", np.nan, 15, 30, "plasma"),
            ]

            for row, (ds, title, nodata, vmin, vmax, cmap) in enumerate(data):
                ax = axes[row, col]
                raster = rt.RasterArray(ds.sel(year=year)["value"], nodata).mask(
                    [admin0.unary_union], nodata
                )
                raster.plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)

                if col == 0:
                    ax.set_ylabel(title)
                if row == 0:
                    ax.set_title(scenario)

        fig.suptitle(year)
        out_path = f"diagnostics/climate/{year}.png"
        fig.savefig(out_path)
        plt.close(fig)

    images = []
    for year in tqdm.trange(2015, 2101):
        images.append(imageio.imread(f"diagnostics/climate/{year}.png"))
    imageio.mimsave("diagnostics/climate.gif", images, duration=0.2)
