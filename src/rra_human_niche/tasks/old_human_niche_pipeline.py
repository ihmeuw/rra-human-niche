# ruff: noqa
# mypy: ignore-errors
import itertools
from pathlib import Path

import geopandas as gpd
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterra as rt
import seaborn as sns
import tqdm
import xarray as xr
from rra_tools import parallel

from rra_population_pipelines.shared.plot_utils import strip_axes
from rra_population_pipelines.shared.data import RRAPopulationData

SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
YEARS = tuple(range(2015, 2101))


def load_niche_datasets(
    scenarios: list[str] = SCENARIOS,
    years: list[int] = YEARS,
) -> dict[tuple[str, int, str], xr.Dataset]:
    datasets = {}
    scenarios_and_years = list(itertools.product(scenarios, years))

    for scenario, year in tqdm.tqdm(scenarios_and_years):
        path = f"diagnostics/data/niche_{scenario}_{year}.nc"
        ds = (
            xr.load_dataset(path)
            .sel(scenario=scenario, year=year)
            .drop(["scenario", "year"])
        )
        for collapse_var in ["dat", "tas"]:
            keep_var = "dat" if collapse_var == "tas" else "tas"
            datasets[(scenario, year, keep_var)] = ds.sum(collapse_var)
    return datasets


def load_pop_forecast(location_name: str = "Kenya") -> pd.Series:
    path = "IHME_POP_2017_2100_POP_REFERENCE_Y2020M05D01.CSV"
    df = pd.read_csv(path)
    df = df[(df.location_name == location_name) & (df.age_group_name == "All Ages")]
    return (
        df.groupby("year_id")["val"]
        .sum()
        .reindex(range(2015, 2101))
        .interpolate(method="slinear", fill_value="extrapolate", limit_direction="both")
    )


def make_heatmap(ds, scenario, year, keep_var, *, save: bool = False) -> None:
    data = {
        ">5km from major river": ds.sel(rwd=0).drop("rwd"),
        "<5km from major river": ds.sel(rwd=1).drop("rwd"),
        "Combined": ds.sum("rwd"),
    }

    fig, axes = plt.subplots(figsize=(12, 8), nrows=3, ncols=3)

    edges = {
        "Available Land (sq. km)": np.array([0, 250, 500, 1000, 2000, 5000, 10000]),
        "Population (per sq. km)": np.array([5, 25, 50, 100, 250, 500, 1000]),
        "Occupancy Rate (%)": np.array([0.01, 0.1, 1, 5, 10, 25, 50]),
    }

    total_land = data["Combined"]["availability"]
    total_land_area = total_land * 40**2 / 1000**2

    for col, (river_label, dataset) in enumerate(data.items()):
        available_land = dataset["availability"] * 40**2 / 1000**2
        population_density = dataset["occupancy"] / total_land_area
        occupancy_rate = dataset["pixel_occupancy"] / total_land

        measures = {
            "Occupancy Rate (%)": np.round(occupancy_rate * 100, 1),
            "Available Land (sq. km)": available_land,
            "Population (per sq. km)": population_density,
        }

        for row, (measure, arr) in enumerate(measures.items()):
            ax = axes[row, col]
            e = edges[measure] + 1e-5
            cmap = mpl.cm.jet
            norm = mpl.colors.BoundaryNorm(e, cmap.N)
            im = arr.plot(ax=ax, cmap=cmap, norm=norm, extend="neither")
            im.cmap.set_under("lightgrey")
            im.cmap.set_bad("lightgrey")
            im.colorbar.ax.set_ylabel(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            if col == 0:
                ax.set_ylabel(measure.replace("_", " ").title(), fontsize=12)
            if row == 0:
                ax.set_title(river_label, fontsize=12)
            if keep_var == "tas":
                ax.set_ylim(10, 32)
            else:
                ax.set_ylim(-1, 370)
            ax.set_xlim(0, 2500)
            sns.despine(ax=ax, left=True, bottom=True)

    if keep_var == "tas":
        fig.supylabel("Mean Annual Temperature (\u2103)", fontsize=15)
    else:
        fig.supylabel("Days above 30\u2103", fontsize=15)

    fig.supxlabel("Precipitation(mm)", fontsize=15)
    tscenario = {
        "ssp126": "RCP 2.6",
        "ssp245": "RCP 4.5",
        "ssp370": "RCP 7.0",
        "ssp585": "RCP 8.5",
    }[scenario]
    fig.suptitle(f"{tscenario} - {year}", fontsize=18)

    if not save:
        plt.show()
    else:
        out_path = f"diagnostics/heatmap/{keep_var}_{scenario}_{year}.png"
        fig.savefig(out_path)
        plt.close(fig)


def make_heatmap_parallel(in_data: tuple[xr.Dataset, str, int, str]) -> None:
    make_heatmap(*in_data, save=True)


def make_heatmap_gifs(datasets: dict[tuple[str, int, str], xr.Dataset]) -> None:
    _ = parallel.run_parallel(
        make_heatmap_parallel,
        [(ds, *key) for key, ds in datasets.items()],
        num_cores=20,
        progress_bar=True,
    )

    scenarios_and_keep_vars = list(
        itertools.product(["ssp126", "ssp245", "ssp370", "ssp585"], ["tas", "dat"])
    )
    for scenario, keep_var in tqdm.tqdm(scenarios_and_keep_vars):
        images = [
            imageio.imread(f"diagnostics/heatmap/{keep_var}_{scenario}_{year}.png")
            for year in range(2015, 2101)
        ]
        imageio.mimsave(
            f"diagnostics/{keep_var}_{scenario}_heatmap.gif", images, duration=0.3
        )


def make_single_scenario_climate_gif(
    pop_data: RRAPopulationData, climate_path_template: str
):
    admin0 = pop_data.load_admin_boundaries("KEN", 0)
    scenario, stitle = "ssp245", "SSP2/RCP4.5"
    dat = xr.open_dataset(
        climate_path_template.format(measure="days_over_thirty", scenario=scenario)
    ).rename({"tas": "value"})
    pr = xr.open_dataset(
        climate_path_template.format(measure="precipitation", scenario=scenario)
    ).rename({"pr": "value"})
    tas = xr.open_dataset(
        climate_path_template.format(measure="temperature", scenario=scenario)
    ).rename({"tas": "value"})

    for year in tqdm.trange(2023, 2024):
        fig, axes = plt.subplots(figsize=(5, 10), nrows=3)
        data = [
            (dat, "Days over 30\u2103", -1, 0, 365, "plasma"),
            (pr, "Precipitation (mm)", np.nan, 0, 5000, "summer_r"),
            (tas, "Temperature (\u2103)", np.nan, 15, 30, "plasma"),
        ]

        for ax, (ds, title, nodata, vmin, vmax, cmap) in zip(axes, data, strict=False):
            raster = rt.RasterArray(ds.sel(year=year)["value"], nodata).mask(
                [admin0.unary_union]
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
        for year in range(2015, 2101)
    ]
    imageio.mimsave(f"diagnostics/climate_{scenario}.gif", images, duration=0.2)


def make_all_scenarios_climate_gif(
    pop_data: RRAPopulationData, climate_path_template: str
) -> None:
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

    images = [
        imageio.imread(f"diagnostics/climate/{year}.png") for year in range(2015, 2101)
    ]
    imageio.mimsave("diagnostics/climate.gif", images, duration=0.2)


def compute_measures(dataset):
    available_land = dataset["availability"] * 40**2 / 1000**2
    population_density = dataset["occupancy"] / available_land
    occupancy_rate = np.round(
        100 * dataset["pixel_occupancy"] / dataset["availability"], 1
    )

    return {
        "Available Land (sq. km)": available_land,
        "Population (per sq. km)": population_density,
        "Occupancy Rate (%)": occupancy_rate,
    }


def plot_hmap(data, axis, cmap, norm):
    im = data.plot(ax=axis, cmap=cmap, norm=norm, extend="neither")
    im.cmap.set_under("lightgrey")
    im.cmap.set_bad("lightgrey")
    im.colorbar.ax.set_ylabel(None)
    axis.set_xlabel(None)
    axis.set_ylabel(None)
    return data


def make_heatmap2(ds_dat, ds_tas, scenario, year, save=False):
    fig, axes = plt.subplots(figsize=(5, 8), nrows=2)

    measure = "Occupancy Rate (%)"

    dat = compute_measures(ds_dat)[measure]
    tas = compute_measures(ds_tas)[measure]

    edges = {
        "Available Land (sq. km)": np.array([0, 250, 500, 1000, 2000, 5000, 10000]),
        "Population (per sq. km)": np.array([5, 25, 50, 100, 250, 500, 1000]),
        "Occupancy Rate (%)": np.array([0.01, 0.1, 1, 5, 10, 25, 50]),
    }[measure] + 1e-5
    cmap = mpl.cm.jet
    norm = mpl.colors.BoundaryNorm(edges, cmap.N)

    ax = axes[0]
    plot_hmap(tas, ax, cmap, norm)
    ax.set_ylabel("Mean Annual Temperature (\u2103)")
    ax.set_xlabel("Precipitation(mm)")
    ax.set_ylim(10, 32)
    ax.set_xlim(100, 2500)
    sns.despine(ax=ax, left=True, bottom=True)

    ax = axes[1]
    plot_hmap(dat, ax, cmap, norm)
    ax.set_ylabel("Days above 30\u2103")
    ax.set_xlabel("Precipitation(mm)")
    ax.set_ylim(-1, 370)
    ax.set_xlim(100, 2500)
    sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle(f"Occupancy Rate (%)\n{scenario} - {year}", fontsize=16)
    fig.tight_layout()

    if not save:
        plt.show()
    else:
        keep_var = "tas"
        out_path = f"diagnostics/heatmap/{keep_var}_{scenario}_{year}.png"
        fig.savefig(out_path)
        plt.close(fig)


def load_upsample_template():
    raster_template_path = "/mnt/team/rapidresponse/pub/population/data/03-processed-data/human-niche/chelsa/downscaled-projections/KEN/precipitation_ssp126.nc"
    raster_template = rt.RasterArray(
        xr.open_dataset(raster_template_path).sel(year=2019)["pr"],
        nodata=np.nan,
    )
    return raster_template


def load_upsampled_pop(pop_data: RRAPopulationData) -> rt.RasterArray:
    cache_path = Path("upsampled_pop.tif")
    if not cache_path.exists():
        pop = pop_data.load_population("KEN")
        upsample_template = load_upsample_template()
        upsampled_pop = pop.resample_to(upsample_template, "sum")
        upsampled_pop.to_file(cache_path)
    return rt.load_raster(cache_path)


def load_upsampled_rivers_mask():
    cache_path = Path("upsampled_rivers_mask.tif")
    if not cache_path.exists():
        rivers = rt.load_raster(
            "/mnt/share/homes/mfiking/population/gis/population/distance_covariates/hydrosheds/rivers/hydrorivers_flow4.tif"
        )
        upsample_template = load_upsample_template()
        upsampled_rivers = rivers.resample_to(upsample_template, "nearest")
        rivers_mask = rt.RasterArray(
            (
                (upsampled_rivers.to_numpy() * upsampled_rivers.x_resolution / 1000) < 5
            ).astype(int),
            transform=upsampled_rivers.transform,
            crs=rivers.crs,
            no_data_value=-1,
        )
        rivers_mask.to_file(cache_path)
    return rt.load_raster(cache_path)


def coerce_to_index(arr: xr.DataArray, data_linspace: np.ndarray):
    index = np.digitize(arr, data_linspace)
    over_mask = index == len(data_linspace)
    index[over_mask] -= 1
    index = np.transpose(index, (1, 2, 0))
    index = xr.Dataset(
        data_vars={"value": (["lat", "lon", "year"], index)}, coords=arr.coords
    )["value"]
    return index


def main():
    datasets = load_niche_datasets()
    pop_forecast = load_pop_forecast()
    admin0 = pop_data.load_admin_boundaries("KEN", 0)
    pop = pop_data.load_population("KEN")
    rivers_mask = load_upsampled_rivers_mask()
    upscaled_pop = load_upsampled_pop()

    scenario = "ssp245"
    temperature_variable = "tas"
    year = 2019
    key = (scenario, year, temperature_variable)
    niche_variables = datasets[key]

    precipitation = pop_data.load_climate_projection("KEN", "pr", scenario).sel(
        year=slice(year, 2101)
    )
    p_index = coerce_to_index(
        precipitation["value"], niche_variables.coords["pr"].to_numpy()
    )
    temperature = pop_data.load_climate_projection(
        "KEN", temperature_variable, scenario
    ).sel(year=slice(year, 2101))
    t_index = coerce_to_index(
        temperature["value"], niche_variables.coords[temperature_variable].to_numpy()
    )


def prep_heatmap_data(dataset):
    available_land = dataset["availability"] * 40**2 / 1000**2
    dataset["occupancy"] / available_land
    occupancy_rate = 100 * dataset["pixel_occupancy"] / dataset["availability"]
    100 * dataset["building_occupancy"] / dataset["availability"]
    100 * dataset["building_pixel_occupancy"] / dataset["availability"]

    measures = {
        #'Available Land (sq. km)':       (available_land,        np.geomspace(1, 10000, cmap_steps)),
        #'Population (count per sq. km)': (population_density,    np.array([0.000001, 1, 3, 10, 25, 100, 500])),
        #'Built Area (%)':                (built_area,            np.array([0.000001, 0.0005, 0.001, 0.002, 0.005, 0.025, 0.1, 0.5])),
        "Occupancy Rate (%)": (
            occupancy_rate,
            np.array([0.000001, 0.005, 0.015, 0.02, 0.05, 0.5, 2.5, 10, 50]),
        ),
        #'Building Occupancy Rate (%)':   (built_occupancy_rate,  np.geomspace(0.005, 0.05, cmap_steps)),
    }
    return measures


def make_niche_heatmap(niche_arr, edges, ax, title, **kwargs):
    im = niche_arr.plot(ax=ax, extend="neither", **kwargs)

    im.colorbar.ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(10, 35)
    ax.set_xlim(0, 2500)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_ylabel("Mean Annual Temperature (\u2103)", fontsize=12)
    ax.set_xlabel("Precipitation(mm)", fontsize=12)
    return im, ax


def make_threshold_line_plot(df, column, ax, cmap, norm, y_max=1.0):
    data = df.set_index(["Year", "Threshold"])[column].unstack()

    for thresh in data.columns:
        ax.plot(data.index, data[thresh], color=cmap(norm(thresh - 1e-5)))
        ax.set_ylim(0, y_max)
        ax.set_ylabel(column, fontsize=12)
        ax.set_xlabel("Year")
    return ax


def compute_rate_ratio(niche_arr, temperature_index, precipitation_index):
    niche_map = niche_arr.values[
        temperature_index.to_numpy(), precipitation_index.to_numpy()
    ]
    rate_ratio = 1 - (niche_map + 1e-5) / (niche_map[:, :, [0]] + 1e-5)
    rate_ratio[:, :, 0] = 0.0
    rate_ratio = (
        xr.Dataset(
            data_vars={"rr": (["lat", "lon", "year"], rate_ratio)},
            coords=temperature_index.coords,
        )
        .fillna(1)
        .clip(0, 1)
    )
    return rate_ratio["rr"]


def compute_climate_stress_map(
    niche_arr, temperature_index, precipitation_index, threshold
):
    stressed = 1 - (niche_arr > threshold).astype(int)
    stressed_map = stressed.values[
        temperature_index.to_numpy(), precipitation_index.to_numpy()
    ]
    stressed_map = xr.Dataset(
        data_vars={"stressed": (["lat", "lon", "year"], stressed_map)},
        coords=temperature_index.coords,
    ).astype(bool)
    return stressed_map["stressed"]


def compute_population_table(upscaled_pop, stressed_map, move_ratio):
    years = np.asarray(stressed_map.coords["year"])
    total_pop = upscaled_pop.to_numpy().sum()
    data = []
    for y_idx, year in enumerate(years):
        stressed_mask = stressed_map[:, :, y_idx]
        pop = upscaled_pop.to_numpy()[stressed_mask]
        rr = np.asarray(move_ratio[:, :, y_idx])[stressed_mask]

        pop_to_move = (rr * pop).sum()
        pop = pop.sum()
        data.append([year, pop_to_move / total_pop, (pop - pop_to_move) / total_pop])
    df = pd.DataFrame(
        data, columns=["Year", "Proportion to Move", "Proportion at Risk"]
    )
    return df


def compute_threshold_population_table(upscaled_pop, stressed, move_ratio):
    data = []
    for threshold, stressed_map in stressed.items():
        threshold_df = compute_population_table(upscaled_pop, stressed_map, move_ratio)
        threshold_df["Threshold"] = threshold
        data.append(threshold_df)
    df = pd.concat(data)
    return df


def make_summary_table(df, ax):
    at_risk = df.set_index(["Year", "Threshold"])["Proportion at Risk"].unstack()
    to_move = df.set_index(["Year", "Threshold"])["Proportion to Move"].unstack()
    start_risk = 100 * at_risk.loc[2019].rename("At Risk (2019)")
    end_risk = 100 * at_risk.loc[2100].rename("At Risk (2100)")
    end_moved = 100 * to_move.loc[2100].rename("Moved (2100)")
    change = ((end_risk + end_moved) - start_risk).rename("Risk Change")
    bounds = pd.concat([start_risk, end_risk, end_moved, change], axis=1).reset_index()

    cell_text = []
    for r in range(len(bounds)):
        cell_text.append(
            [
                f"{np.format_float_positional(bounds.iloc[r, c], precision=2, unique=False, fractional=False, trim='k')}"
                for c in range(len(bounds.columns))
            ]
        )

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        colLabels=bounds.columns,
        loc="upper center",
        edges="BT",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    return ax


def threshold_analysis(dataset, temperature_index, precipitation_index, upscaled_pop):
    measures = prep_heatmap_data(dataset)

    cmap = mpl.cm.viridis
    cmap.set_under("lightgrey")
    cmap.set_bad("lightgrey")

    nrows = 4
    fig, axes = plt.subplots(
        figsize=(7 * len(measures), 3 * nrows), ncols=len(measures), nrows=nrows
    )
    axes = axes.reshape((nrows, len(measures)))

    for col, (measure, (arr, edges)) in enumerate(measures.items()):
        ax = axes[0, col]
        norm = mpl.colors.BoundaryNorm(edges, cmap.N)
        make_niche_heatmap(arr, edges, ax, measure, cmap=cmap, norm=norm)

        if measure != "Available Land (sq. km)":
            move_ratio = compute_rate_ratio(arr, temperature_index, precipitation_index)
            stressed = {
                threshold: compute_climate_stress_map(
                    arr, temperature_index, precipitation_index, threshold
                )
                for threshold in tqdm.tqdm(edges[1:])
            }

            df = compute_threshold_population_table(upscaled_pop, stressed, move_ratio)

            make_threshold_line_plot(
                df, "Proportion at Risk", axes[1, col], cmap, norm, y_max=0.3
            )
            make_threshold_line_plot(
                df, "Proportion to Move", axes[2, col], cmap, norm, y_max=0.3
            )
            make_summary_table(df, axes[3, col])

        else:
            for row in range(1, 4):
                ax = axes[row, col].set_axis_off()

    axes[0, 0].set_ylabel("Mean Annual Temperature (\u2103)", fontsize=12)
    axes[0, 0].set_xlabel("Precipitation(mm)", fontsize=12)
    # axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_ylabel("Proportion at Risk", fontsize=12)
    fig.suptitle("RCP 4.5 - 2019", fontsize=18)
    fig.tight_layout()

    # plt.show()
    # else:
    #     out_path = f'diagnostics/heatmap/{keep_var}_{scenario}_{year}.png'
    #     fig.savefig(out_path)
    #     plt.close(fig)


threshold_analysis(niche_variables.sum("rwd"), t_index, p_index, upscaled_pop)
# make_heatmap3(niche_variables.sel(rwd=0), t_index, p_index, upscaled_pop)
# make_heatmap3(niche_variables.sel(rwd=1), t_index, p_index, upscaled_pop)

mombasa = gpd.GeoSeries([box(39.54, -0.51, 39.75, -0.37)], crs="EPSG:4326")
pop3 = pop.to_crs(mombasa.crs)


def habitability_map(axis, h_raster, a_raster, p_raster, boundary_gdf, pop_max=500):
    habitable_cmap = mpl.colors.ListedColormap(["white", "firebrick"])
    habitable_norm = mpl.colors.BoundaryNorm([0, 0.5, 1], habitable_cmap.N)

    nodata_mask = h_raster.no_data_mask
    extent = plotting_extent(h_raster.to_numpy(), h_raster.transform)

    axis.imshow(
        np.ma.masked_array(p_raster.to_numpy(), mask=nodata_mask),
        cmap="Greys",
        vmax=pop_max,
        extent=extent,
    )

    h = h_raster.to_numpy()
    a = a_raster.to_numpy()
    a[h < 0.8] = 0.5
    axis.imshow(
        np.ma.masked_array(h, mask=nodata_mask),
        cmap=habitable_cmap,
        norm=habitable_norm,
        alpha=0.5,  # a.clip(0.5, 1.0),
        extent=extent,
    )
    boundary_gdf.boundary.plot(ax=axis, color="k", linewidth=0.4)
    axis = strip_ax(axis)
    return axis


for scenario in ["ssp245"]:  # , 'ssp370', 'ssp585']:
    temperature_variable = "tas"
    year = 2019
    key = (scenario, year, temperature_variable)
    niche_variables = datasets[key]

    precipitation = pop_data.load_climate_projection("KEN", "pr", scenario).sel(
        year=slice(year, 2101)
    )
    p_index = coerce_to_index(
        precipitation["value"], niche_variables.coords["pr"].to_numpy()
    )
    temperature = pop_data.load_climate_projection(
        "KEN", temperature_variable, scenario
    ).sel(year=slice(year, 2101))
    t_index = coerce_to_index(
        temperature["value"], niche_variables.coords[temperature_variable].to_numpy()
    )

    arr, _ = prep_heatmap_data(niche_variables.sum("rwd"))["Occupancy Rate (%)"]
    move_ratio = compute_rate_ratio(arr, t_index, p_index)
    stressed = compute_climate_stress_map(arr, t_index, p_index, threshold=0.015)
    df = compute_population_table(upscaled_pop, stressed, move_ratio).set_index("Year")
    pops = pd.concat(
        [
            df.mul(pop_forecast.loc[df.index], axis=0).cumsum(axis=1),
            pop_forecast.loc[df.index].rename("Population"),
        ],
        axis=1,
    )

    for year in tqdm.trange(2019, 2101):
        fig = plt.figure(figsize=(16, 9))
        grid_spec = fig.add_gridspec(
            ncols=2,
            width_ratios=[8, 8],
            wspace=0.2,
        )

        habitable_r = to_raster(
            stressed.sel(year=year).drop("year").astype(int), -1
        ).mask(admin0)
        rr_r = to_raster(move_ratio.sel(year=year).drop("year"), 0).mask(admin0)

        ax_map = fig.add_subplot(grid_spec[0, 0])
        ax_map = habitability_map(
            ax_map,
            habitable_r,
            rr_r,
            upscaled_pop,
            admin0,
        )
        # mombasa.boundary.plot(ax=ax_map, color='k')

        gs_right = grid_spec[0, 1].subgridspec(3, 1, height_ratios=[2, 4, 2])

        # pop_subset_r = pop3.clip(mombasa)
        # habitable_subset_r = habitable_r.clip(mombasa).resample_to(pop_subset_r)
        # rr_subset_r = rr_r.clip(mombasa).resample_to(pop_subset_r)

        # # ax_inset = fig.add_subplot(gs_right[1])
        # # habitability_map(
        #     ax_inset,
        #     habitable_subset_r,
        #     rr_subset_r,
        #     pop_subset_r,
        #     admin0.intersection(mombasa),
        #     pop_max=25,
        # )

        # minx, miny, maxx, maxy = mombasa.total_bounds
        # for coords in [(maxx, maxy), (minx, miny)]:
        #     con = ConnectionPatch(
        #         xyA=coords,
        #         xyB=coords,
        #         coordsA="data",
        #         coordsB="data",
        #         axesA=ax_map,
        #         axesB=ax_inset,
        #         arrowstyle="-",
        #         linewidth=0.4
        #     )
        #     ax_inset.add_artist(con)

        ax_lines = fig.add_subplot(gs_right[1])

        bold = pops.loc[:year]
        ax_lines.plot(bold.index, bold["Population"], color="k")
        ax_lines.plot(bold.index, bold["Proportion at Risk"], color="#560E0B")
        ax_lines.plot(bold.index, bold["Proportion to Move"], color="#560E0B")
        ax_lines.fill_between(
            bold.index, bold["Population"], bold["Proportion at Risk"], color="#9EA0A1"
        )
        ax_lines.fill_between(
            bold.index,
            bold["Proportion at Risk"],
            bold["Proportion to Move"],
            color="firebrick",
            alpha=0.6,
        )
        ax_lines.fill_between(
            bold.index, bold["Proportion to Move"], color="firebrick", alpha=0.9
        )

        non_bold = pops.loc[year:]
        ax_lines.plot(non_bold.index, non_bold["Population"], color="k", alpha=0.6)
        ax_lines.plot(
            non_bold.index, non_bold["Proportion at Risk"], color="#560E0B", alpha=0.6
        )
        ax_lines.plot(non_bold.index, non_bold["Proportion to Move"], color="#560E0B")
        ax_lines.fill_between(
            non_bold.index,
            non_bold["Population"],
            non_bold["Proportion at Risk"],
            color="k",
            alpha=0.2,
        )
        ax_lines.fill_between(
            non_bold.index, non_bold["Proportion at Risk"], color="firebrick", alpha=0.2
        )
        ax_lines.fill_between(
            bold.index, bold["Proportion to Move"], color="firebrick", alpha=0.4
        )

        text_year = 2050
        ax_lines.text(
            text_year,
            6.3e7,
            f"Total population\n{bold.loc[year, 'Population'] / 1e6:.2f} million",
            color="k",
            fontsize=16,
        )
        ypos = 3e7 if scenario == "ssp585" else 2e7
        ax_lines.text(
            text_year,
            ypos,
            f"Climate-stressed population\n{bold.loc[year, 'Proportion at Risk'] / 1e6:.2f} million",
            color="#560E0B",
            fontsize=16,
        )
        ax_lines.text(
            text_year,
            -1.6e7,
            f"Potential climate migrants\n{bold.loc[year, 'Proportion to Move'] / 1e6:.2f} million",
            color="#560E0B",
            fontsize=16,
        )
        strip_ax(ax_lines)
        ax_lines.vlines(year, *ax_lines.get_ylim(), color="k")
        ax_lines.text(year, 9e7, year, color="k", fontsize=20)

        pos = ax_lines.get_position()
        pos.x0 -= 0.02  # for example 0.2, choose your value
        ax_lines.set_position(pos)

        out_path = f"diagnostics/habitability/{year}.png"
        fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    images = []
    for year in tqdm.trange(2019, 2101):
        images.append(iio.imread(f"diagnostics/habitability/{year}.png"))
    durations = [100] * len(images)
    durations[-1] = 3000
    iio.imwrite(
        f"diagnostics/habitability_{scenario}.gif", images, duration=durations, loop=4
    )
    print(f"Duration: {sum(durations[:-1]) / 1000}s\nEnd: {durations[-1] / 1000}")
