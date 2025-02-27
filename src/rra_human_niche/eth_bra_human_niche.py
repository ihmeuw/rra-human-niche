# ruff: noqa
# mypy: ignore-errors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

from rra_population_pipelines.pipelines.models.human_niche import data

####
from rra_population_pipelines.pipelines.models.human_niche.tasks.make_human_niche_plots import (
    coerce_to_index,
    compute_climate_stress_map,
    compute_population_table,
    compute_rate_ratio,
    compute_threshold_population_table,
    make_niche_heatmap,
    make_summary_table,
    make_threshold_line_plot,
    prep_heatmap_data,
)
from rra_population_pipelines.pipelines.models.people_per_structure.data import (
    PEOPLE_PER_STRUCTURE_ROOT as PPS_MODEL_ROOT,
)
from rra_population_pipelines.pipelines.models.people_per_structure.data import (
    PeoplePerStructureData,
)
from rra_population_pipelines.shared.data import RRA_DATA_ROOT, RRAPopulationData
from rra_population_pipelines.shared.formatting import to_raster
from rra_population_pipelines.shared.plot_utils import strip_axes

iso3 = "ETH"
scenario = "ssp245"
temperature_variable = "temperature"
start_year = "2023"
pop_data_dir = RRA_DATA_ROOT
pps_data_dir = PPS_MODEL_ROOT
output_dir = data.DEFAULT_MODEL_ROOT
progress_bar = True


pps_data = PeoplePerStructureData(pps_data_dir)
model_frame = pps_data.load_modeling_frame()

pop_data = RRAPopulationData(pop_data_dir)
pop_forecast = pop_data.load_location_population_forecast(iso3)
admin0 = pop_data.load_ihme_shape(iso3)

precipitation = pop_data.load_climate_data(
    "precipitation", iso3, scenario, lazy=False
).sel(year=slice(start_year, 2101))
temperature = pop_data.load_climate_data(
    temperature_variable, iso3, scenario, lazy=False
).sel(year=slice(start_year, 2101))

pps_data = PeoplePerStructureData(pps_data_dir)
pop_raster = pps_data.load_predictions(
    iso3,
    f"{start_year}q3",
    "upsampled_population",
)

hn_data = data.HumanNicheData(output_dir)
niche_data = hn_data.load_niche_data(
    iso3,
    start_year,
    scenario,
    temperature_variable,
    lazy=False,
)

pr_index = coerce_to_index(precipitation["pr"], niche_data.coords["pr"].to_numpy())
temp_index = coerce_to_index(temperature["tas"], niche_data.coords["tas"].to_numpy())

arr, _ = prep_heatmap_data(niche_data)["Occupancy Rate (%)"]
move_ratio = compute_rate_ratio(arr, temp_index, pr_index)
stressed = compute_climate_stress_map(arr, temp_index, pr_index, threshold=0.015)
df = compute_population_table(pop_raster, stressed, move_ratio).set_index("Year")

pops = pd.concat(
    [
        df.mul(pop_forecast.loc[df.index], axis=0).cumsum(axis=1),
        pop_forecast.loc[df.index].rename("Population"),  ##
    ],
    axis=1,
)

temp_raster = to_raster(temperature.sel(year=2023)["tas"], np.nan)
precip_raster = to_raster(precipitation.sel(year=2023)["pr"], np.nan)

fig, ax = plt.subplots(figsize=(15, 15))
temp_raster.mask(admin0).plot(cmap="hot_r", ax=ax)


def make_heatmap(ds, scenario, year, keep_var, save=False):
    fig, axes = plt.subplots(figsize=(15, 4), ncols=3)

    edges = {
        "Available Land (sq. km)": np.array([0, 250, 500, 1000, 2000, 5000, 10000]),
        "Population (per sq. km)": np.array([5, 25, 50, 100, 250, 500, 1000]),
        "Occupancy Rate (%)": np.array([0.01, 0.1, 1, 5, 10, 25, 50]),
    }

    total_land = ds["availability"]
    total_land_area = total_land * 40**2 / 1000**2

    available_land = ds["availability"] * 40**2 / 1000**2
    population_density = ds["occupancy"] / total_land_area
    occupancy_rate = ds["pixel_occupancy"] / total_land

    measures = {
        "Occupancy Rate (%)": np.round(occupancy_rate * 100, 1),
        "Available Land (sq. km)": available_land,
        "Population (per sq. km)": population_density,
    }

    for col, (measure, arr) in enumerate(measures.items()):
        ax = axes[col]
        e = edges[measure] + 1e-5
        cmap = mpl.cm.jet
        norm = mpl.colors.BoundaryNorm(e, cmap.N)
        im = arr.plot(ax=ax, cmap=cmap, norm=norm, extend="neither")
        im.cmap.set_under("lightgrey")
        im.cmap.set_bad("lightgrey")
        im.colorbar.ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        ax.set_title(measure.replace("_", " ").title(), fontsize=12)
        if keep_var == "tas":
            ax.set_ylim(10, 32)
        else:
            ax.set_ylim(-1, 370)
        ax.set_xlim(0, 2500)
        sns.despine(ax=ax, left=True, bottom=True)

    if keep_var == "tas":
        fig.supylabel("Mean Annual Temperature (\u2103)", fontsize=14)
    else:
        fig.supylabel("Days above 30\u2103", fontsize=14)

    fig.supxlabel("Precipitation(mm)", fontsize=14)
    tscenario = {
        "ssp126": "RCP 2.6",
        "ssp245": "RCP 4.5",
        "ssp370": "RCP 7.0",
        "ssp585": "RCP 8.5",
    }[scenario]
    fig.suptitle(f"{tscenario} - {year}", fontsize=18)

    if not save:
        fig.tight_layout()
        plt.show()
    else:
        out_path = f"diagnostics/heatmap/{keep_var}_{scenario}_{year}.png"
        fig.savefig(out_path)
        plt.close(fig)


make_heatmap(niche_data, scenario, 2023, "tas")


def make_summary_table(df, ax):
    at_risk = df.set_index(["Year", "Threshold"])["Proportion at Risk"].unstack()
    to_move = df.set_index(["Year", "Threshold"])["Proportion to Move"].unstack()
    start_risk = 100 * at_risk.loc[2023].rename("At Risk (2019)")
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
    axes[1, 0].set_ylabel("Proportion at Risk", fontsize=12)
    fig.suptitle("RCP 4.5 - 2019", fontsize=18)
    fig.tight_layout()


threshold_analysis(niche_data, temp_index, pr_index, pop_raster)

pps_data = PeoplePerStructureData(pps_data_dir)
pop_raster = pps_data.load_predictions(
    "BRA",
    f"{start_year}q3",
    "upsampled_population",
)

fig, ax = plt.subplots(figsize=(15, 15))
(pop_raster / 1000**2 * 100).plot(ax=ax, vmin=0.0001, vmax=1, cmap="viridis")
strip_axes(ax)
