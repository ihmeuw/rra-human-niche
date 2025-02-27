import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

YEARS = [
    "2011-2040",
    "2041-2070",
    "2071-2100",
]
SCENARIOS = [
    "ssp126",
    "ssp370",
    "ssp585",
]


def plot1(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(20, 12), nrows=3, ncols=4)

    for col, threshold in enumerate(["1", "2.5", "5", "10"]):
        for row, measure in enumerate(["occupancy", "pixel_density", "availability"]):
            ax = axes[row, col]
            df = data.loc[(threshold, "2011-2040", "ssp126", measure)]  # type: ignore[index]
            df = df.loc[10:35, 0:2500].sort_index(ascending=False)
            bd_threshold = 0.01
            df[df < bd_threshold] = np.nan
            sns.heatmap(np.power(df, 1 / 2), ax=ax, cmap="jet", cbar=False)

            ax.set_xlabel(None)
            ax.set_ylabel(None)
            if row == 0:
                col_name = f"Threshold {threshold}"
                ax.set_title(col_name, fontsize=18)
            if col == 0:
                row_name = {
                    "occupancy": "Population",
                    "population_density": "Population Density",
                    "pixel_occupancy": "Inhabited Area",
                    "pixel_density": "Inhabited Area Density",
                    "availability": "Available Niche",
                }[measure]
                ax.set_ylabel(row_name, fontsize=16)

    fig.supxlabel("Mean Annual Precipitation (mm)", fontsize=20)
    fig.supylabel("Mean Annual Temperature (\u00b0C)", fontsize=20)
    fig.suptitle("Kenya SSP585 (RCP-8.5)", fontsize=24)


def plot2(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(20, 15), nrows=5, ncols=3)
    for col, year in enumerate(YEARS):
        for row, measure in enumerate(
            [
                "occupancy",
                "population_density",
                "pixel_occupancy",
                "pixel_density",
                "availability",
            ]
        ):
            scenario = "ssp126" if col == 0 else "ssp585"
            ax = axes[row, col]
            df = data.loc[(year, scenario, measure)]  # type: ignore[index]
            df = df.loc[10:35, 0:2500].sort_index(ascending=False)
            sns.heatmap(np.power(df, 1 / 3), ax=ax, cmap="jet", cbar=False)

            ax.set_xlabel(None)
            ax.set_ylabel(None)
            if row == 0:
                col_name = {0: "Historical", 1: "2065", 2: "2085"}[col]
                ax.set_title(col_name, fontsize=18)
            if col == 0:
                row_name = {
                    "occupancy": "Population",
                    "population_density": "Population Density",
                    "pixel_occupancy": "Inhabited Area",
                    "pixel_density": "Inhabited Area Density",
                    "availability": "Available Niche",
                }[measure]
                ax.set_ylabel(row_name, fontsize=16)

    fig.supxlabel("Mean Annual Precipitation (mm)", fontsize=20)
    fig.supylabel("Mean Annual Temperature (\u00b0C)", fontsize=20)
    fig.suptitle("Kenya SSP585 (RCP-8.5)", fontsize=24)


def plot3(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=3)

    # dummy axes 1
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.patch.set_visible(False)
    ax.set_ylabel(
        "Available Niche (sq. km.)", color="firebrick", fontsize=20, labelpad=50
    )

    # dummy axes 2 for right ylabel
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.patch.set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Population (millions)", color="dodgerblue", fontsize=20, labelpad=50)

    for row in [0, 1]:
        for col, year in enumerate(YEARS):
            scenario = "ssp126" if col == 0 else "ssp585"

            measure, axis = {
                0: ("Mean Annual Temperature (\u00b0C)", 1),
                1: ("Mean Annual Precipitation (mm)", 0),
            }[row]

            ax = axes[row, col]
            scale = 40**2 / 1000**2
            av = (
                data.loc[(year, scenario, "availability")]  # type: ignore[index]
                .sort_index(ascending=False)
                .sum(axis=axis)  # type: ignore[arg-type]
                * scale
            )
            ax.bar(
                x=av.index,
                height=av,
                color="firebrick",
                alpha=0.5,
                width=40 if axis == 0 else 0.5,
            )
            ax.set_ylim(0, 65000)
            ax.yaxis.label.set_color("firebrick")
            ax.tick_params(axis="y", colors="firebrick")
            if col == 1:
                ax.set_xlabel(measure, fontsize=18)

            ax2 = ax.twinx()
            oc = (
                data.loc[(year, scenario, "occupancy")]  # type: ignore[index]
                .sort_index(ascending=False)
                .sum(axis=axis)  # type: ignore[arg-type]
                / 1_000_000
            )
            ax2.bar(
                x=oc.index,
                height=oc,
                color="dodgerblue",
                alpha=0.5,
                width=40 if axis == 0 else 0.5,
            )
            ax2.set_ylim(0, 5)
            ax2.yaxis.label.set_color("dodgerblue")
            ax2.tick_params(axis="y", colors="dodgerblue")

            if row == 0:
                col_name = {0: "Historical", 1: "2065", 2: "2085"}[col]
                ax.set_title(col_name, fontsize=18)

    fig.suptitle("Kenya SSP585 (RCP-8.5)", fontsize=24)


def plot4(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=2)

    for col in [0, 1]:
        for row, data_type in enumerate(["occupancy", "availability"]):
            ax = axes[row, col]

            measure, axis = {
                0: ("Mean Annual Temperature (\u00b0C)", 1),
                1: ("Mean Annual Precipitation (mm)", 0),
            }[col]

            for year, color in zip(
                YEARS, ["firebrick", "dodgerblue", "forestgreen"], strict=False
            ):
                scenario = "ssp126" if year == "2011-2040" else "ssp585"
                norm = 40**2 / 1000**2 if data_type == "availability" else 1 / 1_000_000
                df = (
                    data.loc[(year, scenario, data_type)]  # type: ignore[index]
                    .sort_index(ascending=False)
                    .sum(axis=axis)  # type: ignore[arg-type]
                    * norm
                )

                df.iloc[-1] = 0
                ax.bar(
                    df.index,
                    height=df,
                    color=color,
                    alpha=0.3,
                    width=40 if axis == 0 else 0.5,
                )
                ylims = {0: (0, 5), 1: (0, 65000)}[row]
                ax.set_ylim(ylims)

            if row == 0 and col == 0:
                ax.set_ylabel("Population (millions)", fontsize=16)
            elif row == 1 and col == 0:
                ax.set_ylabel("Available Niche (sq. km.)", fontsize=16)
                ax.set_xlabel(measure, fontsize=16)
            elif row == 1 and col == 1:
                ax.set_xlabel(measure, fontsize=16)

    handles = [
        mlines.Line2D([], [], color=color, alpha=0.5, label=label, linewidth=2.5)
        for label, color in zip(
            ["Historical", "2065", "2085"],
            ["firebrick", "dodgerblue", "forestgreen"],
            strict=False,
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=12,
        frameon=False,
        ncol=len(handles),
    )
    fig.suptitle("Kenya SSP585 (RCP-8.5)", fontsize=24)
