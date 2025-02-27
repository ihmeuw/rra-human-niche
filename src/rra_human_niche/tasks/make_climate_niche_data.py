import click
import numba
import numpy as np
import numpy.typing as npt
import rasterra as rt
import xarray as xr
from rra_tools import jobmon

from rra_population_pipelines.pipelines.models.human_niche.data import (
    DEFAULT_MODEL_ROOT as HN_MODEL_ROOT,
)
from rra_population_pipelines.pipelines.models.human_niche.data import HumanNicheData
from rra_population_pipelines.pipelines.models.people_per_structure.data import (
    PEOPLE_PER_STRUCTURE_ROOT as PPS_MODEL_ROOT,
)
from rra_population_pipelines.pipelines.models.people_per_structure.data import (
    PeoplePerStructureData,
)
from rra_population_pipelines.shared.cli_tools import options as clio
from rra_population_pipelines.shared.data import RRA_DATA_ROOT, RRAPopulationData
from rra_population_pipelines.shared.formatting import xarray_to_raster

VALID_YEARS = [str(y) for y in range(2017, 2101)]


def load_rivers() -> rt.RasterArray:
    """Load rivers raster and return a mask of rivers within 5 km of a pixel.

    Need to generalize the code that made the og version of this for Kenya.
    """
    rivers_in_path = "tbd"
    rivers = rt.load_raster(rivers_in_path).to_crs("ESRI:54009")
    rivers_data = rivers.to_numpy()
    walking_distance = 5  # km
    rivers_mask = rt.RasterArray(
        ((rivers_data * rivers.x_resolution / 1000) < walking_distance).astype(int),
        transform=rivers.transform,
        crs=rivers.crs,
        no_data_value=-1,
    )
    rivers_mask._ndarray[rivers.no_data_mask] = -1  # noqa: SLF001

    return rivers_mask


@numba.njit  # type: ignore[misc]
def calculate_availability_and_occupancy(
    temperature_index: npt.NDArray[np.int8],
    dat_index: npt.NDArray[np.int8],
    rainfall_index: npt.NDArray[np.int8],
    no_data_mask: npt.NDArray[np.bool_],
    population: npt.NDArray[np.float64],
    buildings: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
]:
    data_shape = (100, 100, 100)
    availability = np.zeros(data_shape, dtype=np.int32)
    occupancy = np.zeros(data_shape, dtype=np.float64)
    pixel_occupancy = np.zeros(data_shape, dtype=np.int32)
    building_occupancy = np.zeros(data_shape, dtype=np.float64)
    building_pixel_occupancy = np.zeros(data_shape, dtype=np.int32)

    for i in range(len(temperature_index)):
        if no_data_mask[i]:
            continue

        x = temperature_index[i]
        y = dat_index[i]
        z = rainfall_index[i]

        p = population[i]
        b = buildings[i]

        availability[x, y, z] += 1
        occupancy[x, y, z] += p
        if p > 1:
            pixel_occupancy[x, y, z] += 1
        building_occupancy[x, y, z] += b
        min_density = 0.01
        if b > min_density:
            building_pixel_occupancy[x, y, z] += 1
    return (
        availability,
        occupancy,
        pixel_occupancy,
        building_occupancy,
        building_pixel_occupancy,
    )


def make_climate_niche_data_main(
    iso3: str,
    year: str,
    climate_scenario: str,
    pop_data_root: str,
    pps_model_root: str,
    hn_model_root: str,
) -> None:
    pop_data = RRAPopulationData(pop_data_root)
    pop_year = min(2023, int(year))
    climate_year = int(year)

    admin1 = pop_data.load_shapefile(1, iso3, 2022).to_crs("ESRI:54009")

    pps_data = PeoplePerStructureData(pps_model_root)
    modeling_frame = pps_data.load_modeling_frame()
    population = pps_data.load_predictions(iso3, f"{pop_year}q3", "raked_population")  # type: ignore[attr-defined]

    a1_ch_polys = admin1.explode(index_parts=True).convex_hull.unary_union
    modeling_frame_subset = modeling_frame[modeling_frame.intersects(a1_ch_polys)]
    a0_poly = admin1.unary_union
    modeling_frame = modeling_frame_subset[modeling_frame_subset.intersects(a0_poly)]
    tile_keys = modeling_frame.tile_key.tolist()
    building_density = pps_data.load_feature(
        40, tile_keys, f"{pop_year}q3", "building_density"
    )

    variables = [
        ("temperature", "tas", np.nan, (-10, 40), "tas"),
        ("days_over_thirty", "tas", -1, (0, 400), "dat"),
        ("precipitation", "pr", np.nan, (0, 5000), "pr"),
    ]

    idx_vars = []
    ranges = {}
    print("prepping data")
    for in_name, in_name_short, null_val, var_range, out_name in variables:
        variable = pop_data.load_climate_data(in_name, iso3, climate_scenario).sel(  # type: ignore[attr-defined]
            year=climate_year
        )[in_name_short]
        raster = (
            xarray_to_raster(variable, null_val)
            .to_crs(admin1.crs)
            .mask(a0_poly)
            .resample_to(population)
        )
        v_range = np.linspace(*var_range, num=100, endpoint=False)
        ranges[out_name] = v_range
        idx = np.digitize(raster.to_numpy().flatten(), v_range).astype(np.int8)  # type: ignore[arg-type]
        idx_vars.append(idx)

    no_data_mask = population.no_data_mask.astype(np.int8)

    print("calculating")
    (
        availability,
        occupancy,
        pixel_occupancy,
        building_occupancy,
        building_pixel_occupancy,
    ) = calculate_availability_and_occupancy(
        *idx_vars,
        no_data_mask.flatten(),
        population.to_numpy().flatten(),
        building_density.to_numpy().flatten(),
    )

    coord_names = ["tas", "dat", "pr", "year", "scenario"]
    ds = xr.Dataset(
        data_vars={
            "availability": (
                coord_names,
                availability[:, :, :, np.newaxis, np.newaxis],
            ),
            "occupancy": (coord_names, occupancy[:, :, :, np.newaxis, np.newaxis]),
            "pixel_occupancy": (
                coord_names,
                pixel_occupancy[:, :, :, np.newaxis, np.newaxis],
            ),
            "building_occupancy": (
                coord_names,
                building_occupancy[:, :, :, np.newaxis, np.newaxis],
            ),
            "building_pixel_occupancy": (
                coord_names,
                building_pixel_occupancy[:, :, :, np.newaxis, np.newaxis],
            ),
        },
        coords={
            **ranges,
            "year": [climate_year],
            "scenario": [climate_scenario],
        },
    )

    hn_data = HumanNicheData(hn_model_root)
    print("Writing")
    hn_data.save_niche_data(iso3, climate_year, climate_scenario, ds)


@click.command()
@clio.with_iso3(allow_all=False)
@clio.with_year(allow_all=False, choices=VALID_YEARS)
@clio.with_climate_scenario(allow_all=False)
@clio.with_input_directory("pop-data", RRA_DATA_ROOT)
@clio.with_input_directory("pps-data", PPS_MODEL_ROOT)
@clio.with_output_directory(HN_MODEL_ROOT)
def make_climate_niche_data_task(
    iso3: str,
    year: str,
    climate_scenario: str,
    pop_data_dir: str,
    pps_data_dir: str,
    output_dir: str,
) -> None:
    make_climate_niche_data_main(
        iso3,
        year,
        climate_scenario,
        pop_data_dir,
        pps_data_dir,
        output_dir,
    )


@click.command()
@clio.with_iso3(allow_all=False)
@clio.with_year(choices=VALID_YEARS)
@clio.with_climate_scenario()
@clio.with_input_directory("pop-data", RRA_DATA_ROOT)
@clio.with_input_directory("pps-data", PPS_MODEL_ROOT)
@clio.with_output_directory(HN_MODEL_ROOT)
@clio.with_queue()
def make_climate_niche_data(
    iso3: str,
    year: str,
    climate_scenario: str,
    pop_data_dir: str,
    pps_data_dir: str,
    output_dir: str,
    queue: str,
) -> None:
    hn_data = HumanNicheData(output_dir)
    years = list(VALID_YEARS) if year == clio.RUN_ALL else [year]
    scenarios = (
        list(clio.VALID_CLIMATE_SCENARIOS)
        if climate_scenario == clio.RUN_ALL
        else [climate_scenario]
    )

    jobmon.run_parallel(
        task_name="make_climate_niche_data",
        node_args={
            "iso3": [iso3],
            "year": years,
            "climate-scenario": scenarios,
        },
        task_args={
            "output-dir": output_dir,
            "pop-data-dir": pop_data_dir,
            "pps-data-dir": pps_data_dir,
        },
        task_resources={
            "queue": queue,
            "cores": 5,
            "memory": "100G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        runner="hntask",
        log_root=hn_data.niche_data,
    )
