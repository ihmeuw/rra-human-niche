from rra_population_pipelines.pipelines.models.human_niche.tasks.make_climate_niche_data import (
    make_climate_niche_data,
    make_climate_niche_data_task,
)

RUNNERS = {
    "make_climate_niche_data": make_climate_niche_data,
}

TASKS = {
    "make_climate_niche_data": make_climate_niche_data_task,
}
