import click

from rra_population_pipelines.pipelines.models.human_niche.tasks import RUNNERS, TASKS


@click.group()
def hnrun() -> None:
    """Run a stage of the human niche pipeline."""


for name, runner in RUNNERS.items():
    hnrun.add_command(runner, name=name)


@click.group()
def hntask() -> None:
    """Run an individual modeling task in the human niche pipeline."""


for name, task in TASKS.items():
    hntask.add_command(task, name=name)
