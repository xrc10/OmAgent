from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from agent.workers.plannning.planning import Planning


def construct_workflow():
    # Initialize simple VQA workflow
    workflow = ConductorWorkflow(name="ut_dog")

    # Configure workflow tasks:
    # 1. Input interface for user interaction
    task1 = simple_task(task_def_name="InputInterface", task_reference_name="input_task")
    # 2. Simple VQA processing based on user input
    planning = simple_task(
        task_def_name=Planning,
        task_reference_name="planning",
    )

    # Configure workflow execution flow: Input -> VQA
    workflow >> planning

    # Register workflow
    workflow.register(True)

    return workflow
