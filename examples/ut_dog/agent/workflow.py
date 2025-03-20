from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from agent.workers.plannning.planning import Planning


def construct_workflow():
    # Initialize simple VQA workflow
    workflow = ConductorWorkflow(name="ut_dog")

    # Configure workflow tasks:
    # 1. Input interface for user interaction
    # 2. Simple VQA processing based on user input
    planning = simple_task(
        task_def_name=Planning,
        task_reference_name="planning",
    )

    excute = simple_task(task_def_name="ReactExcute", task_reference_name="react_task")

    react_loop = DoWhileTask(
    task_ref_name="react_loop",
    tasks=[excute],
    termination_condition='if ($.react_task["all_tasks_finished"] == true){false;} else {true;} ',
)

    # Configure workflow execution flow: Input -> VQA
    workflow >> planning >> react_loop

    # Register workflow
    workflow.register(True)

    return workflow
