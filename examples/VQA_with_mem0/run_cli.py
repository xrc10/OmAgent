# Import required modules and components
from pathlib import Path

from agent.input_interface.input_interface import InputInterface
from omagent_core.clients.devices.cli.client import DefaultClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath("container.yaml"))


# Initialize simple VQA workflow
workflow = ConductorWorkflow(name="VQA_with_mem0")

# Configure workflow tasks
task1 = simple_task(task_def_name="InputInterface", task_reference_name="input_task_vqa_mem0")
task2 = simple_task(
    task_def_name="VQAMemorySearch",
    task_reference_name="memory_search_vqa_mem0",
    inputs={"user_instruction": task1.output("user_instruction")},
)
task3 = simple_task(
    task_def_name="VQAAnswerGenerator",
    task_reference_name="answer_generator_vqa_mem0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# Configure workflow execution flow: Input -> Memory Search -> Answer Generation
workflow >> task1 >> task2 >> task3

# Register workflow
workflow.register(True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
cli_client = DefaultClient(
    interactor=workflow, config_path=config_path, workers=[InputInterface()]
)
cli_client.start_interactor()
