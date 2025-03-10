# Import required modules and components
from pathlib import Path

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

logging.init_logger("omagent", "omagent", level="INFO")

# Import agent-specific components
from agent.input_interface.input_interface import NavigationInputInterface
from agent.depth_processor.depth_processor import DepthProcessor

# Set current working directory path
CURRENT_PATH = root_path = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath("container.yaml"))

# Initialize workflow
workflow = ConductorWorkflow(name="navigation_general")

# Configure workflow tasks
task1 = simple_task(
    task_def_name="NavigationInputInterface", 
    task_reference_name="input_task"
)

task2 = simple_task(
    task_def_name="DepthProcessor",
    task_reference_name="depth_processor",
    inputs={"image_path": task1.output("image_path")},
)

# Connect workflow - simplified to just input and depth processing
workflow >> task1 >> task2

# Register workflow
workflow.register(True)

# Initialize and start app client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=workflow, 
    config_path=config_path, 
    workers=[
        NavigationInputInterface(),
        DepthProcessor(),
    ]
)
agent_client.start_interactor() 