# Import core OmAgent components for workflow management and app functionality
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.clients.devices.app.client import AppClient
from omagent_core.utils.logger import logging
logging.init_logger("omagent", "omagent", level="INFO")

from omagent_core.utils.registry import registry

from pathlib import Path
CURRENT_PATH = Path(__file__).parents[0]

# Import and register worker modules from agent directory
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

# Add parent directory to Python path for imports
import sys
import os
sys.path.append(os.path.abspath(CURRENT_PATH.joinpath('../../')))

# Import custom image input worker
from agent.find_object_input.find_object_input import FindObjectInput
from agent.find_object_detection.find_object_detection import FindObjectDetection
from agent.find_object_location.find_object_location import FindObjectLocation

# Import task type for implementing loops in workflow
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask


# Configure container with Redis storage and load settings
container.register_stm("RedisSTM")
container.from_config(CURRENT_PATH.joinpath('container.yaml'))


# Initialize workflow for finding objects
workflow = ConductorWorkflow(name='find_object_workflow')

# Define workflow tasks:
# 1. Get initial image input and object to find
task1 = simple_task(task_def_name='FindObjectInput', task_reference_name='image_input')

# 2. Detect if requested object exists in image
task2 = simple_task(task_def_name='FindObjectDetection', task_reference_name='object_detection')

# 3. If object exists, determine its location
task3 = simple_task(task_def_name='FindObjectLocation', task_reference_name='object_location')

# Create loop that continues until object is found
# Loop terminates when object is detected or user wants to stop
find_object_loop = DoWhileTask(
    task_ref_name='find_object_loop', 
    tasks=[task1, task2], 
    termination_condition='if ($.object_detection["stop_search"] == true) {false;} else {true;}'
)

# Define workflow sequence: find object loop -> location determination (if found)
workflow >> find_object_loop >> task3

# Register workflow with conductor server
workflow.register(True)

# Initialize and start app client with workflow and workers
config_path = CURRENT_PATH.joinpath('configs')
agent_client = AppClient(
    interactor=workflow, 
    config_path=config_path, 
    workers=[FindObjectInput()]
)
agent_client.start_interactor()
