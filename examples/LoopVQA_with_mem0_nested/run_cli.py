# Import core modules for workflow management and configuration
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.clients.devices.cli.client import DefaultClient
from omagent_core.utils.registry import registry

logging.init_logger("omagent", "omagent", level="INFO")

from pathlib import Path
import os
import sys

CURRENT_PATH = Path(__file__).parents[0]

# Import registry and CLI client modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

# Add parent directory to Python path
sys.path.append(os.path.abspath(CURRENT_PATH.joinpath("../../")))

# Import custom input interface worker
from agent.input_interface.input_interface import InputInterface
from agent.vqa_with_mem0.nested_worker import NestedWorker

# Configure Redis storage and load container settings
container.register_stm("RedisSTM")
container.from_config(CURRENT_PATH.joinpath("container.yaml"))

# Initialize VQA with memory workflow
workflow = ConductorWorkflow(name="loopvqa_with_mem0_nested")

# Define simplified workflow tasks:
# 1. Get input from user (image and question)
task1 = simple_task(task_def_name="InputInterface", task_reference_name="input_interface")

# 2. Process the input with the nested worker that handles all VQA functionality
task2 = simple_task(task_def_name="NestedWorker", 
                    task_reference_name="nested_worker",
                    inputs={"user_instruction": task1.output("user_instruction")})

# Create outer loop that continues until user types "退出"
conversation_loop = DoWhileTask(
    task_ref_name="conversation_loop",
    tasks=[task1, task2],
    termination_condition='if ($.nested_worker["should_exit"] == true){false;} else {true;} ',
)

# Create the main workflow sequence
workflow >> conversation_loop

# Register workflow with conductor server
workflow.register(True)

# Initialize and start CLI client with workflow and input interface worker
config_path = CURRENT_PATH.joinpath("configs")
cli_client = DefaultClient(
    interactor=workflow, config_path=config_path, workers=[InputInterface(), NestedWorker()]
)
cli_client.start_interactor()
