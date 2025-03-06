# Import core modules for workflow management and configuration
from pathlib import Path

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

# Import agent-specific components
from agent.input_interface.input_interface import InputInterface
from agent.vqa_with_mem0.nested_worker import NestedWorker

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

# Register STM and load config
container.register_stm("RedisSTM")
container.from_config(CURRENT_PATH.joinpath("container.yaml"))

# Initialize workflow with new structure
workflow = ConductorWorkflow(name="VQA_with_mem0_v2")

# Define simplified workflow tasks:
# 1. Get input from user (image and question)
task1 = simple_task(task_def_name="InputInterface", task_reference_name="input_interface")

# 2. Process the input with the nested worker that handles all VQA functionality
task2 = simple_task(task_def_name="NestedWorker", 
                    task_reference_name="nested_worker",
                    inputs={"user_instruction": task1.output("user_instruction")})

# Create outer loop that continues until user indicates they want to exit
conversation_loop = DoWhileTask(
    task_ref_name="conversation_loop",
    tasks=[task1, task2],
    termination_condition='if ($.nested_worker["should_exit"] == true){false;} else {true;} ',
)

# Create the main workflow sequence
workflow >> conversation_loop

# Register workflow with conductor server
workflow.register(True)

# Initialize and start app client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=workflow, 
    config_path=config_path, 
    workers=[InputInterface(), NestedWorker()]
)
agent_client.start_interactor()
