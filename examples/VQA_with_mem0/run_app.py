# Import required modules and components
from pathlib import Path

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

logging.init_logger("omagent", "omagent", level="INFO")

# Import agent-specific components
from agent.input_interface.input_interface import InputInterface

# Set current working directory path
CURRENT_PATH = root_path = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath("container.yaml"))


# Initialize workflow with new structure
workflow = ConductorWorkflow(name="VQA_with_mem0")

# Configure workflow tasks
task1 = simple_task(
    task_def_name="InputInterface", 
    task_reference_name="input_task"
)

task2 = simple_task(
    task_def_name="MemoryDecisionWorker",
    task_reference_name="memory_decision",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task3 = simple_task(
    task_def_name="MultimodalQueryGenerator",
    task_reference_name="multimodal_query",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task4 = simple_task(
    task_def_name="MemorySearch",
    task_reference_name="memory_search",
    inputs={
        "user_instruction": task1.output("user_instruction"),
    },
)

task5 = simple_task(
    task_def_name="VQAAnswerGenerator",
    task_reference_name="answer_generator",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task6 = simple_task(
    task_def_name="OutputFormatter",
    task_reference_name="output_formatter"
)

# Configure workflow with switch task
workflow >> task1 >> task2

# Create switch task for routing based on memory_decision output
switch_task = SwitchTask(
    task_ref_name="memory_decision_switch",
    case_expression=task2.output("final_decision")
)

# Add switch cases
switch_task.switch_case("multimodal_query_generator", task3)  # When we need to generate a multimodal query
switch_task.switch_case("memory_search", task4)  # When we need to search memory with text-only query
switch_task.switch_case("answer_generator", task5)  # When we need to generate an answer without memory search
switch_task.switch_case("output_formatter", task6)  # When we need to output the final answer

# Connect task2 to switch_task
task2 >> switch_task

# Connect remaining flow
task3 >> task4  # Multimodal query generator to memory search
task4 >> task5  # Memory search to answer generator

# Register workflow
workflow.register(True)

# Initialize and start app client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=workflow, config_path=config_path, workers=[InputInterface()]
)
agent_client.start_interactor()
