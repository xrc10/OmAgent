# Import required modules and components
from pathlib import Path

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

# Import agent-specific components
from agent.input_interface.input_interface import InputInterface
from agent.output_formatter.output_formatter import OutputFormatter

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

# Split memory search into two tasks
task4_0 = simple_task(
    task_def_name="MemorySearch",
    task_reference_name="memory_search0",
    inputs={
        "user_instruction": task1.output("user_instruction"),
    },
)

task4_1 = simple_task(
    task_def_name="MemorySearch",
    task_reference_name="memory_search1",
    inputs={
        "user_instruction": task1.output("user_instruction"),
    },
)

# Split answer generator into three tasks
task5_0 = simple_task(
    task_def_name="TextAnswerGenerator",
    task_reference_name="text_answer_generator0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task5_1 = simple_task(
    task_def_name="TextAnswerGenerator",
    task_reference_name="text_answer_generator1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task5_2 = simple_task(
    task_def_name="VQAAnswerGenerator",
    task_reference_name="vqa_answer_generator1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# Add new task for text answer generator
task5_3 = simple_task(
    task_def_name="TextAnswerGenerator",
    task_reference_name="text_answer_generator2",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task6 = simple_task(
    task_def_name="OutputFormatter",
    task_reference_name="output_formatter"
)

# Add memory store task after answer generator tasks
task7_0 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task7_1 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task7_2 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store2",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# Add memory store task for text answers
task7_3 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store3",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# Create switch task for routing based on memory_decision output
switch_task = SwitchTask(
    task_ref_name="memory_decision_switch",
    case_expression=task2.output("final_decision")
)

# Add switch cases with task lists
switch_task.switch_case("multimodal_query_generator", [task3, task4_0, task5_0, task7_0])
switch_task.switch_case("memory_search", [task4_1, task5_1, task7_1])
switch_task.switch_case("answer_generator", [task5_2, task7_2])
switch_task.switch_case("text_answer_generator", [task5_3, task7_3])

# Connect workflow
workflow >> task1 >> task2 >> switch_task

# Register workflow
workflow.register(True)

# Initialize and start app client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=workflow, config_path=config_path, workers=[InputInterface(), OutputFormatter()]
)
agent_client.start_interactor()
