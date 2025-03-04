# Import required modules and components
from pathlib import Path

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
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

# Define workflow tasks:
# 1. Get input from user (image and question)
task1 = simple_task(task_def_name="InputInterface", task_reference_name="input_interface")

# 2. Decide if memory search is needed and if image is required
task2 = simple_task(
    task_def_name="MemoryDecisionWorker",
    task_reference_name="memory_decision",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 3. Generate multimodal query for memory search if needed
task3 = simple_task(
    task_def_name="MultimodalQueryGenerator",
    task_reference_name="multimodal_query_generator",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 4. Search memory based on query (with unique reference names for each path)
task4_0 = simple_task(
    task_def_name="MemorySearch",
    task_reference_name="memory_search_0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task4_1 = simple_task(
    task_def_name="MemorySearch",
    task_reference_name="memory_search_1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 5. Generate answer with image and memory context
task5_0 = simple_task(
    task_def_name="VQAAnswerGenerator",
    task_reference_name="answer_generator_0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task5_1 = simple_task(
    task_def_name="VQAAnswerGenerator",
    task_reference_name="answer_generator_1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 6. Generate text-only answer (no image needed)
task6 = simple_task(
    task_def_name="TextAnswerGenerator",
    task_reference_name="text_answer_generator",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 7. Store memory if needed (with unique reference names)
task7_0 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store_0",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task7_1 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store_1",
    inputs={"user_instruction": task1.output("user_instruction")},
)

task7_2 = simple_task(
    task_def_name="MemoryStore",
    task_reference_name="memory_store_2",
    inputs={"user_instruction": task1.output("user_instruction")},
)

# 8. Format and output the final answer
task8 = simple_task(
    task_def_name="OutputFormatter",
    task_reference_name="output_formatter"
)

# 9. Check if user wants to exit the conversation
task9 = simple_task(
    task_def_name="ExitChecker",
    task_reference_name="exit_checker"
)

# Create switch task for routing based on memory_decision output
switch_task = SwitchTask(
    task_ref_name="memory_decision_switch",
    case_expression="${memory_decision.final_decision}"
)

# Add switch cases with task lists
switch_task.switch_case("multimodal_query_generator", [task3, task4_0, task5_0, task7_0])
switch_task.switch_case("memory_search", [task4_1, task6, task7_1])
switch_task.switch_case("answer_generator", [task5_1, task7_2])
switch_task.switch_case("text_answer_generator", [task6, task7_1])

# Create outer loop that continues until user types "退出"
conversation_loop = DoWhileTask(
    task_ref_name="conversation_loop",
    tasks=[task1, task2, switch_task, task8, task9],
    termination_condition='if ($.exit_checker["should_exit"] == true){false;} else {true;} ',
)

# Create the main workflow sequence
workflow >> conversation_loop

# Register workflow
workflow.register(True)

# Initialize and start app client with workflow configuration
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=conversation_loop, config_path=config_path, workers=[InputInterface(), OutputFormatter()]
)
agent_client.start_interactor()
