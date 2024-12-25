# Import core OmAgent components
from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask

logging.init_logger("omagent", "omagent", level="INFO")

from pathlib import Path
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

# Import and register worker modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

import os
import sys
sys.path.append(os.path.abspath(CURRENT_PATH.joinpath("../../")))

# Import custom image input worker
from agent.rpg_image_input.rpg_image_input import RPGImageInput

# Configure container
container.register_stm("RedisSTM")
container.from_config(CURRENT_PATH.joinpath("container.yaml"))

# Initialize workflow
workflow = ConductorWorkflow(name="mm_rpg_game")

# Define workflow tasks:
# 1. Get initial image input
task1 = simple_task(task_def_name="RPGImageInput", task_reference_name="image_input")

# 2. Generate initial story and goals
task2 = simple_task(task_def_name="StoryGenerator", task_reference_name="story_generator")

# 3. Handle dialogue interaction
task3 = simple_task(task_def_name="RPGDialogue", task_reference_name="rpg_dialogue")

# 4. Check story progress
task4 = simple_task(task_def_name="StoryProgress", task_reference_name="story_progress")

# 5. Generate story ending
task5 = simple_task(task_def_name="StoryEnding", task_reference_name="story_ending")

# Create dialogue loop that continues until story is complete or max turns reached
dialogue_loop = DoWhileTask(
    task_ref_name="dialogue_loop",
    tasks=[task3, task4],
    termination_condition='if ($.story_progress["should_end"] == true){false;} else {true;} ',
)

# Define workflow sequence
workflow >> task1 >> task2 >> dialogue_loop >> task5

# Register workflow
workflow.register(True)

# Initialize and start app client
config_path = CURRENT_PATH.joinpath("configs")
agent_client = AppClient(
    interactor=workflow, 
    config_path=config_path, 
    workers=[RPGImageInput()]
)
agent_client.start_interactor() 