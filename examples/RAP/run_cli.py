# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.cli.client import DefaultClient
from omagent_core.engine.workflow.task.set_variable_task import SetVariableTask
from omagent_core.utils.logger import logging
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask

from agent.input_interface.input_interface import InputInterface

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
container.from_config(CURRENT_PATH.joinpath('container.yaml'))



# Initialize simple VQA workflow
workflow = ConductorWorkflow(name='RAP')

# Configure workflow tasks:
client_input_task = simple_task(task_def_name='InputInterface', task_reference_name='input_task')

task_selection = simple_task(task_def_name='Selection', task_reference_name='selection')

task_expansion = simple_task(task_def_name='Expansion', task_reference_name='expansion')

task_simulation_preprocess = simple_task(task_def_name='SimulationPreProcess', task_reference_name='simulation_preprocess')

task_expansion2 = simple_task(task_def_name='Expansion', task_reference_name='expansion2')

task_simulation_postprocess = simple_task(task_def_name='SimulationPostProcess', task_reference_name='simulation_postprocess')

simulation_loop = DoWhileTask(task_ref_name='simulation_loop', tasks=[task_expansion2, task_simulation_postprocess], 
                             termination_condition='if ($.simulation_postprocess["finish"] == true){false;} else {true;} ')

task_back_propagation = simple_task(task_def_name='BackPropagation', task_reference_name='back_propagation')

task_check = simple_task(task_def_name="MCTSCompletionCheck", task_reference_name="mcts_completion_check")

mcts_loop = DoWhileTask(task_ref_name='mcts_loop', tasks=[task_selection, task_expansion, task_simulation_preprocess, simulation_loop, task_back_propagation, task_check], 
                             termination_condition='if ($.mcts_completion_check["finish"] == true){false;} else {true;} ')


output_interface = simple_task(task_def_name='OutputInterface', task_reference_name='output_interface')

# Configure workflow execution flow: 
workflow >> client_input_task >> mcts_loop >> output_interface

# Register workflow
workflow.register(True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath('configs')
cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[InputInterface()])
cli_client.start_interactor()
