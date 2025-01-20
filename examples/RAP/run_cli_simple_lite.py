# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.lite_version.cli import DefaultClient
from omagent_core.engine.workflow.task.set_variable_task import SetVariableTask
from omagent_core.utils.logger import logging
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.engine.worker.base import BaseWorker

from agent.input_interface.input_interface import InputInterface

SUPPORT_TASK = {
    'math': "please input your math problem."
}

@registry.register_worker()
class SimpleInput(BaseWorker):
    def _run(self, query, *args, **kwargs):
        # Get task selection from user
        task = input(f'\nWelcome to use OmAgent RAP Algorithm, please input the task you want to conduct. Choices: {list(SUPPORT_TASK.keys())} ')
        
        if task not in SUPPORT_TASK:
            raise NotImplementedError
        
        self.stm(self.workflow_instance_id)['task'] = task 

        # Use query parameter instead of reading new input
        data_input = query
        
        # For debug
        if data_input == 'x':
            data_input = "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
        
        logging.info(f'data_input: {data_input}\n')
        self.stm(self.workflow_instance_id)['data_input'] = data_input

        return {'data_input': data_input}

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

# Change STM to SharedMemSTM for lite version
container.register_stm("SharedMemSTM")

# Initialize RAP workflow with lite_version=True
workflow = ConductorWorkflow(name='RAP', lite_version=True)

# Configure workflow tasks:
client_input_task = simple_task(task_def_name=SimpleInput, task_reference_name='input_interface')

task_selection = simple_task(task_def_name='Selection', task_reference_name='selection')

task_expansion = simple_task(task_def_name='Expansion', task_reference_name='expansion')

task_simulation_preprocess = simple_task(task_def_name='SimulationPreProcess', task_reference_name='simulation_preprocess')

task_expansion2 = simple_task(task_def_name='Expansion', task_reference_name='expansion2')

task_simulation_postprocess = simple_task(task_def_name='SimulationPostProcess', task_reference_name='simulation_postprocess')

simulation_loop = DoWhileTask(task_ref_name='simulation_loop', tasks=[task_expansion2, task_simulation_postprocess], 
                             termination_condition='if ($.simulation_postprocess["finish"] == true){false;} else {true;} ')

task_back_propagation = simple_task(task_def_name='BackPropagation', task_reference_name='back_propagation')

task_check = simple_task(task_def_name="MCTSCompletionCheck", task_reference_name="mcts_completion_check")

output_interface = simple_task(task_def_name='OutputInterface', task_reference_name='output_interface')

# Configure workflow execution flow: 
workflow >> client_input_task >> task_selection >> task_expansion >> task_simulation_preprocess >> simulation_loop >> task_back_propagation >> task_check >> output_interface

# Register workflow with overwrite=True for consistency with lite version
workflow.register(overwrite=True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath('configs')
cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[SimpleInput()])
# Change start_interactor to start_interaction for lite version
cli_client.start_interaction()
