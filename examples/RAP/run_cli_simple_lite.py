# Import required modules and components
import os
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.lite_version.cli import DefaultClient
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.utils.logger import logging
from omagent_core.engine.worker.base import BaseWorker

# Register a simple input worker similar to general_dnc example
@registry.register_worker()
class SimpleRAPInput(BaseWorker):
    def _run(self, *args, **kwargs):
        self.stm(self.workflow_instance_id)['task'] = 'math'
        self.stm(self.workflow_instance_id)['data_input'] = "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
        return {'data_input': "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"}

# Initialize logging
logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

container.register_stm("RedisSTM")

# Initialize workflow with lite_version=True
workflow = ConductorWorkflow(name='RAP', lite_version=True)

# Configure workflow tasks:
client_input_task = simple_task(task_def_name=SimpleRAPInput, task_reference_name='input_task')

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

# Register workflow with overwrite=True
workflow.register(overwrite=True)

# Initialize and start CLI client with workflow configuration
config_path = CURRENT_PATH.joinpath('configs')
cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[SimpleRAPInput()])
cli_client.start_processor_with_input({"data_input": "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"})

# Uncomment to test with specific input
#cli_client.start_processor_with_input({"query": "your test query here"})
