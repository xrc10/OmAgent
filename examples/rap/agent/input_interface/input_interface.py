from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class InputInterface(BaseWorker):
    """Input interface processor that handles user queries."""

    def _run(self, query=None, *args, **kwargs):
        # Get task selection from user
        task = input(f'\nWelcome to use OmAgent RAP Algorithm, please input the task you want to conduct. Choices: {list(SUPPORT_TASK.keys())} ')
        
        if task not in SUPPORT_TASK:
            raise NotImplementedError
        
        self.stm(self.workflow_instance_id)['task'] = task 

        # Use query parameter if provided, otherwise get user input
        data_input = query if query else input(SUPPORT_TASK[task])
        
        # For debug
        if data_input == 'x':
            data_input = "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
        
        self.stm(self.workflow_instance_id)['data_input'] = data_input
        return {"query": data_input} 