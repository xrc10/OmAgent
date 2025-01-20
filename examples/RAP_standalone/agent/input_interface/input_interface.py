from pathlib import Path

from omagent_core.utils.registry import registry
from omagent_core.utils.general import read_image
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging

CURRENT_PATH = Path(__file__).parents[0]

SUPPORT_TASK= {
    'math': "please input your math problem."
}

@registry.register_worker()
class InputInterface(BaseWorker):

    def _run(self, *args, **kwargs):
        # Read user input through configured input interface
        user_input = self.input.read_input(workflow_instance_id=self.workflow_instance_id, input_prompt=f'Welcome to use OmAgent RAP Algorithm, please input the task you want to conduct. Choices: {list(SUPPORT_TASK.keys())} ')
        
        task =  user_input['messages'][-1]['content'][0]['data']
        if task not in SUPPORT_TASK:
            raise NotImplementedError
        
        self.stm(self.workflow_instance_id)['task'] = task 

        user_input_2 = self.input.read_input(workflow_instance_id=self.workflow_instance_id, input_prompt=SUPPORT_TASK[task])

        content = user_input_2['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'text':
                data_input = content_item['data']
        # For debug
        if data_input == 'x':
            data_input = "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
        
        logging.info(f'data_input: {data_input}\n')
        self.stm(self.workflow_instance_id)['data_input'] = data_input
        

        return {'data_input': data_input}
