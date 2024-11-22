from pathlib import Path

from omagent_core.utils.registry import registry
from omagent_core.utils.general import read_image
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class InputInterface(BaseWorker):
    """Input interface processor that handles user instructions and image input.
    
    This processor:
    1. Reads user input containing question and image via input interface
    2. Extracts text instruction and image path from the input
    3. Loads and caches the image in workflow storage
    4. Returns the user instruction for next steps
    """

    def _run(self, *args, **kwargs):
        # Get or initialize the call counter from STM
        stm = self.stm(self.workflow_instance_id)
        call_count = stm.get('input_interface_calls', 0)
        
        # Only show greeting on first call
        input_prompt = '你好，有什么可以帮你的吗？' if call_count == 0 else None
        
        # Increment and store call counter
        stm['input_interface_calls'] = call_count + 1
        
        # Read user input through configured input interface
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id, 
            input_prompt=input_prompt
        )
        
        image_path = None
        # Extract text and image content from input message
        # try:
        #     content = user_input['messages'][-1]['content']
        # except:
        #     return
        content = user_input['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'text':
                user_instruction = content_item['data']
            elif content_item['type'] == 'image_url':
                image_path = content_item['data']
        
        logging.info(f'User_instruction: {user_instruction}\nImage_path: {image_path}')
        
        # Load image from file system
        if image_path:
            img = read_image(input_source=image_path)
            
            # Store image in workflow shared memory with standard key
            image_cache = {'<image_0>' : img}
            self.stm(self.workflow_instance_id)['image_cache'] = image_cache

        return {'user_instruction': user_instruction}