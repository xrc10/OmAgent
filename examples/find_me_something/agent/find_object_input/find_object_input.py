from pathlib import Path

from omagent_core.utils.registry import registry
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image

CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class FindObjectInput(BaseWorker):
    """Object search input processor that handles user-provided images.
    
    This processor allows users to provide an image in which they want to search for
    specific objects. It accepts either an image URL or local file path as input, reads 
    the image, and caches it in the workflow's short-term memory for use by downstream 
    processors.
    
    The processor gracefully handles cases where users choose not to provide an image or 
    if there are issues reading the provided image.
    
    Attributes:
        None - This worker uses only the base worker functionality
    """

    def _run(self, *args, **kwargs):
        """Process user-provided image input for object detection.
        
        Prompts the user to provide an image to search in, either via URL or local path.
        Reads and caches the image if provided, handling any errors during image loading.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
            
        Returns:
            None - Results are stored in workflow's short-term memory
        """
        # check stm to see if search_info (list) is empty, if not, use the existing user_instruction as user_instruct
        if len(self.stm(self.workflow_instance_id).get("search_info", [])) > 0:
            user_instruct = self.stm(self.workflow_instance_id).get("user_instruction", "")
            user_input = self.input.read_input(
                workflow_instance_id=self.workflow_instance_id, 
                input_prompt=f'Please take a new photo for "{user_instruct}" (you can say anything)'
            )
            if_update_user_instruct = 0
        else:
            user_input = self.input.read_input(
                workflow_instance_id=self.workflow_instance_id, 
                input_prompt='Please tell me what and where to search.'
            )
            if_update_user_instruct = 1
        
        content = user_input['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'image_url':
                image_path = content_item['data']
            elif content_item['type'] == 'text':
                user_instruction = content_item['data']
                if if_update_user_instruct == 1:
                    self.stm(self.workflow_instance_id)['user_instruction'] = user_instruction
        try:
            img = read_image(input_source=image_path)
            image_cache = {'<image_0>' : img}
            self.stm(self.workflow_instance_id)['image_cache'] = image_cache
        except Exception as e:
            pass
        
        return 

