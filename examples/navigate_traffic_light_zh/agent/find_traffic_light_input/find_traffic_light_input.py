from pathlib import Path

from omagent_core.utils.registry import registry
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image

CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class FindTrafficLightInput(BaseWorker):
    """Object search input processor that handles user-provided images.
    
    This processor allows users to provide an image in which the blind people are seeking a traffic light
    
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
        # ask the user to take a new photo of the traffic light
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id, 
            input_prompt=f'请拍一张交通信号灯的照片（你可以说任何话）'
        )
        
        content = user_input['messages'][-1]['content']
        for content_item in content:
            if content_item['type'] == 'image_url':
                image_path = content_item['data']
        try:
            img = read_image(input_source=image_path)
            image_cache = {'<image_0>' : img}
            self.stm(self.workflow_instance_id)['image_cache'] = image_cache
        except Exception as e:
            pass
        
        return 

