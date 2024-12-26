from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class RPGImageInput(BaseWorker):
    """Worker for handling image input in the RPG game."""
        
    def _run(self, *args, **kwargs):
        """Process the image input and store it for story generation."""
        # Get image input from user
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt="请上传一张图片作为故事的起点\n",
        )
        
        # Extract image content
        content = user_input["messages"][-1]["content"]
        image_path = None
        
        # Find image in content
        for content_item in content:
            if content_item["type"] in ["image", "image_url"]:
                image_path = content_item["data"]
                break
        
        if not image_path:
            raise ValueError("未能找到上传的图片")
            
        # Read and store image
        image = read_image(input_source=image_path)
        # Store image in workflow shared memory with standard key
        image_cache = {"<image_0>": image}
        self.stm(self.workflow_instance_id)["image_cache"] = image_cache
        
        return {"image_path": image_path} 
