from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import traceback

import cv2
import numpy as np
from pydantic import field_validator
from PIL import Image

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager
from ..schemas.note import Note, VisionState, Step

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
}


@registry.register_tool()
class GetImageSample(BaseTool):
    """Tool for making Unitree Go2 robot to get image sample."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Get the image sample of the current view of the robot dog."
    network_interface_name: Optional[str] = "eth0"

    def model_post_init(self, __context: Any) -> None:
        self.video_client = None

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name
    
    def take_shot(self):
        if self.video_client is None:
            ChannelFactoryManager.initialize(0, self.network_interface_name)
            self.video_client = ChannelFactoryManager.get_video_client()
        code, data = self.video_client.GetImageSample()
        if code != 0:
            raise Exception(f"Get image sample failed: {code}")

        image_array = np.frombuffer(bytes(data), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)        
        pil_image = Image.fromarray(image)
        
        # Resize image to have longest edge as 512 pixels while maintaining aspect ratio
        width, height = pil_image.size
        max_dim = max(width, height)
        if max_dim > 512:
            scale_factor = 512 / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
        return pil_image
    
    def update_memory(self, vision_states: List[VisionState]):
        cache_data: Note = self.stm(self.workflow_instance_id)["note"]
        cache_data.current_step().vision = vision_states
        self.stm(self.workflow_instance_id)["robot_memory"] = cache_data

    def _run(self, memorize: bool = True) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to get image sample.
        """

        try:
            image = self.take_shot()
            vision_states = [VisionState(image=image, vyaw=0)]
            if memorize:
                self.update_memory(vision_states)

            return {
                "code": 0,
                "msg": "success",
                "result": "Successfully get front image.",
                "vision_states": vision_states
            }
        except Exception as e:
            logging.error(f"Get front image failed: {e}")
            logging.error(traceback.format_exc())
            return {
                "code": 500,
                "msg": "failed",
                "result": f"Failed to get front image. The reason is {e}",
                "vision_states": []
            }
