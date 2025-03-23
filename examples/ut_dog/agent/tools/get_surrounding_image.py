from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import os

import cv2
import numpy as np
from pydantic import field_validator
import traceback
from PIL import Image

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager
CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
}


@registry.register_tool()
class GetSurroundingImage(BaseTool):
    """Tool for making Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Control the Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once."
    network_interface_name: Optional[str] = "eth0"

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.video_client = None
        self.sport_client = None

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name
    
    def take_shot(self):
        if self.video_client is None and self.sport_client is None:
            ChannelFactoryManager.initialize(0, self.network_interface_name)
            self.video_client = ChannelFactoryManager.get_video_client()
            self.sport_client = ChannelFactoryManager.get_sport_client()
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

    def _run(
        self,
    ) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once.
        """

        try:
            surroundings =[]
            
            # Create directory for saving images if it doesn't exist
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_images")
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp_raw = int(time.time())
            formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp_raw))
            
            for i in range(8):
                image = self.take_shot()
                vyaw = 1.5 * i
                
                filename = os.path.join(save_dir, f"vyaw_{formatted_time}_{vyaw:.2f}.jpg")
                image.save(filename)
                logging.info(f"Saved image to {filename}")
                
                surroundings.append({"image": image, "vyaw": vyaw})
                self.sport_client.Move(0,0,1.5)
                time.sleep(1)
                
            cache_data = self.stm(self.workflow_instance_id).get("image_cache", {})
            cache_data.update({f"<image_{timestamp_raw}>": surroundings})
            self.stm(self.workflow_instance_id)["image_cache"] = cache_data
            return {
                "code": 0,
                "msg": "success",
                "surroundings": surroundings
            }
        except Exception as e:
            logging.error(f"Get surrounding image failed: {e}")
            logging.error(traceback.format_exc())
            return {
                "code": 500,
                "msg": "failed",
                "surroundings": surroundings
            }
