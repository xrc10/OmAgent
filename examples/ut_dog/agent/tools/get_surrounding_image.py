from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from pydantic import field_validator
from unitree_sdk2py.go2.video.video_client import VideoClient

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
    network_interface_name: Optional[str]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        ChannelFactoryManager.initialize(0, self.network_interface_name)
        self.video_client = VideoClient()  
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()
        self.sport_client = ChannelFactoryManager.get_sport_client()

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name
    
    def take_shot(self):
        code, data = self.video_client.GetImageSample()
        if code != 0:
            raise Exception(f"Get image sample failed: {code}")

        image_array = np.frombuffer(bytes(data), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)        
        return image

    def _run(
        self,
    ) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once.
        """

        try:
            surroundings =[]
            for i in range(8):
                image = self.take_shot()
                vyaw = 1.5
                surroundings.append({"image": image, "direction": i*vyaw})
                self.sport_client.Move(0,0,vyaw)
                time.sleep(1)
            cache_data = self.stm(self.workflow_instance_id)["image_cache"]
            if cache_data is None:
                cache_data = {}
            cache_data.update({f"<image_{int(time.time())}>": surroundings})
            self.stm(self.workflow_instance_id)["image_cache"] = cache_data
            return {
                "code": 0,
                "msg": "success"
            }
        except Exception as e:
            logging.error(f"Get surrounding image failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
            }
