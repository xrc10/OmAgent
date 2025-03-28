from pathlib import Path
import time
from typing import Any, Dict

import traceback

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema
from .utils.channel_manager import ChannelFactoryManager
from .get_image_sample import GetImageSample
from ..schemas.note import VisionState

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
}


@registry.register_tool()
class GetSurroundingImage(GetImageSample):
    """Tool for making Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once."""

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Get the surrounding images of the current position. This will cost time, so use it only when it is necessary to obtain the complete surrounding environment."

    def model_post_init(self, __context: Any) -> None:
        self.video_client = None
        self.sport_client = None

    def _run(self,memorize: bool = True) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to get surrounding image. Can get images in 8 directions at once.
        """
        if self.sport_client is None:
            ChannelFactoryManager.initialize(0, self.network_interface_name)
            self.sport_client = ChannelFactoryManager.get_sport_client()
        try:
            vision_states = []
            for i in range(8):
                image = self.take_shot()
                vision_states.append(VisionState(image=image, vyaw=1.5 * i))
                self.sport_client.Move(0,0,1.5)
                time.sleep(1)
            if memorize:
                self.update_memory(vision_states)
                
            return {
                "code": 0,
                "msg": "success",
                "result": "Successfully get surrounding images in 8 directions.",
                "vision_states": vision_states
            }
        except Exception as e:
            logging.error(f"Get surrounding image failed: {e}")
            logging.error(traceback.format_exc())
            return {
                "code": 500,
                "msg": "failed",
                "result": f"Failed to get surrounding images. The reason is {e}",
                "vision_states": []
            }
