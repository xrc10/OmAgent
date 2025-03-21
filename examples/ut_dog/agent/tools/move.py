from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import field_validator
import time
import math

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
    "vx": {
        "type": "number",
        "description": "Speed along the x-axis direction, in meters per second (m/s). Positive values indicate forward movement, and negative values indicate backward movement.",
        "required": False,
    },
    "vy": {
        "type": "number",
        "description": "Speed along the y-axis direction, in meters per second (m/s). Positive values indicate movement to the left, and negative values indicate movement to the right.",
        "required": False,
    },
    "vyaw": {
        "type": "number",
        "description": "Angular velocity around the z-axis, in radians per second (rad/s). Positive values indicate counterclockwise rotation, and negative values indicate clockwise rotation.",
        "required": False,
    },
}


@registry.register_tool()
class Move(BaseTool):
    """Tool for making Unitree Go2 robot move."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = """This tool is used to control the robot dog to move. You can make the robot dog move forward, backward, left, right, and rotate by manipulating the vx, vy, and vyaw parameters."""
    network_interface_name: Optional[str] = "eth0"

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.sport_client = None

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name

    def _run(
        self,
        vx: float = 0,
        vy: float = 0,
        vyaw: float = 0
    ) -> Dict[str, Any]:
        """Control the Go2 to move."""
        if self.sport_client is None:
            ChannelFactoryManager.initialize(0, self.network_interface_name)
            self.sport_client = ChannelFactoryManager.get_sport_client()
        try:
            remaining_vx = abs(vx)
            remaining_vy = abs(vy)
            remaining_vyaw = abs(vyaw)
            
            vx_direction = 1 if vx > 0 else -1
            vy_direction = 1 if vy > 0 else -1
            vyaw_direction = 1 if vyaw > 0 else -1
            
            if remaining_vx > 3.8 or remaining_vy > 1.0 or remaining_vyaw > 4:
                while remaining_vx > 0 or remaining_vy > 0 or remaining_vyaw > 0:
                    current_vx = min(2, remaining_vx) * vx_direction if remaining_vx > 0 else 0
                    current_vy = min(1, remaining_vy) * vy_direction if remaining_vy > 0 else 0
                    current_vyaw = min(1.5, remaining_vyaw) * vyaw_direction if remaining_vyaw > 0 else 0
                    
                    code = self.sport_client.Move(current_vx, current_vy, current_vyaw)
                    if code != 0:
                        raise Exception(f"code: {code}")
                    
                    remaining_vx = max(0, remaining_vx - abs(current_vx))
                    remaining_vy = max(0, remaining_vy - abs(current_vy))
                    remaining_vyaw = max(0, remaining_vyaw - abs(current_vyaw))
                    
                    time.sleep(1)
                
                return {
                    "code": 0,
                    "msg": "success",
                }
            
            code = self.sport_client.Move(vx, vy, vyaw)
            if code != 0:
                raise Exception(f"code: {code}")
            return {
                "code": code,
                "msg": "success",
            }
        except Exception as e:
            logging.error(f"Move failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
            }
