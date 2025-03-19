from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import field_validator

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
}


@registry.register_tool()
class StandUp(BaseTool):
    """Tool for making Unitree Go2 robot stand up. The robot stands up normally, with the joint motor locked. Compared to the balanced standing mode, this mode does not maintain a constant balance posture. The default standing height is 0.33m."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Control the Unitree Go2 robot to stand up. The robot stands up normally, with the joint motor locked. Compared to the balanced standing mode, this mode does not maintain a constant balance posture. The default standing height is 0.33m."
    network_interface_name: Optional[str]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        ChannelFactoryManager.initialize(0, self.network_interface_name)
        self.sport_client = ChannelFactoryManager.get_sport_client()

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name

    def _run(
        self
    ) -> Dict[str, Any]:
        """
        Control the Go2 to stand up.
        """

        try:
            code = self.sport_client.StandUp()
            if code != 0:
                raise Exception(f"code: {code}")
            return {
                "code": code,
                "msg": "success",
            }
        except Exception as e:
            logging.error(f"Stand up failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
            }
