from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import field_validator
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager
CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
    "switch": {
        "type": "bool",
        "description": "switch to enable or disable free avoid.",
        "required": True,
    },
}


@registry.register_tool()
class FreeAvoid(BaseTool):
    """Tool for making Unitree Go2 robot to enable or disable free avoid."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Control the Unitree Go2 robot to enable or disable free avoid."
    network_interface_name: Optional[str]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        ChannelFactoryManager.initialize(0, self.network_interface_name)
        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name

    def _run(
        self,
        switch: bool = True
    ) -> Dict[str, Any]:
        """
        Control the Unitree Go2 robot to free avoid.
        """

        try:
            self.sport_client.FreeAvoid(switch)
            return {
                "code": 0,
                "msg": "success",
            }
        except Exception as e:
            logging.error(f"Free avoid failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
            }
