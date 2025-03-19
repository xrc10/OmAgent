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
class StandDown(BaseTool):
    """Tool for making Unitree Go2 robot stand down. The robot lies down, and the motor joints remain locked."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Control the Unitree Go2 robot to stand down. The robot lies down, and the motor joints remain locked."
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
        Control the Go2 to stand down.
        """

        try:
            code = self.sport_client.StandDown()
            if code != 0:
                raise Exception(f"code: {code}")
            return {
                "code": code,
                "msg": "success",
            }
        except Exception as e:
            logging.error(f"Stand down failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
            }

if __name__ == "__main__":
    tool = StandDown(network_interface_name="eth0")
    tool.run()