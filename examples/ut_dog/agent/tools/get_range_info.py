import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import field_validator
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PointStamped_


from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from .utils.channel_manager import ChannelFactoryManager

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
}


@registry.register_tool()
class GetRangeInfo(BaseTool):
    """Tool for making Unitree Go2 robot get range info. Includes distance information in three directions: front, left, and right."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "Control the Go2 to get range info. Includes distance information in three directions: front, left, and right."
    network_interface_name: Optional[str]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        ChannelFactoryManager.initialize(0, self.network_interface_name)

    @field_validator("network_interface_name")
    @classmethod
    def network_interface_name_validator(cls, network_interface_name: Union[str, None]) -> Union[str, None]:
        if network_interface_name == None:
            raise ValueError("network interface name is not provided.")
        return network_interface_name
    
    def get_single_range_info(self):
        # Create a subscriber
        subscriber = ChannelSubscriber("rt/utlidar/range_info", PointStamped_)
        
        # For storing received data
        received_data = None
        
        def single_handler(message):
            nonlocal received_data
            received_data = {
                'timestamp': f"{message.header.stamp.sec}.{message.header.stamp.nanosec}",
                'frame_id': message.header.frame_id,
                'front_distance': message.point.x,
                'left_distance': message.point.y,
                'right_distance': message.point.z
            }
        
        subscriber.Init(single_handler)
        
        # Wait for data reception (up to 3 seconds)
        timeout = 3
        start_time = time.time()
        while received_data is None and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        return received_data

    def _run(
        self,
    ) -> Dict[str, Any]:
        """
        Control the Go2 to get range info.
        """

        try:
            range_info = self.get_single_range_info()
            return {
                "code": 0,
                "msg": "success",
                "data": range_info,
            }
        except Exception as e:
            logging.error(f"Get range info failed: {e}")
            return {
                "code": 500,
                "msg": "failed",
                "data": None,
            }
