from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import field_validator

from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema, BaseTool
from omagent_core.models.llms.base import BaseLLM, PromptTemplate
from ..utils.channel_manager import ChannelFactoryManager
CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {
    "switch": {
        "type": "bool",
        "description": "switch to enable or disable free avoid.",
        "required": True,
    },
}


@registry.register_tool()
class Communicate(BaseTool):
    """Tool for making Unitree Go2 robot to communicate with human."""
    llm: BaseLLM

    class Config:
        """Configuration for this pydantic object."""

        extra = "allow"
        arbitrary_types_allowed = True

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = "This tool is used to communicate with human. When you need help of person like interacting with the environment or getting some informations you are lacking, you can use this tool to ask and get answer."

    def _run(self, goal: str, max_chat_turns: int = 10) -> str:
        current_step = self.stm(self.workflow_instance_id)["note"].current_step()

        
        

        return "Success"