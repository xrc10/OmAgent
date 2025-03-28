from pathlib import Path
from typing import Any, Dict

from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema
from .move import Move

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {}


@registry.register_tool()
class TurnRight(Move):
    """Tool for making Unitree Go2 robot turn right."""

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = """This tool is used to control the robot dog to turn right. """

    def _run(self) -> Dict[str, Any]:
        """Control the Go2 to move."""
        return super()._run(vx=0, vy=0, vyaw=-1.5)
