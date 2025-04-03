from pathlib import Path
from typing import Any, Dict

from omagent_core.utils.registry import registry
from omagent_core.tool_system.base import ArgSchema
from .move import Move

CURRENT_PATH = Path(__file__).parents[0]

ARGSCHEMA = {}


@registry.register_tool()
class TurnDegree(Move):
    """Tool for making Unitree Go2 robot turn a degree."""
    # vyaw=1.5 is about turning left 30 degrees
    # the inputs is a float number, which is the degree to turn
    # negative degree is turning left, positive degree is turning right

    args_schema: ArgSchema = ArgSchema(**ARGSCHEMA)
    description: str = """This tool is used to control the robot dog to turn a degree. """

    def _run(self, degree: float) -> Dict[str, Any]:
        """Control the Go2 to move."""
        vyaw = - degree / 30 * 1.5
        return super()._run(vx=0, vy=0, vyaw=vyaw)
