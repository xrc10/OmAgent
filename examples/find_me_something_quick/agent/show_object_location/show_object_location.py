import json_repair
import re
from pathlib import Path
from typing import List

from pydantic import Field

from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.utils.logger import logging

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class ShowObjectLocation(BaseWorker):
    """Worker that shows the object location in STM
    
    Attributes:
        output_parser (StrParser): Parser for LLM output strings
    """

    def _run(self, *args, **kwargs):
        """Process object detection results and describe object location.
        
        Args:
            *args: Variable length argument list
            **kwargs: Contains workflow context including detection results
            
        Returns:
            None
        """
        object_location = self.stm(self.workflow_instance_id).get("object_location", "")

        # Send feedback via callback and return
        self.callback.send_answer(agent_id=self.workflow_instance_id, msg=object_location)
        
        return object_location
