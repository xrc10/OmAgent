import json_repair
import re
from pathlib import Path
from typing import List

from pydantic import Field

from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.utils.logger import logging

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class FindObjectLocation(BaseLLMBackend, BaseWorker):
    """Worker that determines and describes the location of a detected object in an image.
    
    This processor analyzes the object detection results and generates a natural language
    description of where the object is located in the image.
    
    Attributes:
        output_parser (StrParser): Parser for LLM output strings
        llm (OpenaiGPTLLM): LLM model for generating location descriptions
        prompts (List[PromptTemplate]): System and user prompts for location description
    """
    llm: OpenaiGPTLLM
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("sys_prompt.prompt"), role="system"
            ),
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("user_prompt.prompt"), role="user"
            ),
        ]
    )

    def _run(self, *args, **kwargs):
        """Process object detection results and describe object location.
        
        Args:
            *args: Variable length argument list
            **kwargs: Contains workflow context including detection results
            
        Returns:
            dict: Location description and any relevant coordinates/metadata
        """
        # Get detection results from previous task
        object_description = self.stm(self.workflow_instance_id).get("object_description", "")
        if not object_description:
            return {"location_description": "Object not found in image"}

        # Generate location description using LLM
        chat_complete_res = self.simple_infer(
            object_description=object_description,
            image='<image_0>'
        )
        content = chat_complete_res["choices"][0]["message"].get("content")
        location_info = self._extract_from_result(content)

        # Send feedback via callback and return
        self.callback.send_answer(agent_id=self.workflow_instance_id, msg=location_info.get("location_description", "Object location not found."))
        
        # Store results in workflow memory
        self.stm(self.workflow_instance_id)["location_info"] = location_info
        
        return location_info

    def _extract_from_result(self, result: str) -> dict:
        try:
            pattern = r"```json\s+(.*?)\s+```"
            match = re.search(pattern, result, re.DOTALL)
            if match:
                return json_repair.loads(match.group(1))
            else:
                return json_repair.loads(result)
        except Exception as error:
            raise ValueError("LLM generation is not valid.")
