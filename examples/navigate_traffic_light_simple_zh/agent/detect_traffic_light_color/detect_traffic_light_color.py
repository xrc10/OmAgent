import json_repair
import re
from pathlib import Path
from typing import List
from pydantic import Field

from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.utils.logger import logging


CURRENT_PATH = root_path = Path(__file__).parents[0]


@registry.register_worker()
class DetectTrafficLightColor(BaseLLMBackend, BaseWorker):
    """Object detection processor that determines if the requested object is found in the image.
    
    This processor evaluates whether the requested object exists in the provided image
    by analyzing user instructions and image analysis results. It uses an LLM to
    make this determination.
    
    Returns success if object is found, otherwise returns failed with reason.
    """
    llm: OpenaiGPTLLM
    prompts: List[PromptTemplate] = Field(
        default=[
            # PromptTemplate.from_file(
            #     CURRENT_PATH.joinpath("sys_prompt.prompt"), role="system"
            # ),
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("user_prompt.prompt"), role="user"
            ),
        ]
    )

    def _run(self, *args, **kwargs):
        """Process the current state to determine if the requested object is found.
        
        Args:
            args: Variable length argument list
            kwargs: Arbitrary keyword arguments
            
        Returns:
            dict: Contains:
                - 'object_found': True if object is found, False otherwise
                - 'stop_search': True if user wants to stop searching
                - 'feedback': Additional information about the search result
        """
        logging.info(self.prompts)
        # Query LLM to analyze available information
        chat_complete_res = self.simple_infer(
            image='<image_0>',
        )
        logging.info(chat_complete_res)
        content = chat_complete_res["choices"][0]["message"].get("content")
        logging.info(content)
        content = self._extract_from_result(content)

        if isinstance(content, str):
            content = {"stop_search": False}

        # Return the output json value of light_color

        # send results to user
        self.callback.send_answer(
            agent_id=self.workflow_instance_id,
            msg=f"{content.get('light_color')}"
        )

        content['stop_search'] = False
        return content

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