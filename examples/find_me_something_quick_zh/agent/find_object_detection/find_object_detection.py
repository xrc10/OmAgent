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
class FindObjectDetection(BaseLLMBackend, BaseWorker):
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
        # Retrieve context data from short-term memory, using empty lists as defaults
        if self.stm(self.workflow_instance_id).get("user_instruction"):   
            user_instruct = self.stm(self.workflow_instance_id).get("user_instruction")
        else:
            user_instruct = []
        
        if self.stm(self.workflow_instance_id).get("search_info"):
            search_info = self.stm(self.workflow_instance_id).get("search_info")
        else:
            search_info = []

        logging.info(self.prompts)
        logging.info(user_instruct)
        # Query LLM to analyze available information
        chat_complete_res = self.simple_infer(
            instruction=str(user_instruct),
            # previous_search=str(search_info),
            image='<image_0>',
            temperature=1.0,
            top_p=0.95,
        )
        logging.info(chat_complete_res)
        content = chat_complete_res["choices"][0]["message"].get("content")
        logging.info(content)
        content = self._extract_from_result(content)

        if isinstance(content, str):
            content = {"decision": "not_found", "stop_search": False, "reason": "返回的不是有效的JSON。"}

        # Return decision and handle feedback if more information is needed
        if content.get("decision") == "found":
            object_description = content.get("object description")
            self.stm(self.workflow_instance_id)['object_description'] = object_description

            object_location = content.get("object_location")
            self.stm(self.workflow_instance_id)['object_location'] = object_location

            return {
                "object_found": True,
                "stop_search": True,
                "feedback": content.get("reason", "Object found!"),
            }
        else:
            # update search info
            search_info.append(content.get("reason", "Object found!"))
            self.stm(self.workflow_instance_id)['search_info'] = search_info

            # Send feedback via callback and return
            self.callback.send_answer(
                agent_id=self.workflow_instance_id,
                msg=content.get("reason", "物体未找到，继续搜索...")
            )

            return {
                "object_found": False,
                "stop_search": False,
                "feedback": content.get("reason", "物体未找到，继续搜索...")
            }

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