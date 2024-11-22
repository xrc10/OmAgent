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
class WalkToTrafficLight(BaseLLMBackend, BaseWorker):
    """Object detection processor that determines if the user is walking towards the traffic light.
    
    This processor evaluates whether the user is walking towards the traffic light
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
        """Process the current state to determine if the user is walking towards the traffic light.
        
        Args:
            args: Variable length argument list
            kwargs: Arbitrary keyword arguments
            
        Returns:
            dict: Contains:
                - 'decision': "straight", "left", or "right"
                - 'degree_of_turn': the degree of turn to the traffic light
                - 'distance_to_traffic_light': the distance to the traffic light
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
            # message to user
            self.callback.send_answer(
                agent_id=self.workflow_instance_id,
                msg=f"Continue to walk straight."
            )
            content = {"decision": "not_found", "reason": "Return is not valid JSON."}
            return {
                "decision": "not_found",
                "degree_of_turn": 0,
                "distance_to_traffic_light": 0,
                "stop_search": False
            }

        # Return decision and handle feedback if more information is needed
        if content.get("decision") == "straight":
            # message to user
            self.callback.send_answer(
                agent_id=self.workflow_instance_id,
                msg=f"Continue to walk straight. You are {content.get('distance_to_traffic_light')} meters away from the traffic light."
            )
            return {
                "decision": "straight",
                "degree_of_turn": 0,
                "distance_to_traffic_light": 0,
                "stop_search": False
            }
        elif content.get("decision") == "left" or content.get("decision") == "right":
            # message to user
            self.callback.send_answer(
                agent_id=self.workflow_instance_id,
                msg=f"Turn {content.get('degree_of_turn')} degrees to the {content.get('decision')}. You are {content.get('distance_to_traffic_light')} away from the traffic light."
            )

            return {
                "decision": content.get('decision'),
                "degree_of_turn": content.get('degree_of_turn'),
                "distance_to_traffic_light": content.get('distance_to_traffic_light'),
                "stop_search": False
            }
        else: # no traffic light
            # message to user
            self.callback.send_answer(
                agent_id=self.workflow_instance_id,
                msg=f"No traffic light found."
            )
            return {
                "decision": "not_found",
                "degree_of_turn": 0,
                "distance_to_traffic_light": 0,
                "stop_search": True
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