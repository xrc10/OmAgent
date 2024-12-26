from pathlib import Path
from typing import List

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from omagent_core.utils.general import read_image
from omagent_core.utils.registry import registry

from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class RPGDialogue(BaseLLMBackend, BaseWorker):
    """Worker for handling dialogue interactions in the RPG game."""
    
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
        """Process user input and generate story response."""
        # Get story context
        story_context = self.stm(self.workflow_instance_id)["story_context"]
        
        # Get user input
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt="你的选择是：\n"
        )
        
        # Extract text content
        content = user_input["messages"][-1]["content"]
        user_action = ""
        for content_item in content:
            if content_item["type"] == "text":
                user_action = content_item["data"]
                break

        # Find image in content
        for content_item in content:
            if content_item["type"] in ["image", "image_url"]:
                image_path = content_item["data"]
                break
        # Read and store image
        image = read_image(input_source=image_path)
        
        # Update turn counter
        story_context["current_turn"] += 1
        
        # Generate response using LLM
        chat_complete_res = self.simple_infer(
            story_type=story_context["story_type"],
            background=story_context["background"],
            story_progress=story_context["story_progress"],
            current_turn=story_context["current_turn"],
            max_turns=story_context["max_turns"],
            user_action=user_action,
            image=image
        )
        
        response = chat_complete_res["choices"][0]["message"].get("content")
        
        # Update story progress
        story_context["story_progress"].append({
            "turn": story_context["current_turn"],
            "user_action": user_action,
            "response": response
        })
        
        # Store updated context
        self.stm(self.workflow_instance_id)["story_context"] = story_context
        
        # Send response to user
        if story_context["current_turn"] >= story_context["max_turns"]:
            self.callback.send_block(
                self.workflow_instance_id,
                msg=f"{response}"
            )
        else:
            self.callback.send_block(
                self.workflow_instance_id,
                msg=f"{response}\n\n请告诉我你的下一步行动："
            )
        
        return {
            "current_turn": story_context["current_turn"],
            "user_action": user_action,
            "response": response
        } 