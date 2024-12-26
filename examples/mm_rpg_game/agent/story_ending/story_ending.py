from pathlib import Path
from typing import List

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.utils.registry import registry
from pydantic import Field
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.prompt.prompt import PromptTemplate

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StoryEnding(BaseLLMBackend, BaseWorker):
    """Worker for generating the story ending."""
    
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
        """Generate the story ending based on the story progress."""
        # Get story context
        story_context = self.stm(self.workflow_instance_id)["story_context"]
        
        # Generate ending using LLM
        chat_complete_res = self.simple_infer(
            story_type=story_context["story_type"],
            background=story_context["background"],
            story_progress=story_context["story_progress"],
            current_turn=story_context["current_turn"],
            max_turns=story_context["max_turns"]
        )
        
        ending = chat_complete_res["choices"][0]["message"].get("content")
        
        # Store ending in context
        story_context["ending"] = ending
        self.stm(self.workflow_instance_id)["story_context"] = story_context
        
        # Send ending to user
        self.callback.send_block(
            self.workflow_instance_id,
            msg=f"\n故事结局：\n\n{ending}\n\n游戏结束，感谢你的参与！"
        )
        
        return {"ending": ending} 