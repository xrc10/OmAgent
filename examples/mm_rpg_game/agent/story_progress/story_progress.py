from pathlib import Path
from typing import List

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.utils.registry import registry
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StoryProgress(BaseLLMBackend, BaseWorker):
    """Worker for checking story progress and determining if it should end."""
    
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
        """Check if the story should end based on progress and turns."""
        # Get story context
        story_context = self.stm(self.workflow_instance_id)["story_context"]
        
        # Check if max turns reached
        if story_context["current_turn"] >= story_context["max_turns"]:
            return {"should_end": True, "reason": "max_turns"}
            
        # Generate progress assessment using LLM
        chat_complete_res = self.simple_infer(
            story_type=story_context["story_type"],
            background=story_context["background"],
            story_progress=story_context["story_progress"],
            current_turn=story_context["current_turn"],
            max_turns=story_context["max_turns"]
        )
        
        content = chat_complete_res["choices"][0]["message"].get("content")
        
        # Parse LLM response to determine if story should end
        should_end = "true" in content.lower()
        
        return {
            "should_end": should_end,
            "reason": "natural_conclusion" if should_end else "continue"
        } 