from pathlib import Path
from typing import List

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from omagent_core.utils.registry import registry
from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StorySummarizer(BaseLLMBackend, BaseWorker):
    """Worker for summarizing story progress to save context and tokens."""
    
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
        """Summarize the story progress to a more concise form."""
        # Get story context
        story_context = self.stm(self.workflow_instance_id)["story_context"]

        # take the latest turn's response
        latest_turn_response = story_context["story_progress"][-1]["response"]
        latest_turn_user_action = story_context["story_progress"][-1]["user_action"]
        
        # Generate summary using LLM
        chat_complete_res = self.simple_infer(
            story_type=story_context["story_type"],
            background=story_context["background"],
            user_action=latest_turn_user_action,
            response=latest_turn_response,
            current_turn=story_context["current_turn"],
            max_turns=story_context["max_turns"]
        )
        
        summary = chat_complete_res["choices"][0]["message"].get("content")
        
        # Update story progress with the summary
        story_context["story_progress"][-1]["response"] = summary
        
        # Store updated context
        self.stm(self.workflow_instance_id)["story_context"] = story_context
        
        return {"summary": summary} 