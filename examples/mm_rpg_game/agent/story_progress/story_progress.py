from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.engine.worker.llm.base import BaseLLMBackend
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StoryProgress(BaseLLMBackend, BaseWorker):
    """Worker for checking story progress and determining if it should end."""
    
    def __init__(self):
        super().__init__()
        self.name = "StoryProgress"
        
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