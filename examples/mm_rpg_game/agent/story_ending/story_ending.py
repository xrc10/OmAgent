from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.engine.worker.llm.base import BaseLLMBackend
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StoryEnding(BaseLLMBackend, BaseWorker):
    """Worker for generating the story ending."""
    
    def __init__(self):
        super().__init__()
        self.name = "StoryEnding"
        
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