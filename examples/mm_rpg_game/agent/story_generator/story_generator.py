import random
from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.engine.worker.llm.base import BaseLLMBackend
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class StoryGenerator(BaseLLMBackend, BaseWorker):
    """Worker for generating the initial story and goals based on the input image."""
    
    def __init__(self):
        super().__init__()
        self.name = "StoryGenerator"
        
    def _run(self, *args, **kwargs):
        """Generate story background and goals based on the image."""
        # Get stored image
        image = self.stm(self.workflow_instance_id)["image"]
        
        # Define possible story types
        story_types = [
            "间谍任务", "外星人探索", "灵异事件", "魔法冒险", "侦探推理",
            "时空穿越", "机器人革命", "海盗冒险", "武侠江湖", "末日生存"
        ]
        
        # Randomly select story type
        story_type = random.choice(story_types)
        
        # Generate story using LLM
        chat_complete_res = self.simple_infer(
            image=image,
            story_type=story_type
        )
        
        content = chat_complete_res["choices"][0]["message"].get("content")
        
        # Store story context
        story_context = {
            "story_type": story_type,
            "background": content,
            "current_turn": 0,
            "max_turns": 5,
            "story_progress": []
        }
        
        self.stm(self.workflow_instance_id)["story_context"] = story_context
        
        # Send story background to user
        self.callback.send_block(
            self.workflow_instance_id,
            msg=f"故事背景：\n\n{content}\n\n你现在可以开始你的冒险了！请告诉我你想做什么？"
        )
        
        return story_context 