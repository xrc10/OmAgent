from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.general import read_image, encode_image
from omagent_core.utils.registry import registry

@registry.register_worker()
class ObstacleDetector(BaseWorker, BaseLLMBackend):
    """Worker that uses multimodal LLM to detect and describe obstacles"""

    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        # Get image from cache
        image_cache = self.stm(self.workflow_instance_id).get("image_cache", None)
        if not image_cache:
            return {"error": "No image found"}

        # Get depth results
        depth_result = self.stm(self.workflow_instance_id).get("depth_result", None)
        if not depth_result:
            return {"error": "No depth results found"}

        min_depth = depth_result.get('min_depth', None)
        depth_info = f"画面中心最近的物体距离约为 {min_depth:.1f} 米。" if min_depth else ""

        # Prepare messages for LLM
        messages = [
            Message(
                role="user", 
                message_type="text", 
                content="你正在帮助盲人导航。请只描述最近的或最可能阻挡行走的1-2个障碍物，格式：物体+位置+距离。\n"
                       "- 只关注影响行走的障碍物（如桌子、椅子、墙、柱子等），忽略墙上物品、装饰等\n"
                       "- 位置用：偏左/中间/偏右\n"
                       "- 如果有多个物体，优先描述距离最近或最可能阻挡行走的物体\n"
                       f"{depth_info}"
            )
        ]

        # Add image
        img = image_cache["<image_0>"]
        messages.append(
            Message(
                role="user",
                message_type="image",
                content=[
                    Content(
                        type="image_url",
                        image_url={
                            "url": f"data:image/jpeg;base64,{encode_image(read_image(img))}"
                        },
                    )
                ],
            )
        )

        # Get LLM response
        response = self.llm.generate(records=messages)
        description = response["choices"][0]["message"]["content"]

        # Send brief description to user
        self.callback.send_answer(
            self.workflow_instance_id,
            msg=f"{description}"
        )

        return {"description": description} 