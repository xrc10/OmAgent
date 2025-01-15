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

        # Prepare messages for LLM
        messages = [
            Message(
                role="user", 
                message_type="text", 
                content="你正在帮助一位盲人导航。"
                       "请用1-2个简短的中文句子描述障碍物。包括：\n"
                       "- 物体是什么以及位置（左边/中间/右边）\n"
                       "- 距离\n"
                       "- 简短的安全建议"
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