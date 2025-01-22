from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.general import read_image, encode_image
from omagent_core.utils.registry import registry
from PIL import Image, ImageDraw

PROMPT = """
仅描述红框内主要物体，限6字内
"""

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
                content=PROMPT
            )
        ]

        # Add image
        img = image_cache["<image_0>"]
        image_obj = read_image(img)
        # draw a bounding box at the center of the image, middle 30% to 70% of the image height and width
        draw = ImageDraw.Draw(image_obj)
        draw.rectangle((image_obj.width * 0.3, image_obj.height * 0.3, image_obj.width * 0.7, image_obj.height * 0.7), outline="red", width=5)

        messages.append(
            Message(
                role="user",
                message_type="image",
                content=[
                    Content(
                        type="image_url",
                        image_url={
                            "url": f"data:image/jpeg;base64,{encode_image(image_obj)}" # pass the image with bounding box
                        },
                    )
                ],
            )
        )

        # Get LLM response
        self.llm.use_default_sys_prompt = False
        response = self.llm.generate(records=messages)
        description = response["choices"][0]["message"]["content"]

        # Always send message with description
        output_msg = f"{min_depth:.1f}米 {description}" if min_depth else description
        self.callback.send_answer(
            self.workflow_instance_id,
            msg=output_msg
        )

        return {"description": description} 