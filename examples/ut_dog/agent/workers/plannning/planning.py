from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from agent.tools.move import Move
from agent.tools.get_image_sample import GetImageSample
from time import sleep

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class Planning(BaseWorker):
    def _run(self, *args, **kwargs):
        # Read user input through configured input interface
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt="Please give me a task.",
        )

        # Extract text and image content from input message
        content = user_input["messages"][-1]["content"]
        for content_item in content:
            if content_item["type"] == "text":
                user_instruction = content_item["data"]

        logging.info(f"User_instruction: {user_instruction}")

        surroundings = self.look_around()

        for surrounding in surroundings:
            import cv2
            cv2.imwrite(f"./surrounding_{surrounding['direction']}.jpg", surrounding["image"])

        return {"user_instruction": user_instruction}

    def look_around(self):
        move_tool = Move(network_interface_name="eth0")
        get_image_sample_tool = GetImageSample(network_interface_name="eth0")
        surroundings =[]
        for i in range(8):
            image = get_image_sample_tool.take_shot()
            surroundings.append({"image": image, "direction": i*0.85})
            move_tool._run(vyaw=0.85)
            print("Move to next direction")
            sleep(1)
        return surroundings
