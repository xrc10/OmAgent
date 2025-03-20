from pathlib import Path
import json
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.schemas import Message
from agent.tools.move import Move
from agent.tools.get_image_sample import GetImageSample
from agent.schemas.note import Note
from time import sleep
from PIL import Image

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class Planning(BaseLLMBackend, BaseWorker):
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

        with open(CURRENT_PATH.joinpath("sys_prompt.prompt"), "r") as f:
            system_prompt = f.read()

        user_prompt = [f"The user's instruction is:{user_instruction}", "The environment around you and the corresponding vyaw value are as follows:"]
        for item in surroundings:
            user_prompt.extend([
                f"vyaw: {item['vyaw']}, image:",
                item["image"],
            ])

        result = self.llm.generate(records=[
            Message.system(system_prompt),
            Message.user(user_prompt)],
            response_format=Note
            )
        self.stm(self.workflow_instance_id)["note"] = Note(**json.loads(result["choices"][0]["message"]["content"]))

        self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"In order to complete the task you have assigned to me, I have formulated the following plan.\n{str(self.stm(self.workflow_instance_id).get('note'))}")

        return {"user_instruction": user_instruction}

    def look_around(self):
        move_tool = Move(network_interface_name="eth0")
        get_image_sample_tool = GetImageSample(network_interface_name="eth0")
        surroundings =[]
        for i in range(8):
            image = get_image_sample_tool.take_shot()
            surroundings.append({"image": Image.fromarray(image), "vyaw": i*1.5})
            move_tool._run(vyaw=1.5)
            print("Move to next direction")
            sleep(1)
        return surroundings
