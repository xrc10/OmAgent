from pathlib import Path
import json
from pydantic import BaseModel, Field
from typing import List
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.schemas import Message
from agent.tools.move import Move
from agent.tools.get_image_sample import GetImageSample
from agent.schemas.note import Note, Step
from time import sleep
from agent.tools.get_surrounding_image import GetSurroundingImage

CURRENT_PATH = Path(__file__).parents[0]

class Task4gen(BaseModel):
    instruction: str = Field(description="The instruction of the step, describe what should be done. ")
    proof_of_completion: str = Field(description="The evidences to proof the step has been completed. It should be as comprehensive and necessary as possible and should be able to verify based on the current view of the robot. ")

class Note4gen(BaseModel):
    content: List[Task4gen]

    def to_note_memory(self):
        return Note(**self.model_dump())

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

        get_surrounding_image = GetSurroundingImage()
        get_surrounding_image._parent = self
        res = get_surrounding_image._run(memorize=False)
        if res['code'] != 0:
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Failed to get surrounding images.")
            raise Exception(res['result'])
        vision_states = res['vision_states']

        with open(CURRENT_PATH.joinpath("sys_prompt.prompt"), "r") as f:
            system_prompt = f.read()

        user_prompt = [f"The user's instruction is:{user_instruction}", "The environment around you and the corresponding vyaw value are as follows:"]
        for item in vision_states:
            user_prompt.extend([
                f"vyaw: {item.vyaw}, image:",
                item.image,
            ])

        result = self.llm.generate(records=[
            Message.system(system_prompt),
            Message.user(user_prompt)],
            response_format=Note4gen
            )
        
        note = Note4gen(**json.loads(result["choices"][0]["message"]["content"])).to_note_memory()
        note.origin_vision = vision_states
        note.current_task().steps.append(Step(vision=vision_states))
        note.save()
        self.stm(self.workflow_instance_id)["note"] = note


        self.callback.info(
            agent_id=self.workflow_instance_id,
            progress=f"Planning",
            message=f'Planning completed.\n{str(self.stm(self.workflow_instance_id).get("note"))}',
        )

        return {"user_instruction": user_instruction}
