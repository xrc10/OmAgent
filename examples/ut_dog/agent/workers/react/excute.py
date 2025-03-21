from pathlib import Path
import json
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.tool_system.manager import ToolManager
from omagent_core.models.llms.schemas import Message
from agent.tools.get_surrounding_image import GetSurroundingImage
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import BaseModel, Field
from agent.tools.move import Move
from agent.tools.get_image_sample import GetImageSample
from agent.schemas.note import Note
from time import sleep
from PIL import Image

CURRENT_PATH = Path(__file__).parents[0]

class ObservationResult(BaseModel):
    observation: str = Field(description="The observation of the task")
    reason: str = Field(description="Criteria for determining whether a task is completed")
    is_done: bool = Field(description="Whether the task is done")

@registry.register_worker()
class ReactExcute(BaseLLMBackend, BaseWorker):
    tool_manager: ToolManager
    def _run(self, *args, **kwargs):
        # Read user input through configured input interface
        note = self.stm(self.workflow_instance_id).get("note")
        current_task = note.unfinished_steps()[0]
        observation = None

        for action_limit in range(3):
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is:{current_task.instruction}")
            plan = self.reasoning(current_task.instruction, observation)
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"After reasoning, I have the following plan:\n{plan}")
            
            execution_status, execution_results = self.tool_manager.execute_task(plan)
            if execution_status != "success":
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Plan execution failed")
                raise Exception(f"Execution failed, {execution_results}")
            
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Plan execution completed")
            
            observation = self.observing(current_task.proof_of_completion)
            if observation.is_done:
                current_task.is_done = True
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is completed.\n{str(note)}")
                self.stm(self.workflow_instance_id)["note"] = note
                if len(note.unfinished_steps()) == 0:
                    return {"all_tasks_finished": True}
                else:
                    return {"all_tasks_finished": False}
            else:
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is not completed. Reason: {observation.reason}")
                
        self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"I tried 3 times, but still failed the task {current_task.instruction}")
        raise Exception("All attempts failed")
        
    
    def reasoning(self, task:str, observation:ObservationResult):
        sys_prompt = PromptTemplate.from_file(CURRENT_PATH.joinpath("reasoning_prompt.prompt"), role="system")
        sys_prompt = sys_prompt.format(tools=self.tool_manager.generate_prompt())

        get_surrounding_image = GetSurroundingImage()
        get_surrounding_image._parent = self
        result = get_surrounding_image._run()

        if result["code"] != 0:
            raise Exception("Get surrounding image failed")
        
        surroundings = result["surroundings"]

        user_prompt = [f"The current task is:{task}", "The environment around you and the corresponding vyaw value are as follows:"]
        if observation:
            user_prompt.extend([
                f"Your previous observation is:{observation.observation}",
                f"The reason for determining that the task is not completed is:{observation.reason}",
                "Now, please reason about the current task again."
            ])
        for item in surroundings:
            user_prompt.extend([
                f"vyaw: {item['vyaw']}, image:",
                item["image"],
            ])

        completion = self.llm.generate(records=[
            Message.system(sys_prompt),
            Message.user(content=user_prompt)
        ])

        return completion["choices"][0]["message"]["content"]
    
    def observing(self, proof_of_completion:str):
        sleep(1)
        with open(CURRENT_PATH.joinpath("observing_prompt.prompt"), "r") as f:
            sys_prompt = f.read()

        get_surrounding_image = GetSurroundingImage()
        get_surrounding_image._parent = self
        result = get_surrounding_image._run()
        if result["code"] != 0:
            raise Exception("Get surrounding image failed")
        
        surroundings = result["surroundings"]

        user_prompt = [f"Completing the following conditions means you have completed the task:{proof_of_completion}", "The environment around you and the corresponding vyaw value are as follows:"]    
        for item in surroundings:
            user_prompt.extend([
                f"vyaw: {item['vyaw']}, image:",
                item["image"],
            ])
            
        completion = self.llm.generate(records=[
            Message.system(sys_prompt),
            Message.user(content=user_prompt)
        ],
        response_format=ObservationResult)
        print(11111111111111111, completion["choices"][0]["message"]["content"])
        return ObservationResult.model_validate_json(completion["choices"][0]["message"]["content"])
