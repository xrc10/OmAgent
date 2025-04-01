from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.tool_system.manager import ToolManager
from omagent_core.models.llms.schemas import Message
from agent.tools.get_surrounding_image import GetSurroundingImage
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import BaseModel, Field
from agent.schemas.note import Note, Step
from time import sleep

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
        observation = None

        for action_limit in range(3):
            note: Note= self.stm(self.workflow_instance_id).get("note")
            current_task = note.current_task()

            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is:{current_task.instruction}")
            plan = self.reasoning(current_task.instruction, observation)
            note.current_step().plan = plan
            note.save()
            self.stm(self.workflow_instance_id)["note"] = note
            # self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"After reasoning, I have the following plan:\n{plan}")
            
            execution_status, execution_results = self.tool_manager.execute_task(plan)
            if execution_status != "success":
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Plan execution failed")
                raise Exception(f"Execution failed, {execution_results['result']}")
            
            note.current_step().execute_result = execution_results[-1]['result']
            note.save()
            self.stm(self.workflow_instance_id)["note"] = note
            
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Plan execution completed")
            
            observation, vision_states = self.observing(current_task.proof_of_completion)

            self.callback.info(
                agent_id=self.workflow_instance_id,
                progress=f"The {action_limit+1} action finished",
                message=f'Here is the current status\n{str(self.stm(self.workflow_instance_id).get("note"))}',
            )
            if observation.is_done:
                current_task.is_done = True
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is completed.\n{str(note)}")
                note.save()
                self.stm(self.workflow_instance_id)["note"] = note
                if len(note.unfinished_tasks()) == 0:
                    return {"all_tasks_finished": True}
                else:
                    note.unfinished_tasks()[0].steps.append(Step(vision=vision_states))
                    note.save()
                    self.stm(self.workflow_instance_id)["note"] = note
                    return {"all_tasks_finished": False}
            else:
                note.current_step().vision = vision_states
                note.save()
                self.stm(self.workflow_instance_id)["note"] = note
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is not completed. Reason: {observation.reason}")
                
        self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"I tried 3 times, but still failed the task {current_task.instruction}")
        raise Exception("All attempts failed")
        
    
    def reasoning(self, task:str, observation:ObservationResult):
        note: Note = self.stm(self.workflow_instance_id)["note"]

        sys_prompt = PromptTemplate.from_file(CURRENT_PATH.joinpath("reasoning_prompt.prompt"), role="system")
        sys_prompt = sys_prompt.format(tools=self.tool_manager.generate_prompt())

        user_prompt = [f"The current task is:{task}"]
        if observation:
            user_prompt.extend([
                f"Your previous observation is:{observation.observation}",
                f"The reason for determining that the task is not completed is:{observation.reason}",
                "Now, please reason about the current task again."
            ])
        if note.current_step().vision:
            user_prompt.extend([
                "The environment around you and the corresponding vyaw value are as follows:"
            ])
            for item in note.current_step().vision:
                user_prompt.extend([
                    f"vyaw: {item.vyaw}, image:",
                    item.image,
                ])
        else:
            user_prompt.extend(["The current environment images are not provided. Only obtain them when absolutely necessary, otherwise you can finish the task."])

        completion = self.llm.generate(records=[
            Message.system(sys_prompt),
            Message.user(content=user_prompt)
        ])

        self.callback.info(
            agent_id=self.workflow_instance_id,
            progress=f"Reasoning",
            message=f'Reasoning completed. I have the following plan:\n{completion["choices"][0]["message"]["content"]}',
        )

        return completion["choices"][0]["message"]["content"]
    
    def observing(self, proof_of_completion:str):
        sleep(1)
        with open(CURRENT_PATH.joinpath("observing_prompt.prompt"), "r") as f:
            sys_prompt = f.read()

        get_surrounding_image = GetSurroundingImage()
        get_surrounding_image._parent = self
        res = get_surrounding_image._run(memorize=False)
        if res["code"] != 0:
            raise Exception("Get surrounding image failed")
        
        vision_states = res["vision_states"]

        user_prompt = [f"Completing the following conditions means you have completed the task:{proof_of_completion}", "The environment around you and the corresponding vyaw value are as follows:"]    
        for item in vision_states:
            user_prompt.extend([
                f"vyaw: {item.vyaw}, image:",
                item.image,
            ])
            
        completion = self.llm.generate(records=[
            Message.system(sys_prompt),
            Message.user(content=user_prompt)
        ],
        response_format=ObservationResult)

        res = ObservationResult.model_validate_json(completion["choices"][0]["message"]["content"])

        self.callback.info(
            agent_id=self.workflow_instance_id,
            progress=f"Observing",
            message=f'Observing completed.\n My observation is: {res.observation}\nIs the task completed? {res.is_done} \nThe reason is: {res.reason}',
        )
        
        return res, vision_states
