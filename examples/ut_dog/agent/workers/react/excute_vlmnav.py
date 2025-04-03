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
import openai
from agent.tools.move import Move
import base64
import requests
import re
import json

CURRENT_PATH = Path(__file__).parents[0]

# name: OpenaiGPTLLM
# model_id: gpt-4o-utdog
# api_key: ${env| custom_openai_key, sk-2ju1fJ8hzN63oAva3tOiajUOyO756tqRt2XPD8N0yl2nqWL7}
# endpoint: ${env| custom_openai_endpoint, http://10.8.17.22:3000/v1}
# temperature: 0
# vision: true

VLMNAV_SYSTEM_PROMPT = """
You are an embodied robotic assistant,with an RGB image sensor.You observe the image and instructions given to you and output a textual response,which is converted into actions that physically move you within the environment
"""

VLMNAV_USER_PROMPT = """

### **Task Instructions:**  
**Goal:** Navigate to the {goal_object} and get as close to it as possible.  

**Guidelines:**  
1. Use your knowledge of where items are typically located (e.g., fridge in the kitchen, towels in the bathroom).  
2. You will see **red arrows** on your observation, each labeled with a number in a white circle. These represent possible actions (movement directions).  
3. If you choose an action, you will move to the location that arrow points to.  
4. **Action 0** means **turn around**—use this if no good options are available or you need to backtrack.  
5. **Obstacles:** You **cannot** go through closed doors, and you **do not** need to go up or down stairs.  

**Response Format:**  
1. **Describe what you see:** List objects, doors, and potential actions (red arrows with numbers). Mention if you spot the {goal_object} or clues about its location.  
2. **Determine direction:** Based on typical item placement, decide where the {goal_object} is likely to be (e.g., "toward the kitchen").  
3. **Choose the best action:** Pick the numbered arrow that aligns with your chosen direction. Return your decision as `{'action': <number>}`.  
"""

class VLM_API:
    def __init__(self):
        self.api_key = "sk-2ju1fJ8hzN63oAva3tOiajUOyO756tqRt2XPD8N0yl2nqWL7"
        self.endpoint = "http://10.8.17.22:3000/v1"
        self.model_id = "gpt-4o-utdog"
        self.temperature = 0
        self.vision = True

    def move_forward(self, distance: float):
        '''
        move forward with a certain distance
        '''
        move = Move()
        move._parent = self
        move._run(vx=distance, vy=0, vyaw=0)

    def turn_degree(self, degree: float):
        '''
        turn with a certain degree
        '''
        move = Move()
        move._parent = self
        vyaw = - degree / 45 * 1.5
        move._run(vx=0, vy=0, vyaw=vyaw)

    def call_vlm(self, system_prompt:str, user_prompt:str, image_base64:str):
        openai.api_key = self.api_key
        openai.api_base = self.endpoint
        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            images=[{"type": "base64", "image": image_base64}],
            temperature=self.temperature,
            vision=self.vision,
        )
        return response.choices[0].message.content

class ObservationResult(BaseModel):
    observation: str = Field(description="The observation of the task")
    reason: str = Field(description="Criteria for determining whether a task is completed")
    is_done: bool = Field(description="Whether the task is done")

@registry.register_worker()
class ReactExcuteVLMNav(BaseWorker):
    # tool_manager: ToolManager

    def _run(self, *args, **kwargs):
        # Read user input through configured input interface
        observation = None
        vlm_api = VLM_API()

        for action_limit in range(3):
            note: Note= self.stm(self.workflow_instance_id).get("note")
            current_task = note.current_task()

            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"The current task is:{current_task.instruction}")
            plan = self.reasoning(current_task.instruction, observation)
            note.current_step().plan = plan
            note.save()
            self.stm(self.workflow_instance_id)["note"] = note
            # self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"After reasoning, I have the following plan:\n{plan}")
            
            # execution_status, execution_results = self.tool_manager.execute_task(plan)
            # if execution_status != "success":
            #     self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Plan execution failed")
            #     raise Exception(f"Execution failed, {execution_results['result']}")
            
            # note.current_step().execute_result = execution_results[-1]['result']
            # note.save()
            # self.stm(self.workflow_instance_id)["note"] = note

            # start of execution
            # 1. Get front camera image and add visual action options
            get_surrounding_image = GetSurroundingImage()
            get_surrounding_image._parent = self
            res = get_surrounding_image._run(memorize=False)
            if res["code"] != 0:
                raise Exception("Get surrounding image failed")
            
            front_image_base64 = res["image"]
            
            # Call action proposal API to add navigation options
            action_api_url = "http://localhost:8075/generate_action_proposals"
            action_payload = {
                "image": front_image_base64,
                "min_angle": 40,
                "number_size": 30,
                "min_path_length": 200
            }
            
            action_response = requests.post(action_api_url, json=action_payload, timeout=60)
            if action_response.status_code != 200:
                raise Exception(f"Action proposal API failed: {action_response.text}")
            
            action_data = action_response.json()
            actions = action_data["actions"]
            image_with_actions = action_data["image"]
            
            # 2. Call VLM with the system and user prompt
            goal_object = current_task.instruction.replace("navigate to", "").strip().replace("导航到", "").strip()
            formatted_user_prompt = VLMNAV_USER_PROMPT.format(goal_object=goal_object)
            
            vlm_response = vlm_api.call_vlm(
                system_prompt=VLMNAV_SYSTEM_PROMPT,
                user_prompt=formatted_user_prompt,
                image_base64=image_with_actions
            )
            
            self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"VLM response: {vlm_response}")
            
            # 3. Parse the model's choice of action

            
            # Extract action from response using regex
            action_match = re.search(r"{'action'\s*:\s*(\d+)}", vlm_response)
            if not action_match:
                raise Exception(f"Could not parse action from VLM response: {vlm_response}")
                
            chosen_action = int(action_match.group(1))
            
            # Find the corresponding turning degree for the chosen action
            turning_degree = None
            for action in actions:
                if action["action_number"] == chosen_action:
                    turning_degree = action["turning_degree"]
                    break
            
            if turning_degree is None:
                raise Exception(f"Could not find turning degree for action {chosen_action}")
                
            # Execute the turning action
            vlm_api.turn_degree(turning_degree)

            # Execute the move forward action
            vlm_api.move_forward(0.5)

            # Save the result to the API
            save_result_url = "http://localhost:8075/save_result"
            save_payload = {
                "image": image_with_actions,
                "vlm_output": vlm_response,
                "action_number": chosen_action,
                "step_id": action_limit,
                "additional_info": {
                    "turning_degree": turning_degree,
                    "goal_object": goal_object
                }
            }
            
            save_response = requests.post(save_result_url, json=save_payload, timeout=60)
            if save_response.status_code != 200:
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Warning: Failed to save result: {save_response.text}")
            else:
                self.callback.send_block(agent_id=self.workflow_instance_id, msg=f"Result saved successfully")

            # end of execution
            
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
