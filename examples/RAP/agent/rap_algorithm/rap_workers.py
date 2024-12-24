from pathlib import Path
from typing import List

from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.utils.registry import registry
from omagent_core.models.llms.schemas import Message, Content
from omagent_core.utils.general import encode_image
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.container import container
from omagent_core.models.llms.prompt.prompt import PromptTemplate
from pydantic import Field
from collections import defaultdict
import re
from typing import List
import math
from .tree_structure import *


CURRENT_PATH = Path(__file__).parents[0]

# I dispay some variables here.
N_SHOT = 4 # number of the in-context examples
DEPTH_LIMIT = 4 # avoiding too deep search
SUB_QUESTION_GEN_NUM = 3
YES_SAMPLE_NUM = 10
ANSWER_GEN_NUM = 5
MCTS_ITER_NUM = 1 # This should be set a large number when applying in the realworld. I set it as 1 for debug.


@registry.register_worker()
class Selection(BaseWorker, BaseLLMBackend):

    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        assert self.stm(self.workflow_instance_id).get('data_input', None) is not None
        
        data_input = self.stm(self.workflow_instance_id)['data_input']
        task = self.stm(self.workflow_instance_id)['task']

        # Init the tree and some variables
        if self.stm(self.workflow_instance_id).get('tree', None) is None:
            self.stm(self.workflow_instance_id)['tree'] = SearchTree(data_input)
            self.stm(self.workflow_instance_id)['in_simulation'] = False # Whether the process is currently in simulation.
        tree:SearchTree = self.stm(self.workflow_instance_id)['tree']

        # Selection
        selected_path = []
        node = tree.root
        while True:
            selected_path.append(node)
            if node.children is None or len(node.children) == 0 or node.depth >= DEPTH_LIMIT:
                break
            node = MCTS.uct_select(node)

        # For debug
        info_str = '\n'.join([n.state for n in selected_path])
        self.callback.send_answer(self.workflow_instance_id, msg=info_str)
        self.stm(self.workflow_instance_id)['selected_path'] = selected_path

        return {"selected_path": selected_path}
    
@registry.register_worker()
class Expansion(BaseWorker, BaseLLMBackend):
    llm: OpenaiGPTLLM
    def reset_max_tokens(self):
        self.llm.max_tokens = 2048

    def get_actions_math(self, node: Node, path: List[Node]):
        task = self.stm(self.workflow_instance_id)['task']
        question = self.stm(self.workflow_instance_id)['tree'].data_input
        qid = N_SHOT + 1
        # Get new sub_questions
        question_info = f"Question {qid}: {question}\n"
        for i, n in enumerate(path):
            if i==0:
                continue
            question_info += f"Question {qid}.{i}: {n.action}\n"
            question_info += f"Answer {qid}.{i}: {n.state}\n"
        question_info += f"Question {qid}.{len(path)}:"
        self.prompts = self.set_prompts([CURRENT_PATH.joinpath(f'prompts/{task}/ic_examples.prompt')])
        # Genarating 4 sub_quesitons
        sub_questions = [self.simple_infer(input=question_info)["choices"][0]["message"]["content"] for _ in range(SUB_QUESTION_GEN_NUM)]
        sub_questions = [s.split('\n')[0].strip() for s in sub_questions]
        sub_questions = [s.replace(f"Question {qid}.{len(path)}: ","") for s in sub_questions]
        self.callback.send_answer(self.workflow_instance_id, msg='\n'.join(sub_questions))
        actions = sub_questions
        # deduplicate
        actions = list(set(actions))
        return actions
    
    def get_action_reward_math(self, action, path: List[Node]):
        task = self.stm(self.workflow_instance_id)['task']
        question = self.stm(self.workflow_instance_id)['tree'].data_input
        qid = N_SHOT + 1
        question_info = f"Question {qid}: {question}\n"
        # Evaluate whether the new sub_quesiton is useful.
        for i, n in enumerate(path):
            if i==0:
                continue
            question_info += f"Question {qid}.{i}: {n.action}\n"
        question_info += f"New question {qid}.{len(path)+1}: {action}"
        self.llm.max_tokens = 1
        # In RAP paper, it calculate the logits ratio between Yes and No just like ZoomEye
        # Since OpenAI’s API cannot return logits, I use this approach to replace the original computation method.
        self.prompts = self.set_prompts([CURRENT_PATH.joinpath(f'prompts/{task}/action_reward.prompt')])
        responses = [self.simple_infer(input=question_info)["choices"][0]["message"]["content"] for _ in range(YES_SAMPLE_NUM)]
        # For debug
        self.callback.send_answer(self.workflow_instance_id, msg='\n'.join(responses))
        yes_ratio = sum(["Yes" in x for x in responses]) / YES_SAMPLE_NUM
        self.reset_max_tokens()
        return yes_ratio, {"r_useful": yes_ratio}
    
    def get_state_math(self, action, path):
        task = self.stm(self.workflow_instance_id)['task']
        question = self.stm(self.workflow_instance_id)['tree'].data_input
        qid = N_SHOT + 1
        question_info = f"Question {qid}: {question}\n"
        for i, n in enumerate(path):
            if i==0:
                continue
            if i==len(path) - 1:
                break
            question_info += f"Question {qid}.{i}: {n.action}\n"
            question_info += f"Answer {qid}.{i}: {n.state}"
        question_info += f"Question {qid}.{len(path) - 1}: {n.action}\n"
        question_info += f"Answer {qid}.{len(path) - 1}:"

        force = False
        if path[-1].depth >= DEPTH_LIMIT:
            # Answer by force
            question_info += " Now we can answer the question."
            force = True

        self.prompts = self.set_prompts([CURRENT_PATH.joinpath(f'prompts/{task}/ic_examples.prompt')])
        def retrieve_answer(output: str) -> Optional[str]:
            match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
            if match is None:
                return None
            answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
            if '=' in answer:
                answer = answer[answer.rindex('=') + 1:]
            return answer
        # Answering sub_question
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        # For debug
        self.callback.send_answer(self.workflow_instance_id, msg=question_info)
        for _ in range(ANSWER_GEN_NUM):
            output = self.simple_infer(input=question_info)["choices"][0]["message"]["content"]
            output = output.split('\n')[0].strip()
            self.callback.send_answer(self.workflow_instance_id, msg=output)
            answer = retrieve_answer(output)
            answer_dict[answer].append(output)
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())
        
        state = answer
        if force:
            state = "Now we can answer the question. " + state
        aux = {'confidence': confidence}
        return state, aux

    def cal_reward(self, r_useful, confidence=None):
        if confidence is None:
            confidence = 1
        return (r_useful ** 0.8) * confidence ** (1 - 0.8), {'r_useful': r_useful, 'r_conf': confidence}

    def _run(self, *args, **kwargs):
        path = self.stm(self.workflow_instance_id)['selected_path']
        node: Node = path[-1]
        task = self.stm(self.workflow_instance_id)['task']
        if node.state is None:
            get_state = getattr(self, f"get_state_{task}", None)
            state, aux = get_state(node.action, path)
            node.state = state
            node.reward, node.reward_details = self.cal_reward(**node.fast_reward_details, **aux)
            if "Now we can answer" in state:
                node.is_terminal = True
        if node.children is None and not node.is_terminal:
            children = []
            # Get actions
            get_actions = getattr(self, f"get_actions_{task}", None)
            actions = get_actions(node, path)
            # Expansion
            for action in actions:
                get_action_reward = getattr(self, f"get_action_reward_{task}", None)
                fast_reward, fast_reward_details = get_action_reward(action, path)
                # For debug
                self.callback.send_answer(self.workflow_instance_id, msg=f'{action}\n{fast_reward}')
                child = Node(state=None, action=action, parent=node, fast_reward=fast_reward, fast_reward_details=fast_reward_details)
                children.append(child)
            
            node.children = children
            self.stm(self.workflow_instance_id)['selected_path'] = path
        return 


@registry.register_worker()
class SimulationPreProcess(BaseWorker, BaseLLMBackend):

    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        self.stm(self.workflow_instance_id)['in_simulation'] = True
        self.callback.send_answer(self.workflow_instance_id, msg=f'start simulation')
        return 


@registry.register_worker()
class SimulationPostProcess(BaseWorker, BaseLLMBackend):

    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        node: Node = self.stm(self.workflow_instance_id)['selected_path'][-1]
        if node.depth >= DEPTH_LIMIT or node.children is None:
            self.stm(self.workflow_instance_id)['in_simulation'] = False
            self.callback.send_answer(self.workflow_instance_id, msg=f'Done.{node.action}\n{node.state}')
            return {"finish": True}
        fast_rewards = [child.fast_reward for child in node.children]
        # For debug
        for c, r in zip(node.children, fast_rewards):
            self.callback.send_answer(self.workflow_instance_id, msg=f'{c.action}\n{r}')
        node = node.children[np.argmax(fast_rewards)]
        self.callback.send_answer(self.workflow_instance_id, msg=f'Choose node: {node.action}')
        # Using append here does not work...
        # self.stm(self.workflow_instance_id)['selected_path'].append(node)
        self.stm(self.workflow_instance_id)['selected_path'] = self.stm(self.workflow_instance_id)['selected_path'] + [node]
        return {"finish": False}
    

@registry.register_worker()
class BackPropagation(BaseWorker, BaseLLMBackend):
    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        path = self.stm(self.workflow_instance_id)['selected_path']
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return

    def cum_reward(self, rewards):
        return sum(rewards)


@registry.register_worker()
class MCTSCompletionCheck(BaseWorker, BaseLLMBackend):

    llm: OpenaiGPTLLM

    def _run(self, *args, **kwargs):
        if self.stm(self.workflow_instance_id).get("loop_index", None) is None:
            self.stm(self.workflow_instance_id)["loop_index"] = 0
            self.stm(self.workflow_instance_id)['candidates_path'] = []
        self.stm(self.workflow_instance_id)["loop_index"] = self.stm(self.workflow_instance_id)["loop_index"] + 1

        path = self.stm(self.workflow_instance_id)['selected_path']
        self.stm(self.workflow_instance_id)['candidates_path'] = self.stm(self.workflow_instance_id)['candidates_path'] + [path]
        finish = self.stm(self.workflow_instance_id)["loop_index"] >= MCTS_ITER_NUM
        

        return {"finish": finish}

@registry.register_worker()
class OutputInterface(BaseWorker, BaseLLMBackend):

    llm: OpenaiGPTLLM

    def format_output_math(self, path):
        question = self.stm(self.workflow_instance_id)['tree'].data_input
        qid = N_SHOT + 1
        # Get new sub_questions
        question_info = f"Question: {question}\n"
        for i, n in enumerate(path):
            if i==0:
                continue
            question_info += f"Question 1.{i}: {n.action}\n"
            question_info += f"Answer 1.{i}: {n.state}\n"
        return question_info


    def _run(self, *args, **kwargs):
        # I choose a path randomly here for debug, and it should be choosen based on some rational metric such as max_reward
        import random
        path = random.choice(self.stm(self.workflow_instance_id)['candidates_path'])
        task = self.stm(self.workflow_instance_id)['task']
        format_output = getattr(self, f"format_output_{task}", None)
        output = format_output(path)
        self.callback.send_answer(self.workflow_instance_id, msg=output)

        return None