from pathlib import Path
from typing import List
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.utils.registry import registry
from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class Expansion(BaseLLMBackend, BaseWorker):
    """Expansion worker that implements MCTS expansion phase"""
    
    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("state_prediction.prompt"), 
                role="system"
            ),
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("action_reward.prompt"),
                role="user" 
            )
        ]
    )

    def _run(self, *args, **kwargs):
        path = self.stm(self.workflow_instance_id)['selected_path']
        node = path[-1]
        task = self.stm(self.workflow_instance_id)['task']

        # Get state if needed
        if node.state is None:
            get_state = getattr(self, f"get_state_{task}", None)
            state, aux = get_state(node.action, path)
            node.state = state
            node.reward, node.reward_details = self.cal_reward(**node.fast_reward_details, **aux)
            if "Now we can answer" in state:
                node.is_terminal = True

        # Expand if needed
        if node.children is None and not node.is_terminal:
            children = []
            get_actions = getattr(self, f"get_actions_{task}", None)
            actions = get_actions(node, path)

            # Create child nodes
            for action in actions:
                get_action_reward = getattr(self, f"get_action_reward_{task}", None)
                fast_reward, fast_reward_details = get_action_reward(action, path)
                child = Node(state=None, action=action, parent=node, 
                           fast_reward=fast_reward, fast_reward_details=fast_reward_details)
                children.append(child)
            
            node.children = children
            self.stm(self.workflow_instance_id)['selected_path'] = path

        return {} 