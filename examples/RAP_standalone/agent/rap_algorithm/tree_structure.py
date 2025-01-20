import itertools
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import numpy as np
'''
Haozhan comments:
    This tree structure is initialized for RAP implementation, planning to generalize to more algorithms, such as
    ToT and CoT (ChainList is a special form of a Tree).
'''

class SearchTree:
    def __init__(self, data_input):
        self.root = Node(state=data_input,action=None)
        self.data_input = data_input

# class State(NamedTuple):
#     info: None

# class Action(NamedTuple):
#     info: None

class Node:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: str, action: str, parent: "Optional[Node]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.max):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.max
        """
        self.id = next(Node.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[Node]]' = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)

class MCTS:

    w_exp: float = 1.

    @staticmethod
    def uct_select(node: Node):
        return max(node.children, key=MCTS.uct)
    
    @staticmethod
    def uct(node: Node):
        return node.Q + MCTS.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))