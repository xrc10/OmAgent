from pathlib import Path
from typing import List
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.utils.registry import registry
from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class MCTSCompletionCheck(BaseLLMBackend, BaseWorker):
    """MCTS completion check worker that determines when to stop MCTS"""

    def _run(self, *args, **kwargs):
        # Initialize or increment loop counter
        if self.stm(self.workflow_instance_id).get("loop_index", None) is None:
            self.stm(self.workflow_instance_id)["loop_index"] = 0
            self.stm(self.workflow_instance_id)['candidates_path'] = []
            
        self.stm(self.workflow_instance_id)["loop_index"] += 1

        # Store current path as candidate
        path = self.stm(self.workflow_instance_id)['selected_path']
        self.stm(self.workflow_instance_id)['candidates_path'].append(path)

        # Check if we've reached max iterations
        finish = self.stm(self.workflow_instance_id)["loop_index"] >= MCTS_ITER_NUM

        return {"finish": finish} 