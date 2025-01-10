from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class OutputFormatter(BaseWorker):
    """Output the final_answer in STM to user
    """

    def _run(self, *args, **kwargs):
        final_answer = self.stm(self.workflow_instance_id).get("final_answer", None)
        if final_answer:
            self.callback.send_answer(
                self.workflow_instance_id,
                msg=f"{final_answer}"
            )

        return {"final_answer": final_answer}

