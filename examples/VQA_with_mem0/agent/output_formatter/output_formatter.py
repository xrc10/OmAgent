from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

@registry.register_worker()
class OutputFormatter(BaseWorker):
    """Output the final_answer in STM to user
    """

    def _run(self, *args, **kwargs):
        final_answer = self.stm(self.workflow_instance_id).get("final_answer", None)
        if final_answer:
            self.output.send_output(
                workflow_instance_id=self.workflow_instance_id,
                output=final_answer,
            )

        return {"final_answer": final_answer}

