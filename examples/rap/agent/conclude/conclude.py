from pathlib import Path
from typing import List
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.prompt import PromptTemplate
from omagent_core.utils.registry import registry
from pydantic import Field

CURRENT_PATH = Path(__file__).parents[0]

@registry.register_worker()
class Conclude(BaseLLMBackend, BaseWorker):
    """Conclude worker that formats the final reasoning output"""

    prompts: List[PromptTemplate] = Field(
        default=[
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("sys_prompt.prompt"),
                role="system"
            ),
            PromptTemplate.from_file(
                CURRENT_PATH.joinpath("user_prompt.prompt"),
                role="user"
            )
        ]
    )

    def _run(self, rap_structure: dict, final_answer: str, *args, **kwargs):
        """Format and present the final reasoning results.
        
        Args:
            rap_structure: The tree structure of the reasoning process
            final_answer: The final answer from RAP
            
        Returns:
            dict: Formatted response with conclusion
        """
        chat_complete_res = self.simple_infer(
            rap_structure=rap_structure,
            final_answer=final_answer
        )
        
        conclusion = chat_complete_res["choices"][0]["message"]["content"]
        
        self.callback.send_answer(
            agent_id=self.workflow_instance_id,
            msg=f"Final conclusion: {conclusion}"
        )
        
        return {"conclusion": conclusion} 