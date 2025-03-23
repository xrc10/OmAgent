from pydantic import BaseModel, Field
from typing import List

class Step(BaseModel):
    instruction: str = Field(description="The instruction of the step, describe what should be done. Should be simple and feasible")
    proof_of_completion: str = Field(description="How to prove the step is done. It should be able to verify everything at once. ")
    is_done: bool = Field(description="Whether the step is done. Makesure to use False", enum=[False])

class Note(BaseModel):
    content: List[Step]

    def __str__(self):
        result = []
        for step in self.content:
            checkbox = "[x]" if step.is_done else "[ ]"
            result.append(f"{checkbox} {step.instruction}")
        return "\n".join(result)
    
    def unfinished_steps(self):
        return [step for step in self.content if not step.is_done]
    
    def finished_steps(self):
        return [step for step in self.content if step.is_done]
    
    