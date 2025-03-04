from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class ExitChecker(BaseWorker):
    """Checks if the user wants to exit the conversation"""

    def _run(self, *args, **kwargs):
        # Get user instruction from STM
        user_instruction = self.stm(self.workflow_instance_id).get("user_instruction", "")
        
        # Check if user wants to exit
        should_exit = user_instruction.strip().lower() == "退出"
        
        # Store conversation history for context
        conversation_history = self.stm(self.workflow_instance_id).get("conversation_history", [])
        
        # Get the current Q&A pair
        current_question = user_instruction
        current_answer = self.stm(self.workflow_instance_id).get("answer", "")
        
        # Add current Q&A to history if not exiting
        if not should_exit and current_question and current_answer:
            conversation_history.append({
                "question": current_question,
                "answer": current_answer
            })
            
            # Keep only the last 5 conversations for context
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
            
            # Update conversation history in STM
            self.stm(self.workflow_instance_id)["conversation_history"] = conversation_history
        
        return {"should_exit": should_exit} 