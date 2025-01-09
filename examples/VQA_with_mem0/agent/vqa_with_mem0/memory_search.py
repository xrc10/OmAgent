from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from .memory_manager import MemoryManager

@registry.register_worker()
class MemorySearch(BaseWorker):
    """Memory search worker"""

    llm: OpenaiGPTLLM
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_manager = MemoryManager()

    def _run(self, user_instruction: str, *args, **kwargs):
        memory_search_query = self.stm(self.workflow_instance_id).get("memory_search_query", None)

        # Directly use the user instruction as the search query
        relevant_memories = self.memory_manager.search_memory(memory_search_query)
        
        # Store results in STM for next step
        self.stm(self.workflow_instance_id)["memory_search_results"] = {
            "search_success": bool(relevant_memories),  # True if memories found
            "search_query": memory_search_query,
            "relevant_memories": relevant_memories
        }

        return {
            "search_success": bool(relevant_memories),
            "search_query": memory_search_query,
            "relevant_memories": relevant_memories
        } 