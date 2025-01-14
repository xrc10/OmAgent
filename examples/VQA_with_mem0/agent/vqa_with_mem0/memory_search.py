from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from .memory_manager import MemoryManager
import time

@registry.register_worker()
class MemorySearch(BaseWorker):
    """Memory search worker"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self, user_instruction: str, *args, **kwargs):
        # Start timing
        start_time = time.time()
        
        user_id = self.stm(self.workflow_instance_id).get("user_id", "default_user")
        self.memory_manager = MemoryManager(user_id=user_id)

        memory_search_query = self.stm(self.workflow_instance_id).get("memory_search_query", None)

        # Directly use the user instruction as the search query
        relevant_memories = self.memory_manager.search_memory(memory_search_query)
        
        # Keep only the top 5 memories
        relevant_memories = relevant_memories[:5]
        
        # Calculate elapsed time
        search_time = time.time() - start_time
        
        # Store results in STM for next step
        results = {
            "search_success": bool(relevant_memories),
            "search_query": memory_search_query,
            "relevant_memories": relevant_memories,
            "search_time": search_time
        }
        
        self.stm(self.workflow_instance_id)["memory_search_results"] = results
        return results