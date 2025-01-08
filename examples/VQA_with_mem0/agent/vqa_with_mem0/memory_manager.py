from mem0 import Memory
from omagent_core.utils.logger import logging

class MemoryManager:
    """Manages memory operations using mem0 with vector store"""
    
    def __init__(self, user_id="default_user"):
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                }
            },
        }
        self.memory = Memory.from_config(config)
        self.user_id = user_id
        
    def search_memory(self, query):
        """Search for relevant memories using vector similarity"""
        try:
            memories = self.memory.search(query=query, user_id=self.user_id)
            return memories
        except Exception as e:
            logging.error(f"Error searching memory: {e}")
            return None
            
    def add_memory(self, content, metadata=None):
        """Add new memory to vector store"""
        try:
            result = self.memory.add(
                content,
                user_id=self.user_id,
                metadata=metadata or {}
            )
            return result
        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            return None 