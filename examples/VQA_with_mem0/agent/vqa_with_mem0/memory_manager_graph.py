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

if __name__ == "__main__":
    config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://32e3e83b.databases.neo4j.io",
            "username": "neo4j",
            "password": "owxykmr4P7AZjwgcteZLcrTvqx5mLyLxvOOC-XMu5gM"
        }
    },
    "version": "v1.1"
}

m = Memory.from_config(config_dict=config)

m.add("I like soccer", user_id="iuasodfu83")
m.add("My name is Ruochen", user_id="iuasodfu83")

m.get_all(user_id="iuasodfu83")

res = m.search("tell me my name.", user_id="iuasodfu83")
print(res)