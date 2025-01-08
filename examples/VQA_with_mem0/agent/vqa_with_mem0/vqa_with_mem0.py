from pathlib import Path
from typing import List

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.prompt.parser import StrParser
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.container import container
from omagent_core.utils.general import encode_image
from omagent_core.utils.registry import registry

from mem0 import Memory
from omagent_core.utils.logger import logging

class MemoryManager:
    """Manages memory operations using mem0 with vector store"""
    
    def __init__(self, user_id="default_user"):
        # Initialize mem0 with default vector store
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

MEMORY_SEARCH_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

For the given user query, you should decide if searching memory would be helpful.

Format your response as:
SEARCH_MEMORY: YES/NO
SEARCH_QUERY: <query if searching>"""

ANSWER_AND_STORE_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

Use the provided image and any retrieved memories (if available) to answer the user's question.
After answering, decide if the current interaction should be stored in memory.

Format your response as:
ANSWER: <your answer to the user>
STORE_MEMORY: YES/NO
MEMORY_CONTENT: <content to store if storing>"""

@registry.register_worker()
class VQAWithMem0(BaseWorker, BaseLLMBackend):
    """Visual Question Answering processor with memory capabilities"""

    llm: OpenaiGPTLLM
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_manager = MemoryManager()

    def _run(self, user_instruction: str, *args, **kwargs):
        # First LLM call - Decide about memory search
        search_messages = [
            Message(role="system", message_type="text", content=MEMORY_SEARCH_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
        ]

        # Add image to first LLM call if available
        if self.stm(self.workflow_instance_id).get("image_cache", None):
            img = self.stm(self.workflow_instance_id)["image_cache"]["<image_0>"]
            search_messages.append(
                Message(
                    role="user",
                    message_type="image",
                    content=[
                        Content(
                            type="image_url",
                            image_url={
                                "url": f"data:image/jpeg;base64,{encode_image(img)}"
                            },
                        )
                    ],
                )
            )

        response = self.llm.generate(records=search_messages)
        search_response = response["choices"][0]["message"]["content"]

        self.callback.send_answer(self.workflow_instance_id, msg=f"Memory search response: {search_response}")

        # Parse search decision
        lines = search_response.split('\n')
        search_memory = False
        search_query = None
        relevant_memories = None

        for line in lines:
            if line.startswith('SEARCH_MEMORY:'):
                search_memory = 'YES' in line
            elif line.startswith('SEARCH_QUERY:'):
                search_query = line.split(':', 1)[1].strip()

        self.callback.send_answer(self.workflow_instance_id, msg=f"Search memory: {search_memory}\nSearch query: {search_query}")

        # Second LLM call - Answer question and decide about storing
        answer_messages = [
            Message(role="system", message_type="text", content=ANSWER_AND_STORE_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
        ]

        # Add image if available
        if self.stm(self.workflow_instance_id).get("image_cache", None):
            img = self.stm(self.workflow_instance_id)["image_cache"]["<image_0>"]
            answer_messages.append(
                Message(
                    role="user",
                    message_type="image",
                    content=[
                        Content(
                            type="image_url",
                            image_url={
                                "url": f"data:image/jpeg;base64,{encode_image(img)}"
                            },
                        )
                    ],
                )
            )

        # Add relevant memories if found
        if search_memory and search_query:
            self.callback.send_answer(
                self.workflow_instance_id,
                msg=f"📚 Searching memories for query: {search_query}"
            )
            relevant_memories = self.memory_manager.search_memory(search_query)
            if relevant_memories:
                # Send retrieved memories to user
                self.callback.send_answer(
                    self.workflow_instance_id,
                    msg=f"📚 Retrieved memories: {relevant_memories}"
                )
                answer_messages.append(
                    Message(
                        role="system",
                        message_type="text",
                        content=f"Relevant memories found: {relevant_memories}"
                    )
                )

        # Get final answer and memory storage decision
        response = self.llm.generate(records=answer_messages)
        answer_response = response["choices"][0]["message"]["content"]

        self.callback.send_answer(self.workflow_instance_id, msg=f"Final answer response: {answer_response}")

        # Parse answer and storage decision
        lines = answer_response.split('\n')
        store_memory = False
        memory_content = None
        answer = None

        for line in lines:
            if line.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()
            elif line.startswith('STORE_MEMORY:'):
                store_memory = 'YES' in line
            elif line.startswith('MEMORY_CONTENT:'):
                memory_content = line.split(':', 1)[1].strip()

        # Store memory if needed
        if store_memory and memory_content:
            self.memory_manager.add_memory(
                memory_content,
                metadata={"type": "vqa_interaction"}
            )
            # Send notification about stored memory
            self.callback.send_answer(
                self.workflow_instance_id,
                msg=f"💾 Stored new memory: {memory_content}"
            )

        # Send final answer and return
        self.callback.send_answer(self.workflow_instance_id, msg=answer)
        return answer
