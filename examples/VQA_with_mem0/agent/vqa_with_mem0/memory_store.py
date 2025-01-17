from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry
from .memory_manager import MemoryManager

MEMORY_DECISION_PROMPT = """Analyze the following interaction and decide if it should be stored in memory.

Special handling:
- If the question contains phrases like "记一下", "记住", "记录" (meaning "remember", "note down", "record"), ALWAYS store the memory
- If it's a memory request:
  * For direct information in question (e.g. "记一下我吃了米饭"), extract and store that information
  * For image-related requests (e.g. "记录图片中的文档"), extract key information from the answer

When storing memories, include key details like:
- Visual descriptions (color, size, shape, brand)
- Temporal/contextual information
- Locations and spatial relationships
- Unique identifying characteristics

Store memories that contain:
- Factual information about objects/scenes
- Important context or relationships
- Unique/distinctive features
- Personal information or history
- ANY information when explicitly asked to remember

Don't store:
- Simple yes/no answers (unless explicitly asked to remember)
- Subjective opinions (unless explicitly asked to remember)
- Redundant information
- Generic observations

Previous interaction:
Question: {question}
Answer: {answer}

Format response as:
STORE_MEMORY: YES/NO
MEMORY_CONTENT: <detailed memory if storing>"""

@registry.register_worker()
class MemoryStore(BaseWorker, BaseLLMBackend):
    """Worker that handles memory storage decisions and operations"""
    
    llm: OpenaiGPTLLM

    def handle_memory(self, question: str, answer: str, user_id: str) -> tuple:
        memory_prompt = MEMORY_DECISION_PROMPT.format(
            question=question,
            answer=answer
        )
        
        memory_messages = [
            Message(role="system", message_type="text", content=memory_prompt)
        ]

        try:
            response = self.llm.generate(records=memory_messages)
            if not response or 'choices' not in response:
                self.logger.error(f"Invalid LLM response for memory decision: {response}")
                return True, answer

            memory_response = response["choices"][0]["message"]["content"]

            store_memory = False
            memory_content = None

            try:
                lines = memory_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('STORE_MEMORY:'):
                        store_memory = 'YES' in line.upper()
                    elif line.startswith('MEMORY_CONTENT:'):
                        memory_content = line.split(':', 1)[1].strip()
            except Exception as e:
                self.logger.error(f"Error parsing memory decision: {e}")
                store_memory = True
                memory_content = answer

        except Exception as e:
            self.logger.error(f"Error during memory decision LLM call: {e}")
            return True, answer

        return store_memory, memory_content

    def _run(self, user_instruction: str, *args, **kwargs):
        # Get answer from STM
        answer = self.stm(self.workflow_instance_id).get("answer", None)
        if not answer:
            return {"store_memory": False, "memory_content": None}

        # Get user_id from STM
        user_id = self.stm(self.workflow_instance_id).get("user_id", "default_user")

        # Make memory storage decision
        store_memory, memory_content = self.handle_memory(user_instruction, answer, user_id)

        # add user_instruction to memory_content
        memory_content = "User query: " + user_instruction + "\n" + "Answer: " + answer + "\n" + "Memory content: " + memory_content

        # Store memory if needed
        if store_memory and memory_content:
            memory_manager = MemoryManager(user_id=user_id)
            memory_manager.add_memory(
                memory_content,
                metadata={
                    "type": "vqa_interaction",
                    "user_id": user_id
                }
            )
            self.callback.send_answer(self.workflow_instance_id, msg="记忆已记录")

        return {
            "store_memory": store_memory,
            "memory_content": memory_content,
            "answer": answer
        } 