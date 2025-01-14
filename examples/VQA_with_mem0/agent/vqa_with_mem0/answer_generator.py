from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from .memory_manager import MemoryManager
from time import time

ANSWER_AND_STORE_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

Answer questions concisely based on the image and retrieved memories (if available). Use the same language as the query.
After answering, decide if the current interaction should be stored in memory.

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

Don't store:
- Simple yes/no answers
- Subjective opinions
- Redundant information
- Generic observations

Format response as:
ANSWER: <concise answer>
STORE_MEMORY: YES/NO
MEMORY_CONTENT: <detailed memory if storing>"""

@registry.register_worker()
class VQAAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Second step of VQA process - generates answer and handles memory storage"""

    llm: OpenaiGPTLLM
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self, user_instruction: str, *args, **kwargs):
        user_id = self.stm(self.workflow_instance_id).get("user_id", "default_user")
        self.memory_manager = MemoryManager(user_id=user_id)

        # Get memory search results from previous step
        memory_search_results = self.stm(self.workflow_instance_id).get("memory_search_results", {})
        relevant_memories = memory_search_results.get("relevant_memories", None)

        # Filter memories by score threshold and format them
        memory_context = ""
        if relevant_memories:
            filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= 0.2]
            if filtered_memories:
                memory_context = "\nRelevant memories from past interactions:\n" + "\n".join(
                    [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                )

        answer_messages = [
            Message(role="system", message_type="text", content=ANSWER_AND_STORE_PROMPT),
            Message(role="user", message_type="text", content=user_instruction + memory_context)
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

        # Time the LLM API call
        start_time = time()
        response = self.llm.generate(records=answer_messages)
        llm_time = time() - start_time
        answer_response = response["choices"][0]["message"]["content"]

        # Parse response - more robust parsing
        lines = answer_response.split('\n')
        store_memory = False
        memory_content = None
        answer = answer_response  # Default to full response if parsing fails

        try:
            for line in lines:
                line = line.strip()
                if line.startswith('ANSWER:'):
                    answer = line.split(':', 1)[1].strip()
                elif line.startswith('STORE_MEMORY:'):
                    store_memory = 'YES' in line.upper()
                elif line.startswith('MEMORY_CONTENT:'):
                    memory_content = line.split(':', 1)[1].strip()
            
            # If no structured format was found, use the entire response as the answer
            if answer == answer_response and len(lines) > 0:
                # Assume we should store verification codes
                store_memory = True
                memory_content = answer
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            answer = answer_response  # Use full response as fallback
            store_memory = True  # Store verification codes by default
            memory_content = answer_response

        self.callback.send_answer(self.workflow_instance_id, msg=f"{answer}")

        # Store memory if needed
        if store_memory and memory_content:
            self.memory_manager.add_memory(
                memory_content,
                metadata={"type": "vqa_interaction"}
            )

        # remove the image part before saving answer messages
        answer_messages = [msg for msg in answer_messages if msg.message_type != "image"]

        return {
            "answer": answer,
            "relevant_memories": relevant_memories,
            "memory_stored": store_memory,
            "memory_content": memory_content,
            "llm_time": llm_time,  # Add timing to output
            "answer_messages_without_image": answer_messages
        }
