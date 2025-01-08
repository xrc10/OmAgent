from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from .memory_manager import MemoryManager

ANSWER_AND_STORE_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

Use the provided image and any retrieved memories (if available) to answer the user's question.
After answering, decide if the current interaction should be stored in memory.

Store memories that contain:
- Factual information about objects or scenes in the image
- Important relationships or context that might be useful later
- Unique or distinctive features worth remembering

Don't store memories that are:
- Simple yes/no answers
- Subjective opinions
- Redundant information already stored
- Generic observations that wouldn't help future interactions

Examples:
Q: "Is this a cat?"
A: "Yes, this is a cat."
Store: NO (simple yes/no answer)

Q: "What's in this kitchen?"
A: "The kitchen has white cabinets, a marble countertop, and stainless steel appliances including a double-door refrigerator."
Store: YES (detailed description that could be relevant for future questions)

Q: "Do you like how this room is decorated?"
A: "The room has a modern and appealing design."
Store: NO (subjective opinion)

Format your response as:
ANSWER: <your answer to the user>
STORE_MEMORY: YES/NO
MEMORY_CONTENT: <content to store if storing>"""

@registry.register_worker()
class VQAAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Second step of VQA process - generates answer and handles memory storage"""

    llm: OpenaiGPTLLM
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_manager = MemoryManager()

    def _run(self, user_instruction: str, *args, **kwargs):
        memory_search_results = self.stm(self.workflow_instance_id).get("memory_search", {})
        search_memory = memory_search_results.get("search_needed", False)
        search_query = memory_search_results.get("search_query")

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

        relevant_memories = None
        # Search and add memories if needed
        if search_memory and search_query:
            relevant_memories = self.memory_manager.search_memory(search_query)
            if relevant_memories:
                # self.callback.send_answer(self.workflow_instance_id, msg=f"Relevant memories found: {relevant_memories[0]}")
                answer_messages.append(
                    Message(
                        role="system",
                        message_type="text",
                        content=f"Relevant memories found: {relevant_memories}"
                    )
                )

        # Generate final answer
        response = self.llm.generate(records=answer_messages)
        answer_response = response["choices"][0]["message"]["content"]

        # Parse response
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
            # self.callback.send_answer(self.workflow_instance_id, msg=f"Memory stored: {memory_content}")
        self.callback.send_answer(self.workflow_instance_id, msg=f"{answer}")

        return {
            "answer": answer,
            "relevant_memories": relevant_memories,
            "memory_stored": store_memory,
            "memory_content": memory_content
        }
