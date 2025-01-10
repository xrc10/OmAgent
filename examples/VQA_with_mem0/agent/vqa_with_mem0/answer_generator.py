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

When storing memories, always include:
- Detailed visual descriptions of the objects/scenes being discussed (color, size, shape, brand if visible)
- Temporal or contextual information provided by the user
- Specific locations or spatial relationships
- Any unique identifying characteristics

For example, instead of just storing "This is a pen I bought yesterday", store "This is a blue and silver Parker ballpoint pen, approximately 5.5 inches long, with a metallic clip, purchased yesterday from the stationery store"

Store memories that contain:
- Factual information about objects or scenes in the image
- Important relationships or context that might be useful later
- Unique or distinctive features worth remembering
- Personal information or history related to the objects

Don't store memories that are:
- Simple yes/no answers
- Subjective opinions
- Redundant information already stored
- Generic observations that wouldn't help future interactions

Examples:
Q: "Is this a cat?"
A: "Yes, this is a cat."
Store: NO (simple yes/no answer)

Q: "这支笔是我前天买的"
A: "好的，我已经记下这支笔是您前天购买的。这是一支银色的钢笔，笔身上有金色装饰，长度大约14厘米。"
Store: YES (includes both temporal and detailed visual information)
MEMORY_CONTENT: "用户前天购买了一支银色钢笔，笔身有金色装饰，长约14厘米，笔尖细长。"

Format your response as:
ANSWER: <your answer to the user>
STORE_MEMORY: YES/NO
MEMORY_CONTENT: <content to store if storing, including detailed visual description>"""

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

        # Add relevant memories if available
        if relevant_memories:
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

        return {
            "answer": answer,
            "relevant_memories": relevant_memories,
            "memory_stored": store_memory,
            "memory_content": memory_content
        }
