from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry

MEMORY_DECISION_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

For the given text query, you should:
1. Decide if searching memory would be helpful
2. If memory search is needed, determine if the image is required to form the search query
3. If no memory or image is needed, provide a direct answer

Examples:
"What did I eat yesterday?"
-> Memory required: Yes
-> Image required: No
-> Query: "food items consumed yesterday"

"Did I eat this before?"
-> Memory required: Yes
-> Image required: Yes
-> Reason: Need image to identify the food item for memory search

"What is in this image?"
-> Memory required: No
-> Image required: Yes
-> Reason: Question requires direct description of the current image without needing past context

"What is the capital of France?"
-> Memory required: No
-> Image required: N/A
-> Direct answer: The capital of France is Paris.
-> Reason: Simple factual question that doesn't need memory or image

Format your response as:
MEMORY_REQUIRED: YES/NO
IMAGE_REQUIRED: YES/NO/N/A
DIRECT_QUERY: <query if memory needed and image not required>
DIRECT_ANSWER: <answer if no memory or image needed>
REASON: <brief explanation>"""

@registry.register_worker()
class MemoryDecisionWorker(BaseWorker, BaseLLMBackend):
    """Initial worker that decides if memory search is needed and if image is required"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        # Make decision based on text only
        decision_messages = [
            Message(role="system", message_type="text", content=MEMORY_DECISION_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
        ]

        response = self.llm.generate(records=decision_messages)
        decision_response = response["choices"][0]["message"]["content"]

        # Parse response
        lines = decision_response.split('\n')
        memory_required = False
        image_required = False
        direct_query = None
        direct_answer = None
        reason = None

        for line in lines:
            if line.startswith('MEMORY_REQUIRED:'):
                memory_required = 'YES' in line
            elif line.startswith('IMAGE_REQUIRED:'):
                image_required = 'YES' in line
            elif line.startswith('DIRECT_QUERY:'):
                direct_query = line.split(':', 1)[1].strip()
            elif line.startswith('DIRECT_ANSWER:'):
                direct_answer = line.split(':', 1)[1].strip()
            elif line.startswith('REASON:'):
                reason = line.split(':', 1)[1].strip()

        input_has_image = self.stm(self.workflow_instance_id).get("image_cache", None) is not None

        # store the query in the STM if exists
        if direct_query and direct_query != "N/A":
            self.stm(self.workflow_instance_id)["memory_search_query"] = direct_query

        if direct_answer:
            final_decision = "output_formatter"
        elif memory_required and image_required:
            final_decision = "multimodal_query_generator"
        elif memory_required:
            final_decision = "memory_search"
        else:
            final_decision = "answer_generator"

        return {
            "memory_required": memory_required,
            "image_required": image_required and input_has_image,
            "memory_search_query": direct_query if direct_query != "N/A" else None,
            "no_memory_required": not memory_required,
            "skip_everything": not memory_required and not image_required,
            "final_answer": direct_answer if not memory_required and not image_required else None,
            "reason": reason,
            "final_decision": final_decision
        } 