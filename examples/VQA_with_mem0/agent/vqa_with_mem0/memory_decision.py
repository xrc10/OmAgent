from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry

MEMORY_DECISION_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions. You should lean towards using memory when there's any possibility of relevant past information.

For the given text query, you should:
1. Decide if searching memory could be helpful (be generous - if there's any chance past information could be relevant, say yes)
2. Determine if the image is required for memory search (if memory is needed)
3. Determine if the image is required to answer the query (regardless of memory)
4. If no memory and no image is needed everywhere and the query is trivial to answer, provide a direct answer

Memory should be used when:
- Query asks about past information (what did I do, when did I see, etc.)
- Query mentions or implies past time (today, yesterday, before, again, usually, etc.) when asking for information
- Query asks about patterns or frequencies (how often, how many times, etc.)
- Query compares current with past (is this different, has this changed, etc.)
- Query refers to user preferences or history (do I like, have I seen, what's my favorite, etc.)
- Query might benefit from past context (even if not explicitly asked for)

Memory should NOT be used when:
- User is only trying to record new information ("help me remember", "记一下", "记住", etc.)
- Making new notes or annotations
- Setting new preferences or information

Image is required for answer when:
- Query needs to extract information from the current image
- Query asks about visual details or content in the image
- Query involves recording or remembering information from the image
- Query asks to describe, analyze, or understand anything in the image

Image is required for memory search when:
- Query refers to a specific object ("this pen", "this book", etc.)
- Query needs visual details to identify the correct memory
- Query combines temporal aspects with visual objects
- Query needs to match the current object with past observations

Examples:
"What did I eat yesterday?"
-> Memory required: Yes
-> Image required for memory: No
-> Image required for answer: No
-> Query: "food items consumed yesterday"

"When did I buy this pen?"
-> Memory required: Yes
-> Image required for memory: Yes
-> Image required for answer: Yes
-> Reason: Need image details to search memories about this specific pen

"Have I seen anything like this?"
-> Memory required: Yes
-> Image required for memory: Yes
-> Image required for answer: Yes
-> Reason: Need to compare current image with past observations

"What is the capital of France?"
-> Memory required: No
-> Image required for memory: N/A
-> Image required for answer: No
-> Direct answer: The capital of France is Paris.
-> Reason: Simple factual question that doesn't need memory or image

"Describe the image"
-> Memory required: No
-> Image required for memory: N/A
-> Image required for answer: Yes
-> Reason: Need to describe the image without using past context

"记住这个名片上的所有信息。" (Remember all information on this business card)
-> Memory required: No
-> Image required for memory: N/A
-> Image required for answer: Yes
-> Reason: Need to extract information from the business card image to record it

Format your response as:
MEMORY_REQUIRED: YES/NO
IMAGE_REQUIRED_MEMORY: YES/NO/N/A
IMAGE_REQUIRED_ANSWER: YES/NO
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
        image_required_memory = False
        image_required_answer = False
        direct_query = None
        direct_answer = None
        reason = None

        for line in lines:
            if line.startswith('MEMORY_REQUIRED:'):
                memory_required = 'YES' in line
            elif line.startswith('IMAGE_REQUIRED_MEMORY:'):
                image_required_memory = 'YES' in line
            elif line.startswith('IMAGE_REQUIRED_ANSWER:'):
                image_required_answer = 'YES' in line
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

        if direct_answer and direct_answer != "N/A" and (not memory_required) and (not image_required_answer):
            final_decision = "output_formatter"
            self.stm(self.workflow_instance_id)["final_answer"] = direct_answer
        elif memory_required and image_required_memory:
            final_decision = "multimodal_query_generator"
        elif memory_required:
            final_decision = "memory_search"
        elif image_required_answer:
            final_decision = "answer_generator"
        else:
            # remove the image in stm and run answer_generator
            self.stm(self.workflow_instance_id)["image_cache"] = None
            final_decision = "answer_generator"

        return {
            "memory_required": memory_required,
            "image_required_memory": image_required_memory and input_has_image,
            "image_required_answer": image_required_answer and input_has_image,
            "memory_search_query": direct_query if direct_query != "N/A" else None,
            "no_memory_required": not memory_required,
            "skip_everything": not memory_required and not image_required_answer,
            "final_answer": direct_answer if not memory_required and not image_required_answer else None,
            "reason": reason,
            "final_decision": final_decision
        }