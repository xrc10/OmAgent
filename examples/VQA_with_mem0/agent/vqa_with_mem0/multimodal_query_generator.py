from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from time import time

MULTIMODAL_QUERY_PROMPT = """You are an AI assistant that helps generate memory search queries based on both text and image input.

Given the user's question and the image, generate a clear and specific query to search in memory.

Examples:
Q: "Did I eat this before?" (with image of pizza)
-> Query: "previous instances of eating pizza"

Q: "Have I been to this place?" (with image of park)
-> Query: "visits to this park with green benches and fountain"

Format your response as:
SEARCH_QUERY: <your specific search query>"""

@registry.register_worker()
class MultimodalQueryGenerator(BaseWorker, BaseLLMBackend):
    """Generates memory search queries using both text and image input"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        query_messages = [
            Message(role="system", message_type="text", content=MULTIMODAL_QUERY_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
        ]

        # Add image from cache
        if self.stm(self.workflow_instance_id).get("image_cache", None):
            img = self.stm(self.workflow_instance_id)["image_cache"]["<image_0>"]
            query_messages.append(
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
        response = self.llm.generate(records=query_messages)
        llm_time = time() - start_time
        query_response = response["choices"][0]["message"]["content"]

        # Parse query
        search_query = None
        for line in query_response.split('\n'):
            if line.startswith('SEARCH_QUERY:'):
                search_query = line.split(':', 1)[1].strip()
                break

        # store the query in the STM if exists
        if search_query:
            self.stm(self.workflow_instance_id)["memory_search_query"] = search_query
        else:
            # use the user instruction as the query
            search_query = user_instruction
            self.stm(self.workflow_instance_id)["memory_search_query"] = search_query

        return {
            "memory_search_query": search_query,
            "llm_time": llm_time
        }