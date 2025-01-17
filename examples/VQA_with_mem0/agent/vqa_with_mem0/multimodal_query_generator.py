from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from time import time

SYSTEM_PROMPT = """You are a helpful AI assistant that generates clear and specific search queries."""

USER_PROMPT = """Given the user's question and the image, first briefly describe the key details of the image, then generate a clear and specific query to search in memory. If the user's question is in Chinese, respond in Chinese.

Examples:
User question: Did I eat this before?
IMAGE: A round pizza with cheese and pepperoni toppings on a wooden serving board.
SEARCH_QUERY: previous instances of eating pepperoni pizza

User question: 这个我以前吃过吗？
IMAGE: 一个10寸的芝士披萨，表面铺满了融化的马苏里拉奶酪。
SEARCH_QUERY: 之前吃芝士披萨的记录

User question: 我什么时候买的这个？
IMAGE: 一个棕色的中号皮包，有金色的金属扣件和长肩带。
SEARCH_QUERY: 购买棕色中号皮包的时间记录

Format your response with:
IMAGE: <brief description of the key details in the image>
SEARCH_QUERY: <your specific search query>

User question: {user_instruction}"""

@registry.register_worker()
class MultimodalQueryGenerator(BaseWorker, BaseLLMBackend):
    """Generates memory search queries using both text and image input"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        query_messages = [
            Message(role="system", message_type="text", content=SYSTEM_PROMPT),
            Message(role="user", message_type="text", content=USER_PROMPT.format(user_instruction=user_instruction))
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
        else:
            raise ValueError("Image cache is None, please check the image_cache in STM")

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

        # store both image description and query in the STM if exists
        if search_query:
            self.stm(self.workflow_instance_id)["memory_search_query"] = search_query
            # Store image description if present
            for line in query_response.split('\n'):
                if line.startswith('IMAGE:'):
                    image_description = line.split(':', 1)[1].strip()
                    self.stm(self.workflow_instance_id)["image_description"] = image_description
                    break
        else:
            # use the user instruction as the query
            search_query = user_instruction
            self.stm(self.workflow_instance_id)["memory_search_query"] = search_query

        return {
            "memory_search_query": search_query,
            "llm_time": llm_time,
            "image_description": image_description,
            "query_response": query_response
        }