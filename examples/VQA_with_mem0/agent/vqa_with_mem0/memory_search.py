from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image

MEMORY_SEARCH_PROMPT = """You are an AI assistant that helps with visual question answering while maintaining a memory of past interactions.

For the given user query, you should decide if searching memory would be helpful.

You should search memory when:
1. User asks about past events or interactions
2. User refers to previous conversations
3. User asks about things they did before
4. User asks "what did I..." type questions
5. Questions about history or past purchases

Examples:
- "What did I buy today?" -> SEARCH: YES
- "What is in this image?" -> SEARCH: NO
- "Did I talk about this before?" -> SEARCH: YES
- "Can you describe this picture?" -> SEARCH: NO

Format your response as:
SEARCH_MEMORY: YES/NO
SEARCH_QUERY: <query if searching>"""

@registry.register_worker()
class VQAMemorySearch(BaseWorker, BaseLLMBackend):
    """First step of VQA process - decides if memory search is needed"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        search_messages = [
            Message(role="system", message_type="text", content=MEMORY_SEARCH_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
        ]

        # Add image if available
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

        # Parse search decision
        lines = search_response.split('\n')
        search_memory = False
        search_query = None

        for line in lines:
            if line.startswith('SEARCH_MEMORY:'):
                search_memory = 'YES' in line
            elif line.startswith('SEARCH_QUERY:'):
                search_query = line.split(':', 1)[1].strip()

        # Store results in STM for next step
        self.stm(self.workflow_instance_id)["memory_search"] = {
            "search_needed": search_memory,
            "search_query": search_query
        }

        # self.callback.send_answer(self.workflow_instance_id, msg=f"Memory search needed: {search_memory}\nSearch query: {search_query}")

        return {"search_needed": search_memory, "search_query": search_query} 