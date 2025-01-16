from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from time import time

ANSWER_PROMPT = """You are an AI assistant that helps with visual question answering."""

@registry.register_worker()
class VQAAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Generates answers for visual questions using image and memory context"""

    llm: OpenaiGPTLLM

    def _generate_answer(self, user_instruction: str, memory_context: str, image_cache: dict) -> tuple:
        # add concise requirement to the prompt
        user_instruction = "Keep your answer within 50 words unless specified otherwise. If response in Chinese, please response within 50 characters. Use the same language as the query.\nNow answer the following question: " + user_instruction

        answer_messages = [
            Message(role="system", message_type="text", content=ANSWER_PROMPT),
            Message(role="user", message_type="text", content=user_instruction + memory_context)
        ]

        if image_cache:
            img = image_cache["<image_0>"]
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

        start_time = time()
        response = self.llm.generate(records=answer_messages)
        llm_time = time() - start_time
        answer = response["choices"][0]["message"]["content"]

        # Remove image messages before returning
        answer_messages = [msg for msg in answer_messages if msg.message_type != "image"]
        
        return answer, llm_time, answer_messages

    def _run(self, user_instruction: str, *args, **kwargs):
        memory_search_results = self.stm(self.workflow_instance_id).get("memory_search_results", {})
        relevant_memories = memory_search_results.get("relevant_memories", None)

        memory_context = ""
        if relevant_memories:
            filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= 0.2]
            if filtered_memories:
                memory_context = "\nRelevant memories from past interactions:\n" + "\n".join(
                    [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                )

        # Generate answer
        answer, llm_time, answer_messages = self._generate_answer(
            user_instruction, 
            memory_context,
            self.stm(self.workflow_instance_id).get("image_cache", None)
        )

        # Store answer in STM for memory store worker
        self.stm(self.workflow_instance_id)["answer"] = answer

        # Send answer immediately
        self.callback.send_answer(self.workflow_instance_id, msg=f"{answer}")

        return {
            "answer": answer,
            "relevant_memories": relevant_memories,
            "llm_time": llm_time,
            "answer_messages": answer_messages
        }
