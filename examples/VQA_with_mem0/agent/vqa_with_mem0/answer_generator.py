from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from time import time
from openai import Stream
from datetime import datetime

THRESHOLD = 0.45

ANSWER_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手，专门用于回答与图像相关的问题。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

GENERAL_PROMPT = """
请根据提供的图片和相关记忆回答问题。重要指引：

1. 保持回答简洁，最多50个汉字
2. 如果问题涉及过去的事件/购买，且没有找到相关记忆，请回答"抱歉，我没有找到相关的记录"
3. 始终使用中文回答

相关记忆：
{memory_context}

当前时间：{datetime}

问题：{user_instruction}"""

GENERAL_PROMPT_WITHOUT_MEMORY = """
请根据提供的图片回答问题。重要指引：

1. 保持回答简洁，最多50个汉字
2. 始终使用中文回答

问题：{user_instruction}"""

MEMORY_STORE_PROMPT = """请根据图片内容创建一条简短的记忆记录。要求：

1. 用20字以内简洁描述图片的主要内容
2. 记录时间：{datetime}

记忆请求：{user_instruction}

请按以下格式回复：
[图片的内容]
好的，我已经记住了。"""

@registry.register_worker()
class VQAAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Generates answers for visual questions using image and memory context"""

    llm: OpenaiGPTLLM

    def _generate_answer(self, user_instruction: str, memory_context: str, image_cache: dict) -> tuple:
        # Get current datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this is a memory store request
        is_store_request = self.stm(self.workflow_instance_id).get("is_store_request", False)
        
        if is_store_request:
            user_instruction = MEMORY_STORE_PROMPT.format(
                user_instruction=user_instruction,
                datetime=current_datetime
            )
        elif len(memory_context) == 0:    
            user_instruction = GENERAL_PROMPT_WITHOUT_MEMORY.format(
                user_instruction=user_instruction
            )
        else:
            user_instruction = GENERAL_PROMPT.format(
                user_instruction=user_instruction,
                memory_context=memory_context,
                datetime=current_datetime
            )

        answer_messages = [
            Message(role="system", message_type="text", content=ANSWER_PROMPT),
            Message(role="user", message_type="text", content=user_instruction)
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
                                # "url": self.stm(self.workflow_instance_id)["image_url"]
                            },
                        )
                    ],
                ),
            )

        start_time = time()
        self.llm.stream = False
        response = self.llm.generate(records=answer_messages)
        llm_time = time() - start_time

        # Handle streaming response
        if isinstance(response, Stream):
            self.callback.send_answer(self.workflow_instance_id, msg="Streaming...")
            answer = ""
            self.callback.send_incomplete(self.workflow_instance_id, msg="")
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    self.callback.send_incomplete(self.workflow_instance_id, msg=content)
                    answer += content
                else:
                    self.callback.send_block(self.workflow_instance_id, msg="")
                    break
        else:
            answer = response["choices"][0]["message"]["content"]

        # Remove image messages before returning
        answer_messages = [msg for msg in answer_messages if msg.message_type != "image"]
        
        return answer, llm_time, answer_messages

    def _run(self, user_instruction: str, *args, **kwargs):
        memory_search_results = self.stm(self.workflow_instance_id).get("memory_search_results", {})
        relevant_memories = memory_search_results.get("relevant_memories", None)

        memory_context = ""
        if relevant_memories:
            filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= THRESHOLD]
            if filtered_memories:
                memory_context = "\n".join(
                    [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                )
            # if len(filtered_memories) == 0:
            #     self.callback.send_answer(self.workflow_instance_id, msg="抱歉未找到相关记忆")

        # Generate answer
        answer, llm_time, answer_messages = self._generate_answer(
            user_instruction, 
            memory_context,
            self.stm(self.workflow_instance_id).get("image_cache", None)
        )

        # Store answer in STM for memory store worker
        self.stm(self.workflow_instance_id)["answer"] = answer

        # Send answer only if not already streamed
        if not self.llm.stream:
            self.callback.send_answer(self.workflow_instance_id, msg=f"{answer}")

        return {
            "answer": answer,
            "relevant_memories": relevant_memories,
            "llm_time": llm_time,
            "memory_context": memory_context,
            "answer_messages": answer_messages
        }
