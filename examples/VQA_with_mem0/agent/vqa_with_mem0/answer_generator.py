from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Content, Message
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from time import time
from openai import Stream
from datetime import datetime

THRESHOLD = 0.5

ANSWER_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手，专门用于回答与图像相关的问题。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

USER_PROMPT = """
请根据提供的图片和/或记忆回答以下问题。重要指引：

1. 关于记忆：只使用与当前问题直接相关的信息，忽略任何不相关的记忆。

2. 识别问题类型：
   a. 对于关于过去事件/购买的问题（例如"我什么时候买过这个？"，"这个东西是什么时候买的？"）：
      - 查找包含日期/时间的相关记忆
      - 如果没有找到相关记忆：
        - 回答"抱歉，我没有找到相关的购买记录"或"我的记忆中没有这个购买信息"
   
   b. 对于记忆存储请求：
      - 如果提供了图片并要求记住（如"记一下图片内容"或"记住我吃了这个药"）：
        - 首先简洁描述图片的主要内容
        - 以确认结束：
          - 回答"好的，我记住了"或"明白了"
      
      - 如果没有图片但要求记住文字/信息：
        - 简单回答"好的，记住了"或"明白了"

3. 保持回答简洁：
   - 回答最多50个汉字
   - 对于图片记忆请求：先简要描述图片内容，然后确认
   - 对于文字记忆请求：只需简短确认

4. 始终使用中文回答

5. 考虑当前上下文 - 日期/时间：{datetime}

相关记忆：
{memory_context}

问题：{user_instruction}"""

@registry.register_worker()
class VQAAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Generates answers for visual questions using image and memory context"""

    llm: OpenaiGPTLLM

    def _generate_answer(self, user_instruction: str, memory_context: str, image_cache: dict) -> tuple:
        # Get current datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # add concise requirement and memory awareness to the prompt
        user_instruction = USER_PROMPT.format(user_instruction=user_instruction, memory_context=memory_context, datetime=current_datetime)

        answer_messages = [
            Message(role="system", message_type="text", content=ANSWER_PROMPT.format(datetime=current_datetime)),
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
            "answer_messages": answer_messages
        }
