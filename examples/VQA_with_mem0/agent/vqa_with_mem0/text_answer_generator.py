from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry
import time
from datetime import datetime

THRESHOLD = 0.5

SYSTEM_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

GENERAL_PROMPT = """
请回答以下问题。重要指引：

1. 保持回答简洁：
   - 回答最多50个汉字

2. 始终使用中文回答

3. 考虑当前上下文 - 日期/时间：{datetime}

{memory_section}

问题：{user_instruction}"""

MEMORY_CONTEXT_SECTION = """
4. 关于记忆：
   - 只使用与当前问题直接相关的信息
   - 对于关于过去事件/购买的问题（例如"我什么时候买过这个？"）：
     - 如果没有找到相关记忆，回答"抱歉，我没有找到相关的记录"

相关记忆：
{memory_context}
"""

MEMORY_STORE_PROMPT = """
这是一个记忆存储请求。请按以下方式回应：

1. 保持回答简洁：
   - 首先用"好的，记住了"或"明白了"等简短话语确认
   - 然后复述需要记忆的信息
   
2. 始终使用中文回答

要存储的内容：{user_instruction}"""

@registry.register_worker()
class TextAnswerGenerator(BaseWorker, BaseLLMBackend):
    """Generates text-only answers for questions that don't require images or memory"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this is a memory store request
        is_store_request = self.stm(self.workflow_instance_id).get("is_store_request", False)
        
        if is_store_request:
            formatted_instruction = MEMORY_STORE_PROMPT.format(
                user_instruction=user_instruction
            )
        else:
            memory_search_results = self.stm(self.workflow_instance_id).get("memory_search_results", {})
            relevant_memories = memory_search_results.get("relevant_memories", None)

            memory_section = ""
            if relevant_memories:
                filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= THRESHOLD]
                if filtered_memories:
                    memory_context = "\n".join(
                        [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                    )
                    memory_section = MEMORY_CONTEXT_SECTION.format(memory_context=memory_context)
            
            formatted_instruction = GENERAL_PROMPT.format(
                user_instruction=user_instruction,
                memory_section=memory_section,
                datetime=current_datetime
            )

        messages = [
            Message(role="system", message_type="text", content=SYSTEM_PROMPT),
            Message(role="user", message_type="text", content=formatted_instruction)
        ]

        start_time = time.time()
        response = self.llm.generate(records=messages)
        llm_time = time.time() - start_time

        answer = response["choices"][0]["message"]["content"]
        
        # Store answer in STM for memory store worker
        self.stm(self.workflow_instance_id)["answer"] = answer
        
        # Send answer to user
        self.callback.send_answer(self.workflow_instance_id, msg=answer)

        return {
            "answer": answer,
            "llm_time": llm_time
        } 