from omagent_core.engine.worker.base import BaseWorker
from omagent_core.models.llms.base import BaseLLMBackend
from omagent_core.models.llms.openai_gpt import OpenaiGPTLLM
from omagent_core.models.llms.schemas import Message
from omagent_core.utils.registry import registry
import time

MEMORY_DECISION_PROMPT = """You are 小欧, an AI assistant created by Om AI to help with visual question answering. You should lean towards using memory when there's any possibility of relevant past information. Pay special attention to demonstrative pronouns like 'this', 'that', '这个', '那个' which usually indicate the current image is needed."""

USER_MEMORY_DECISION_PROMPT = """For the given text query, you should:
1. Analyze if searching memory could be helpful (be generous - if there's any chance past information could be relevant, say yes)
2. Determine if the image is required for memory search (if memory is needed)
3. Determine if the image is required to answer the query (regardless of memory, but be conservative; only say 'No' if you are very confident that the image is not needed)
4. If no memory and no image is needed and the query is trivial to answer, provide a direct answer using the same language as the query

First provide a brief analysis, then output your decisions in the following strict format:

Analysis: <brief explanation of the reasoning>
Memory required: Yes/No
Image required for memory: Yes/No/N/A (N/A if memory not required)
Image required for answer: Yes/No
Direct query: <memory search query if memory needed and image not required, otherwise N/A>
Direct answer: <answer if no memory or image needed, otherwise N/A>

Memory should be used when:
- Query asks about past information (what did I do, when did I see, etc.)
- Query mentions or implies past time (today, yesterday, before, again, usually, etc.)
- Query asks about patterns or frequencies (how often, how many times, etc.)
- Query compares current with past (is this different, has this changed, etc.)
- Query refers to user preferences or history (do I like, have I seen, what's my favorite, etc.)
- Query might benefit from past context (even if not explicitly asked for)

Image is REQUIRED for memory search only when Memory is required, and:
- Query contains demonstrative pronouns ("这个", "那个", "this", "that", etc.)
- Query refers to a specific object ("this pen", "this book", "这支笔", etc.)
- Query needs visual details to identify the correct memory
- Query combines temporal aspects with visual objects
- Query needs to match the current object with past observations

Memory should NOT be used when:
- User is trying to record new information (keywords like "记一下", "记住", "remember this", etc.)
- Making new notes or annotations
- Setting new preferences or information
- Asking to describe or analyze the current image only

Image is required for answer when:
- Query needs to extract information from the current image
- Query asks about visual details or content in the image
- Query involves recording or remembering information from the image
- Query asks to describe, analyze, or understand anything in the image
- Query mentions "图片里", "图中", "in the image", etc.

Examples:
"我昨天吃了什么？"
Analysis: 需要查询过去的饮食记录，不需要图片信息
Memory required: Yes
Image required for memory: No
Image required for answer: No
Direct query: 昨天食用的食物
Direct answer: N/A

"我早上9点吃了什么？"
Analysis: 需要查询特定时间点的饮食记录，不需要图片信息
Memory required: Yes
Image required for memory: No
Image required for answer: No
Direct query: 早上9点食用的食物
Direct answer: N/A

"我什么时候买的这支笔？"
Analysis: 包含指示代词"这支"，需要图片来确定具体是哪支笔，然后搜索相关购买记录
Memory required: Yes
Image required for memory: Yes
Image required for answer: Yes
Direct query: N/A
Direct answer: N/A

"这个东西以前见过吗？"
Analysis: 包含指示代词"这个"，需要图片来确定具体物品，并与过去记忆比较
Memory required: Yes
Image required for memory: Yes
Image required for answer: Yes
Direct query: N/A
Direct answer: N/A

"法国的首都是什么？"
Analysis: 简单的事实性问题，不需要记忆或图片
Memory required: No
Image required for memory: N/A
Image required for answer: No
Direct query: N/A
Direct answer: 法国的首都是巴黎。

"描述一下这张图片"
Analysis: 需要描述图片内容，不需要使用过去记忆
Memory required: No
Image required for memory: N/A
Image required for answer: Yes
Direct query: N/A
Direct answer: N/A

"记住这个名片上的所有信息"
Analysis: 需要从名片图片中提取信息来记录
Memory required: No
Image required for memory: N/A
Image required for answer: Yes
Direct query: N/A
Direct answer: N/A

"记一下这个物品"
Analysis: 需要从图片中提取信息来记录，不需要查询历史记忆
Memory required: No
Image required for memory: N/A
Image required for answer: Yes
Direct query: N/A
Direct answer: N/A

Your query is: """

@registry.register_worker()
class MemoryDecisionWorker(BaseWorker, BaseLLMBackend):
    """Initial worker that decides if memory search is needed and if image is required"""

    llm: OpenaiGPTLLM

    def _run(self, user_instruction: str, *args, **kwargs):
        # Make decision based on text only
        decision_messages = [
            Message(role="system", message_type="text", content=MEMORY_DECISION_PROMPT),
            Message(role="user", message_type="text", content=USER_MEMORY_DECISION_PROMPT + user_instruction)
        ]

        # Time the LLM API call
        start_time = time.time()
        response = self.llm.generate(records=decision_messages)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Store the elapsed time in STM for later use
        # self.stm(self.workflow_instance_id)["llm_decision_time"] = elapsed_time
        
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
            if line.startswith('Analysis:'):
                reason = line.split(':', 1)[1].strip()
            elif line.startswith('Memory required:'):
                memory_required = 'yes' in line.lower()
            elif line.startswith('Image required for memory:'):
                image_required_memory = 'yes' in line.lower()
            elif line.startswith('Image required for answer:'):
                image_required_answer = 'yes' in line.lower()
            elif line.startswith('Direct query:'):
                direct_query = line.split(':', 1)[1].strip()
            elif line.startswith('Direct answer:'):
                direct_answer = line.split(':', 1)[1].strip()

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
            final_decision = "answer_generator"

        # # remove the image in stm if not needed
        # if (not image_required_answer) and (not image_required_memory):
        #     self.stm(self.workflow_instance_id)["image_cache"] = None

        return {
            "memory_required": memory_required,
            "image_required_memory": image_required_memory and input_has_image,
            "image_required_answer": image_required_answer and input_has_image,
            "memory_search_query": direct_query if direct_query != "N/A" else None,
            "no_memory_required": not memory_required,
            "skip_everything": not memory_required and not image_required_answer,
            "final_answer": direct_answer if not memory_required and not image_required_answer else None,
            "reason": reason,
            "final_decision": final_decision,
            "llm_time": elapsed_time,
            "decision_response": decision_response
        }