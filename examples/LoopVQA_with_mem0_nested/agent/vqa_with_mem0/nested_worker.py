from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.utils.general import encode_image
from datetime import datetime
from time import time
import re
import os
import yaml
from pathlib import Path
import openai
from openai import OpenAI
import asyncio
import threading

# Import memory manager
from .memory_manager import MemoryManager
from .llm_manager import LLMManager
from .decision_engine import DecisionEngine
from .answer_generator import AnswerGenerator
from .query_generator import QueryGenerator
from .utils import extract_model_config
from .prompts import THRESHOLD, load_prompt

# Constants for memory decision
THRESHOLD = 0.40

# Prompts for different components
ANSWER_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手，专门用于回答与图像相关的问题。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

GENERAL_PROMPT = """
请回答问题，并参考提供的相关记忆和对话历史。重要指引：

1. 保持回答简洁，最多50个汉字
2. 如果问题涉及过去的事件，且没有找到相关记忆，请回答"抱歉，我没有找到相关的记录"
3. 始终使用中文回答

相关记忆：
{memory_context}

对话历史：
{conversation_history}

当前时间：{datetime}

问题：{user_instruction}"""

GENERAL_PROMPT_WITHOUT_MEMORY = """
请回答问题，并参考提供的对话历史。始终使用中文回答。

对话历史：
{conversation_history}

问题：{user_instruction}"""

MEMORY_STORE_PROMPT = """请根据图片内容创建一条简短的记忆记录。要求：

1. 结合图片，用20字以内简洁描述需要记忆的内容
2. 如果存在相对时间，例如"昨天"，请参考当前时间：{datetime}

对话历史：
{conversation_history}

记忆请求：{user_instruction}

请按以下格式回复：
[复述记忆内容]
好的，我已经记住了。"""

MULTIMODAL_QUERY_PROMPT = """Given the user's question and the image, first briefly describe the key details of the image, then generate a clear and specific query to search in memory. If the user's question is in Chinese, respond in Chinese.

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

TEXT_SYSTEM_PROMPT = """你是小欧，一个由 Om AI 创建的 AI 助手。请始终基于可用信息提供有帮助、准确和简洁的回答。"""

TEXT_GENERAL_PROMPT = """
请回答以下问题。重要指引：

1. 始终使用中文回答

{datetime_section}{memory_section}{conversation_section}

问题：{user_instruction}"""

TEXT_MEMORY_CONTEXT_SECTION = """
2. 保持回答简洁：
   - 回答最多50个汉字

3. 关于记忆：
   - 只使用与当前问题直接相关的信息
   - 对于关于过去事件/购买的问题（例如"我什么时候买过这个？"）：
     - 如果没有找到相关记忆，回答"抱歉，我没有找到相关的记录"

相关记忆：
{memory_context}
"""

TEXT_CONVERSATION_HISTORY_SECTION = """
对话历史：
{conversation_history}
"""

TEXT_MEMORY_STORE_PROMPT = """
这是一个记忆存储请求。请按以下方式回应：

1. 保持回答简洁：
   - 首先用"好的，记住了"或"明白了"等简短话语确认
   - 然后复述需要记忆的信息
   
2. 始终使用中文回答

对话历史：
{conversation_history}

要存储的内容：{user_instruction}"""

@registry.register_worker()
class NestedWorker(BaseWorker):
    """Consolidated worker that handles the entire VQA with memory workflow"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_keywords = {
            'chinese': ['想一下', '想下'],
            'english': ['think about', 'recall']
        }
        self.image_keywords = {
            'chinese': ['看一下', '看下'],
            'english': ['look at', 'check']
        }
        self.store_keywords = {
            'chinese': ['记一下', '记下'],
            'english': ['note down', 'remember']
        }
        
        # Initialize components
        self.llm_manager = LLMManager()
        self.decision_engine = DecisionEngine()
        self.answer_generator = AnswerGenerator(self.llm_manager)
        self.query_generator = QueryGenerator(self.llm_manager)

    def _load_prompts(self):
        """Load all prompts from text files to ensure we have the latest versions"""
        from .prompts import (
            load_prompt, ANSWER_PROMPT, GENERAL_PROMPT, GENERAL_PROMPT_WITHOUT_MEMORY,
            MEMORY_STORE_PROMPT, MULTIMODAL_QUERY_PROMPT, TEXT_SYSTEM_PROMPT,
            TEXT_GENERAL_PROMPT, TEXT_MEMORY_CONTEXT_SECTION,
            TEXT_CONVERSATION_HISTORY_SECTION, TEXT_MEMORY_STORE_PROMPT
        )
        
        # Reload all prompts
        self.prompts = {
            "ANSWER_PROMPT": ANSWER_PROMPT,
            "GENERAL_PROMPT": GENERAL_PROMPT,
            "GENERAL_PROMPT_WITHOUT_MEMORY": GENERAL_PROMPT_WITHOUT_MEMORY,
            "MEMORY_STORE_PROMPT": MEMORY_STORE_PROMPT,
            "MULTIMODAL_QUERY_PROMPT": MULTIMODAL_QUERY_PROMPT,
            "TEXT_SYSTEM_PROMPT": TEXT_SYSTEM_PROMPT,
            "TEXT_GENERAL_PROMPT": TEXT_GENERAL_PROMPT,
            "TEXT_MEMORY_CONTEXT_SECTION": TEXT_MEMORY_CONTEXT_SECTION,
            "TEXT_CONVERSATION_HISTORY_SECTION": TEXT_CONVERSATION_HISTORY_SECTION,
            "TEXT_MEMORY_STORE_PROMPT": TEXT_MEMORY_STORE_PROMPT
        }
        
        # Update the components with the latest prompts
        self.answer_generator.update_prompts(self.prompts)
        self.query_generator.update_prompts(self.prompts)
        
        return self.prompts

    def _check_exit(self, user_instruction):
        """Check if user wants to exit the conversation"""
        # Check if user wants to exit
        should_exit = "退出" in user_instruction.strip().lower() or "exit" in user_instruction.strip().lower()
        
        # Store conversation history for context
        conversation_history = self.stm(self.workflow_instance_id).get("conversation_history", [])
        
        # Get the current Q&A pair
        current_question = user_instruction
        current_answer = self.stm(self.workflow_instance_id).get("answer", "")
        
        # Add current Q&A to history if not exiting
        if not should_exit and current_question and current_answer:
            conversation_history.append({
                "question": current_question,
                "answer": current_answer
            })
            
            # Keep only the last 5 conversations for context
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
            
            # Update conversation history in STM
            self.stm(self.workflow_instance_id)["conversation_history"] = conversation_history
        
        return {"should_exit": should_exit}

    def _memory_search(self, user_instruction):
        """Search for relevant memories"""
        # Start timing
        start_time = time()
        
        user_id = self.stm(self.workflow_instance_id).get("user_id", "default_user")
        memory_manager = MemoryManager(user_id=user_id)

        memory_search_query = self.stm(self.workflow_instance_id).get("memory_search_query", None)

        # if no search query, use the user instruction as the search query
        if not memory_search_query:
            memory_search_query = user_instruction

        # Directly use the user instruction as the search query
        relevant_memories = memory_manager.search_memory(memory_search_query)
        
        # Keep only the top 5 memories
        relevant_memories = relevant_memories[:5] if relevant_memories else []
        
        # Calculate elapsed time
        search_time = time() - start_time

        print(f"Found {len(relevant_memories)} memories in {search_time:.2f} seconds")
        
        # Store results in STM for next step
        results = {
            "search_success": bool(relevant_memories),
            "search_query": memory_search_query,
            "relevant_memories": relevant_memories,
            "search_time": search_time
        }
        
        self.stm(self.workflow_instance_id)["memory_search_results"] = results
        return results

    def _memory_store(self, user_instruction, answer):
        """Store memory if needed"""
        if not self.stm(self.workflow_instance_id).get("is_store_request", False):
            return {"store_memory": False, "memory_content": None}

        user_id = self.stm(self.workflow_instance_id).get("user_id", "default_user")

        # add user_instruction to memory_content
        memory_content = "User query: " + user_instruction + "\n" + "Answer: " + answer

        memory_manager = MemoryManager(user_id=user_id)
        memory_manager.add_memory(
            memory_content,
            metadata={
                "type": "vqa_interaction",
                "user_id": user_id
            }
        )

        return {
            "store_memory": True,
            "memory_content": memory_content,
            "answer": answer
        }
        
    def _async_memory_store(self, user_instruction, answer):
        """Run memory store in a background thread"""
        def _run_store():
            self._memory_store(user_instruction, answer)
            
        # Start a new thread to handle the memory storage
        thread = threading.Thread(target=_run_store)
        thread.daemon = True  # Allow the thread to exit when main program exits
        thread.start()
        
        return {
            "store_memory": True,
            "memory_content": "User query: " + user_instruction + "\n" + "Answer: " + answer,
            "answer": answer
        }

    def _run(self, *args, **kwargs):
        # Load the latest prompts
        self._load_prompts()
        
        # Get user instruction from STM
        user_instruction = self.stm(self.workflow_instance_id).get("user_instruction", "")
        
        # Check if user wants to exit
        exit_result = self._check_exit(user_instruction)
        if exit_result["should_exit"]:
            self.callback.send_answer(self.workflow_instance_id, msg="再见！")
            return {"should_exit": True}
        
        # Check for model selection in user instruction
        model_config, user_instruction = extract_model_config(user_instruction, self.llm_manager.llm_configs)
        if model_config:
            self.stm(self.workflow_instance_id)["user_instruction"] = user_instruction
        
        # Step 1: Memory decision
        decision_result = self.decision_engine.memory_decision(
            user_instruction, 
            self.stm(self.workflow_instance_id)
        )
        
        # Save decision results to STM
        for key, value in decision_result.items():
            self.stm(self.workflow_instance_id)[key] = value
            
        final_decision = decision_result["final_decision"]
        print("Final decision:", final_decision)
        
        # Step 2: Process based on decision
        if final_decision == "multimodal_query_generator":
            # Generate query using image and text
            query_result = self.query_generator.generate_multimodal_query(
                user_instruction,
                self.stm(self.workflow_instance_id).get("image_cache", None),
                model_config
            )
            
            # Save query results to STM
            self.stm(self.workflow_instance_id)["memory_search_query"] = query_result["memory_search_query"]
            if "image_description" in query_result:
                self.stm(self.workflow_instance_id)["image_description"] = query_result["image_description"]
            
            # Search memory with generated query
            memory_result = self._memory_search(query_result["memory_search_query"])
            
            # Generate answer with image and memory context
            memory_search_results = self.stm(self.workflow_instance_id).get("memory_search_results", {})
            relevant_memories = memory_search_results.get("relevant_memories", None)
            
            memory_context = ""
            if relevant_memories:
                filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= THRESHOLD]
                if filtered_memories:
                    memory_context = "\n".join(
                        [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                    )
                if len(filtered_memories) == 0:
                    memory_context = "没有找到相关的记录"
            
            answer, llm_time, model_used = self.answer_generator.generate_answer(
                user_instruction, 
                memory_context,
                self.stm(self.workflow_instance_id).get("image_cache", None),
                self.stm(self.workflow_instance_id).get("conversation_history", []),
                self.stm(self.workflow_instance_id).get("is_store_request", False),
                model_config
            )
            
        elif final_decision == "memory_search":
            # Search memory directly
            memory_result = self._memory_search(user_instruction)
            # Generate text answer with memory context
            answer, llm_time, model_used = self.answer_generator.generate_text_answer(
                user_instruction,
                self.stm(self.workflow_instance_id).get("conversation_history", []),
                self.stm(self.workflow_instance_id).get("memory_search_results", {}),
                self.stm(self.workflow_instance_id).get("is_store_request", False),
                model_config
            )
            
        elif final_decision == "answer_generator":
            # Generate answer with image but no memory
            answer, llm_time, model_used = self.answer_generator.generate_answer(
                user_instruction, 
                "",
                self.stm(self.workflow_instance_id).get("image_cache", None),
                self.stm(self.workflow_instance_id).get("conversation_history", []),
                self.stm(self.workflow_instance_id).get("is_store_request", False),
                model_config
            )
            
        elif final_decision == "text_answer_generator":
            # Generate text-only answer
            answer, llm_time, model_used = self.answer_generator.generate_text_answer(
                user_instruction,
                self.stm(self.workflow_instance_id).get("conversation_history", []),
                self.stm(self.workflow_instance_id).get("memory_search_results", {}),
                self.stm(self.workflow_instance_id).get("is_store_request", False),
                model_config
            )
        
        # Store answer in STM
        self.stm(self.workflow_instance_id)["answer"] = answer
        self.stm(self.workflow_instance_id)["final_answer"] = answer
        self.stm(self.workflow_instance_id)["model_used"] = model_used
        
        # Step 3: Store memory if needed
        if self.stm(self.workflow_instance_id).get("is_store_request", False):
            # Run memory store asynchronously in the background
            store_result = self._async_memory_store(user_instruction, answer)
        
        # Send answer to user
        model_info = f"\n[使用模型: {model_used}]" if model_used else ""
        self.callback.send_answer(self.workflow_instance_id, msg=f"{answer}{model_info}")
        
        return {
            "answer": answer,
            "final_decision": final_decision,
            "should_exit": False,
            "model_used": model_used
        }
