from datetime import datetime
from time import time
from typing import Dict, Any, List, Optional, Tuple
from omagent_core.utils.general import encode_image
from .prompts import (
    ANSWER_PROMPT, GENERAL_PROMPT, GENERAL_PROMPT_WITHOUT_MEMORY,
    MEMORY_STORE_PROMPT, TEXT_SYSTEM_PROMPT, TEXT_GENERAL_PROMPT,
    TEXT_MEMORY_CONTEXT_SECTION, TEXT_CONVERSATION_HISTORY_SECTION,
    TEXT_MEMORY_STORE_PROMPT, THRESHOLD
)
from .utils import format_conversation_history

class AnswerGenerator:
    """Generates answers based on user queries, images, and memory context"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.prompts = {}
    
    def generate_answer(self, 
                        user_instruction: str, 
                        memory_context: str, 
                        image_cache: Optional[Dict[str, Any]], 
                        conversation_history: List[Dict[str, str]],
                        is_store_request: bool = False,
                        config_name: Optional[str] = None) -> Tuple[str, float, str]:
        """Generate answer using image and memory context
        
        Args:
            user_instruction: User's query
            memory_context: Context from memory search
            image_cache: Image data if available
            conversation_history: Previous conversation history
            is_store_request: Whether this is a memory store request
            config_name: Name of LLM config to use
            
        Returns:
            Tuple of (answer, llm_time, model_used)
        """
        # Get LLM client
        llm_info = self.llm_manager.get_llm_client(config_name, "answer_generator")
        client = llm_info["client"]
        model_id = llm_info["model_id"]
        temperature = llm_info["temperature"]
        
        # Get current datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format conversation history
        formatted_history = format_conversation_history(conversation_history)
        
        # Use prompts from self.prompts if available, otherwise use the imported ones
        answer_prompt = self.prompts.get("ANSWER_PROMPT", ANSWER_PROMPT)
        general_prompt = self.prompts.get("GENERAL_PROMPT", GENERAL_PROMPT)
        general_prompt_without_memory = self.prompts.get("GENERAL_PROMPT_WITHOUT_MEMORY", GENERAL_PROMPT_WITHOUT_MEMORY)
        memory_store_prompt = self.prompts.get("MEMORY_STORE_PROMPT", MEMORY_STORE_PROMPT)
        
        # Select appropriate prompt based on request type and memory context
        if is_store_request:
            prompt = memory_store_prompt.format(
                user_instruction=user_instruction,
                datetime=current_datetime,
                conversation_history=formatted_history
            )
        elif len(memory_context) == 0:    
            prompt = general_prompt_without_memory.format(
                user_instruction=user_instruction,
                conversation_history=formatted_history
            )
        else:
            prompt = general_prompt.format(
                user_instruction=user_instruction,
                memory_context=memory_context,
                datetime=current_datetime,
                conversation_history=formatted_history
            )

        messages = [
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        # Add image if available
        if image_cache:
            img = image_cache["<image_0>"]
            # Add image to the user message content
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(img)}"
                }
            })

        # Make API call
        start_time = time()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        llm_time = time() - start_time

        answer = response.choices[0].message.content
        
        return answer, llm_time, model_id
    
    def generate_text_answer(self, 
                            user_instruction: str, 
                            conversation_history: List[Dict[str, str]],
                            memory_search_results: Optional[Dict[str, Any]] = None,
                            is_store_request: bool = False,
                            config_name: Optional[str] = None) -> Tuple[str, float, str]:
        """Generate text-only answer
        
        Args:
            user_instruction: User's query
            conversation_history: Previous conversation history
            memory_search_results: Results from memory search if available
            is_store_request: Whether this is a memory store request
            config_name: Name of LLM config to use
            
        Returns:
            Tuple of (answer, llm_time, model_used)
        """
        # Get LLM client
        llm_info = self.llm_manager.get_llm_client(config_name, "text_answer")
        client = llm_info["client"]
        model_id = llm_info["model_id"]
        temperature = llm_info["temperature"]
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format conversation history
        formatted_history = format_conversation_history(conversation_history)
        
        # Use prompts from self.prompts if available, otherwise use the imported ones
        text_system_prompt = self.prompts.get("TEXT_SYSTEM_PROMPT", TEXT_SYSTEM_PROMPT)
        text_general_prompt = self.prompts.get("TEXT_GENERAL_PROMPT", TEXT_GENERAL_PROMPT)
        text_memory_context_section = self.prompts.get("TEXT_MEMORY_CONTEXT_SECTION", TEXT_MEMORY_CONTEXT_SECTION)
        text_conversation_history_section = self.prompts.get("TEXT_CONVERSATION_HISTORY_SECTION", TEXT_CONVERSATION_HISTORY_SECTION)
        text_memory_store_prompt = self.prompts.get("TEXT_MEMORY_STORE_PROMPT", TEXT_MEMORY_STORE_PROMPT)
        
        # Select appropriate prompt based on request type
        if is_store_request:
            formatted_instruction = text_memory_store_prompt.format(
                user_instruction=user_instruction,
                conversation_history=formatted_history
            )
        else:
            memory_section = ""
            datetime_section = ""
            conversation_section = text_conversation_history_section.format(conversation_history=formatted_history)
            
            # Add memory context if available
            if memory_search_results:
                relevant_memories = memory_search_results.get("relevant_memories", None)
                if relevant_memories:
                    filtered_memories = [mem for mem in relevant_memories if mem.get("score", 0) >= THRESHOLD]
                    if len(filtered_memories) == 0:
                        print(f"Warning: filtered_memories is empty after threshold {THRESHOLD}")
                    if filtered_memories:
                        memory_context = "\n".join(
                            [f"- {mem.get('memory', '')}" for mem in filtered_memories]
                        )
                        memory_section = text_memory_context_section.format(memory_context=memory_context)
                        datetime_section = "3. 考虑当前上下文 - 日期/时间：{datetime}\n".format(datetime=current_datetime)
            
            formatted_instruction = text_general_prompt.format(
                user_instruction=user_instruction,
                memory_section=memory_section,
                datetime_section=datetime_section,
                conversation_section=conversation_section
            )

        messages = [
            {"role": "system", "content": text_system_prompt},
            {"role": "user", "content": formatted_instruction}
        ]

        # Make API call
        start_time = time()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        llm_time = time() - start_time

        answer = response.choices[0].message.content
        
        return answer, llm_time, model_id

    def update_prompts(self, prompts):
        """Update the prompts used by the answer generator"""
        self.prompts = prompts
        return self.prompts 