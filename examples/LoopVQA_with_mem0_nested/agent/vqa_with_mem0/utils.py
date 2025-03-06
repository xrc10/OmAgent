import re
from typing import Dict, List, Any, Optional
from time import time

def is_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    return bool(re.search('[\u4e00-\u9fff]', text))

def contains_keywords(text: str, keyword_list: List[str]) -> bool:
    """Check if text contains any keywords from the list"""
    return any(keyword in text.lower() for keyword in keyword_list)

def format_conversation_history(conversation_history: List[Dict[str, str]]) -> str:
    """Format conversation history for inclusion in prompts"""
    if not conversation_history:
        return "无"
        
    formatted_history = ""
    for i, conv in enumerate(conversation_history):
        formatted_history += f"问: {conv['question']}\n答: {conv['answer']}\n"
        
    return formatted_history

def extract_model_config(user_instruction: str, llm_configs: Dict[str, Any]) -> tuple:
    """Extract model config name from user instruction if present
    
    Returns:
        Tuple of (model_config_name, cleaned_instruction)
    """
    model_config = None
    if "使用模型" in user_instruction:
        for config_name in llm_configs.keys():
            if config_name in user_instruction:
                model_config = config_name
                user_instruction = user_instruction.replace(f"使用模型{config_name}", "").strip()
                break
    
    return model_config, user_instruction

def timed_execution(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        execution_time = time() - start_time
        
        # If result is a tuple, return it with the execution time
        if isinstance(result, tuple):
            return (*result, execution_time)
        # Otherwise, return the result and execution time
        return result, execution_time
    
    return wrapper 