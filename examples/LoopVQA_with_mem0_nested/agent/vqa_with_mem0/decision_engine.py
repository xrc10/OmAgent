from typing import Dict, Any, List
from .utils import is_chinese, contains_keywords

class DecisionEngine:
    """Handles decision making for the VQA workflow"""
    
    def __init__(self):
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
    
    def memory_decision(self, user_instruction: str, stm: Dict[str, Any]) -> Dict[str, Any]:
        """Decide if memory search is needed and if image is required
        
        Args:
            user_instruction: The user's query
            stm: Short-term memory dictionary
            
        Returns:
            Dictionary with decision results
        """
        # Determine language
        lang_keywords = 'chinese' if is_chinese(user_instruction) else 'english'
        
        # Check for memory keywords
        memory_required = contains_keywords(
            user_instruction, 
            self.memory_keywords[lang_keywords]
        )
        
        # Always use image if available
        image_required = True
        
        # Check for store keywords
        is_store_request = contains_keywords(
            user_instruction, 
            self.store_keywords[lang_keywords]
        )

        input_has_image = stm.get("image_cache", None) is not None

        # Decision logic
        if is_store_request:
            final_decision = "answer_generator" if (input_has_image and image_required) else "text_answer_generator"
            memory_required = False
            image_required_memory = False
            image_required_answer = input_has_image and image_required
        elif memory_required:
            if input_has_image and image_required:
                final_decision = "multimodal_query_generator"
                image_required_memory = True
                image_required_answer = True
            else:
                final_decision = "memory_search"
                image_required_memory = False
                image_required_answer = False
            stm["memory_search_query"] = user_instruction.replace("想一下", "").replace("想下", "")
        else:
            if input_has_image and image_required:
                final_decision = "answer_generator"
                memory_required = False
                image_required_memory = False
                image_required_answer = True
            else:
                final_decision = "text_answer_generator"
                memory_required = False
                image_required_memory = False
                image_required_answer = False

        # Return decision results
        return {
            "memory_required": memory_required,
            "image_required_memory": image_required_memory,
            "image_required_answer": image_required_answer,
            "is_store_request": is_store_request,
            "final_decision": final_decision,
        } 