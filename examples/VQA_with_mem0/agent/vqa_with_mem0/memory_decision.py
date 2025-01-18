from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
import time
import re

@registry.register_worker()
class MemoryDecisionWorker(BaseWorker):
    """Initial worker that decides if memory search is needed and if image is required using keyword matching"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keywords indicating memory search is needed
        self.memory_keywords = {
            'chinese': ['想一下', '想下'],
            'english': ['think about', 'recall']
        }

        # Keywords indicating image is needed
        self.image_keywords = {
            'chinese': ['看一下', '看下'],
            'english': ['look at', 'check']
        }

        # Keywords indicating storing new information
        self.store_keywords = {
            'chinese': ['记一下', '记下'],
            'english': ['note down', 'remember']
        }

    def is_chinese(self, text):
        """Check if text contains Chinese characters"""
        return bool(re.search('[\u4e00-\u9fff]', text))

    def contains_keywords(self, text, keyword_list):
        """Check if text contains any keywords from the list"""
        return any(keyword in text.lower() for keyword in keyword_list)

    def _run(self, user_instruction: str, *args, **kwargs):
        start_time = time.time()
        
        # Determine language
        lang_keywords = 'chinese' if self.is_chinese(user_instruction) else 'english'
        
        # Check for memory keywords
        memory_required = self.contains_keywords(
            user_instruction, 
            self.memory_keywords[lang_keywords]
        )
        
        # Check for image keywords
        image_required = self.contains_keywords(
            user_instruction, 
            self.image_keywords[lang_keywords]
        )
        
        # Check for store keywords
        is_store_request = self.contains_keywords(
            user_instruction, 
            self.store_keywords[lang_keywords]
        )

        input_has_image = self.stm(self.workflow_instance_id).get("image_cache", None) is not None

        # Decision logic
        if is_store_request:
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
        elif memory_required:
            if input_has_image and image_required:
                final_decision = "multimodal_query_generator"
                image_required_memory = True
                image_required_answer = True
            else:
                final_decision = "memory_search"
                image_required_memory = False
                image_required_answer = False
            self.stm(self.workflow_instance_id)["memory_search_query"] = user_instruction
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

        elapsed_time = time.time() - start_time

        # save all results to STM
        self.stm(self.workflow_instance_id)["memory_required"] = memory_required
        self.stm(self.workflow_instance_id)["image_required_memory"] = image_required_memory
        self.stm(self.workflow_instance_id)["image_required_answer"] = image_required_answer
        self.stm(self.workflow_instance_id)["is_store_request"] = is_store_request
        self.stm(self.workflow_instance_id)["final_decision"] = final_decision

        return {
            "memory_required": memory_required,
            "image_required_memory": image_required_memory,
            "image_required_answer": image_required_answer,
            "is_store_request": is_store_request,
            "final_decision": final_decision,
        }