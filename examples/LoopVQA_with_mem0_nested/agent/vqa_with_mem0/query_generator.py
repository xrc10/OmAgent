from time import time
from typing import Dict, Any, Optional
from omagent_core.utils.general import encode_image
from .prompts import MULTIMODAL_QUERY_PROMPT

class QueryGenerator:
    """Generates memory search queries from multimodal inputs"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.prompts = {}
    
    def generate_multimodal_query(self, 
                                 user_instruction: str, 
                                 image_cache: Dict[str, Any],
                                 config_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate memory search query using both text and image input
        
        Args:
            user_instruction: User's query
            image_cache: Image data
            config_name: Name of LLM config to use
            
        Returns:
            Dictionary with query results
        """
        # Get LLM client
        llm_info = self.llm_manager.get_llm_client(config_name, "memory_query")
        client = llm_info["client"]
        model_id = llm_info["model_id"]
        temperature = llm_info["temperature"]
        
        # Use the prompt from self.prompts if available, otherwise use the imported one
        query_prompt = self.prompts.get("MULTIMODAL_QUERY_PROMPT", MULTIMODAL_QUERY_PROMPT)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that generates clear and specific search queries."},
            {"role": "user", "content": [
                {"type": "text", "text": query_prompt.format(user_instruction=user_instruction)}
            ]}
        ]

        # Add image from cache
        if image_cache:
            img = image_cache["<image_0>"]
            # Add image to the user message content
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(img)}"
                }
            })
        else:
            raise ValueError("Image cache is None, please check the image_cache in STM")

        # Make API call
        start_time = time()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        llm_time = time() - start_time
        query_response = response.choices[0].message.content

        # Parse query
        search_query = None
        image_description = None
        for line in query_response.split('\n'):
            if line.startswith('SEARCH_QUERY:'):
                search_query = line.split(':', 1)[1].strip()
            elif line.startswith('IMAGE:'):
                image_description = line.split(':', 1)[1].strip()

        # Use user instruction as fallback if no query found
        if not search_query:
            search_query = user_instruction

        return {
            "memory_search_query": search_query,
            "llm_time": llm_time,
            "image_description": image_description,
            "query_response": query_response,
            "model_used": model_id
        } 

    def update_prompts(self, prompts):
        """Update the prompts used by the query generator"""
        self.prompts = prompts
        return self.prompts 