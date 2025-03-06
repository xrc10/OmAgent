import os
import yaml
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, Optional, Tuple

class LLMManager:
    """Manages LLM configurations and client instances"""
    
    def __init__(self):
        self.llm_clients = {}
        self.llm_configs = {}
        
        # Default LLM config names for different tasks
        self.default_config_names = {
            "memory_query": "intern25_vl_1b_remote.yml",
            "answer_generator": "qwen25_vl_3b_remote.yml",
            "text_answer": "qwen25_7b_remote.yml"
        }
        
        # Load available LLM configs
        self._load_llm_configs()
    
    def _load_llm_configs(self):
        """Load all available LLM configurations from the configs directory"""
        # Find the config directory
        current_dir = Path(__file__).parents[1]
        config_dir = os.path.join(current_dir, "../configs", "llms")
        # print("Config directory:", config_dir)
        config_dir = Path(config_dir)
        
        # List all YAML files in the directory
        config_files = list(config_dir.glob("*.yml"))
        # print("Loaded LLM configs:", config_files)
        
        # Load each config file
        for config_file in config_files:
            config_name = config_file.name
            with open(config_file, 'r') as file:
                self.llm_configs[config_name] = yaml.safe_load(file)
    
    def get_llm_client(self, config_name: Optional[str] = None, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get or create an LLM client for the specified configuration
        
        Args:
            config_name: Name of the config file to use
            task_type: Type of task to get default config for
            
        Returns:
            Dictionary with client, model_id and temperature
        """
        # If task_type is provided and config_name is not, use default for task
        if not config_name and task_type and task_type in self.default_config_names:
            config_name = self.default_config_names[task_type]
            
        # If client already exists, return it
        if config_name in self.llm_clients:
            return self.llm_clients[config_name]
        
        # If config doesn't exist, use default
        if config_name not in self.llm_configs:
            config_name = "gpt.yml"
        
        # Get configuration
        config = self.llm_configs[config_name]
        
        # Extract configuration values
        api_key = config.get('api_key', '')
        endpoint = config.get('endpoint', 'https://api.openai.com/v1')
        model_id = config.get('model_id', 'gpt-4o-mini')
        temperature = config.get('temperature', 0)
        
        # Handle environment variable references
        if isinstance(api_key, str) and api_key.startswith('${env|'):
            parts = api_key.strip('${env|}').split(',')
            env_var = parts[0].strip()
            default_value = parts[1].strip() if len(parts) > 1 else ''
            api_key = os.environ.get(env_var, default_value)
        
        if isinstance(endpoint, str) and endpoint.startswith('${env|'):
            parts = endpoint.strip('${env|}').split(',')
            env_var = parts[0].strip()
            default_value = parts[1].strip() if len(parts) > 1 else ''
            endpoint = os.environ.get(env_var, default_value)
        
        # Create client
        client = OpenAI(api_key=api_key, base_url=endpoint)
        
        # Store client and model info
        self.llm_clients[config_name] = {
            "client": client,
            "model_id": model_id,
            "temperature": temperature
        }
        
        return self.llm_clients[config_name] 