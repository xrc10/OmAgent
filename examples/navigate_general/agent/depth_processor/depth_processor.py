from pathlib import Path
import requests
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

API_KEY = '_meCERCZI4jhim5zm5Jh0yScxtTSGKFqWei2G0-boS0'
API_URL = 'http://140.207.201.47:8085/predict'

@registry.register_worker()
class DepthProcessor(BaseWorker):
    """Worker that processes image through depth estimation API"""

    def _run(self, *args, **kwargs):
        image_cache = self.stm(self.workflow_instance_id).get("image_cache", None)
        if not image_cache:
            return {"error": "No image found in cache"}

        image_url = image_cache["<image_0>"]
        
        # Prepare API request
        headers = {'X-API-Key': API_KEY}
        payload = {
            'url': image_url,  # Using URL directly instead of base64 image
            "x1": 0.3,  # Center region coordinates
            "y1": 0.3,
            "x2": 0.7,
            "y2": 0.7
        }

        # Call depth API
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            result = response.json()
            
            if result['status'] == 'success':
                depth_stats = result['depth_statistics']
                self.stm(self.workflow_instance_id)["depth_result"] = depth_stats
                return {
                    "success": True,
                    "min_depth": depth_stats['min_depth'],
                    "max_depth": depth_stats['max_depth'],
                    "avg_depth": depth_stats['avg_depth']
                }
            else:
                return {"success": False, "error": result.get('message', 'Unknown error')}
                
        except Exception as e:
            logging.error(f"Error calling depth API: {e}")
            return {"success": False, "error": str(e)} 