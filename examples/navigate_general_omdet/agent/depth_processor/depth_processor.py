from pathlib import Path
import requests
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

# API_KEY = '_meCERCZI4jhim5zm5Jh0yScxtTSGKFqWei2G0-boS0'
# API_URL = 'http://140.207.201.47:8085/predict'

API_KEY = 'XLPkBBDhcNwFmaAtbXE1i-G4gmcN1DHQTG3vECxWvO0'
API_URL = 'http://140.207.201.47:8089/predict'

@registry.register_worker()
class DepthProcessor(BaseWorker):
    """Worker that processes image through depth and object detection API"""

    def _run(self, *args, **kwargs):
        image_cache = self.stm(self.workflow_instance_id).get("image_cache", None)
        if not image_cache:
            return {"error": "No image found in cache"}

        image_url = image_cache["<image_0>"]
        
        # Prepare API request
        headers = {'X-API-Key': API_KEY}
        payload = {
            'url': image_url,
            "x1": 0.3,  # ROI coordinates
            "y1": 0.3,
            "x2": 0.7,
            "y2": 0.7,
            "depth_threshold": 3.0,  # Consider objects closer than 3m as obstacles
            "min_valid_ratio": 0.1,
            "save_results": False
        }

        # Call combined depth and object detection API
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            result = response.json()
            
            if result['status'] == 'success':
                # Store full result in shared memory
                self.stm(self.workflow_instance_id)["depth_result"] = result
                
                # Extract depth statistics
                depth_stats = result['depth_statistics']
                min_depth = depth_stats['min_depth']
                max_depth = depth_stats['max_depth']
                avg_depth = depth_stats['avg_depth']
                
                # Process obstacle information
                obstacles = result.get('obstacles', [])
                total_objects = result.get('total_objects', 0)
                roi_objects = result.get('roi_objects', 0)
                
                # Generate user feedback based on obstacles
                if obstacles:
                    # Filter out obstacles with None depth values and sort by depth (closest first)
                    valid_obstacles = [obs for obs in obstacles if obs.get('avg_depth') is not None]
                    if valid_obstacles:
                        valid_obstacles.sort(key=lambda x: x.get('avg_depth', float('inf')))
                        closest_obstacle = valid_obstacles[0]
                        
                        # Create simplified message with only obstacle name and distance
                        obstacle_name = closest_obstacle.get('name', 'object')
                        obstacle_depth = closest_obstacle.get('avg_depth', min_depth)
                        
                        message = f"{obstacle_name} {obstacle_depth:.1f}米"
                        
                        # Send message to user
                        self.callback.send_answer(
                            self.workflow_instance_id,
                            msg=message
                        )
                    else:
                        # Obstacles exist but none have valid depth values
                        message = "障碍物 未知距离"
                        self.callback.send_answer(
                            self.workflow_instance_id,
                            msg=message
                        )
                else:
                    # No obstacles detected
                    if min_depth < 3.0:
                        message = f"未识别物体 {min_depth:.1f}米"
                    else:
                        message = "无障碍物"
                    
                    self.callback.send_answer(
                        self.workflow_instance_id,
                        msg=message
                    )
                
                return {
                    "success": True,
                    "min_depth": min_depth,
                    "max_depth": max_depth,
                    "avg_depth": avg_depth,
                    "obstacles": obstacles,
                    "total_objects": total_objects,
                    "roi_objects": roi_objects
                }
            else:
                error_msg = result.get('message', 'Unknown error')
                self.callback.send_answer(
                    self.workflow_instance_id,
                    msg=f"处理失败: {error_msg}"
                )
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logging.error(f"Error calling depth and object detection API: {e}")
            self.callback.send_answer(
                self.workflow_instance_id,
                msg=f"系统错误: {str(e)}"
            )
            return {"success": False, "error": str(e)} 