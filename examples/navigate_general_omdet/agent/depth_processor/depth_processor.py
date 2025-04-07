from pathlib import Path
import requests
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from PIL import Image
from io import BytesIO

# API_KEY = '_meCERCZI4jhim5zm5Jh0yScxtTSGKFqWei2G0-boS0'
# API_URL = 'http://140.207.201.47:8085/predict'

API_KEY = 'XLPkBBDhcNwFmaAtbXE1i-G4gmcN1DHQTG3vECxWvO0'
# API_URL = 'http://140.207.201.47:8089/predict'
API_URL = 'http://localhost:8089/predict'

CONFIDENCE_THRESHOLD = 0.65
DEPTH_THRESHOLD = 3.0
MIN_VALID_RATIO = 0.1

@registry.register_worker()
class DepthProcessor(BaseWorker):
    """Worker that processes image through depth and object detection API"""

    def _run(self, *args, **kwargs):
        image_cache = self.stm(self.workflow_instance_id).get("image_cache", None)
        if not image_cache:
            return {"error": "No image found in cache"}

        image_url = image_cache["<image_0>"]
        
        # Determine if image is vertical or horizontal
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            width, height = img.size
            is_horizontal = width > height
            
            # Set ROI coordinates based on image orientation
            if is_horizontal:
                x1, y1, x2, y2 = 0.25, 0.25, 0.75, 0.75
            else:
                x1, y1, x2, y2 = 0.3, 0.3, 0.7, 0.7
                
            logging.info(f"Image dimensions: {width}x{height}, is_horizontal: {is_horizontal}, ROI: {x1},{y1},{x2},{y2}")
        except Exception as e:
            logging.error(f"Error determining image orientation: {e}")
            # Default to vertical ROI if there's an error
            x1, y1, x2, y2 = 0.3, 0.3, 0.7, 0.7
        
        # Prepare API request
        headers = {'X-API-Key': API_KEY}
        payload = {
            'url': image_url,
            "x1": x1,  # ROI coordinates
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "depth_threshold": DEPTH_THRESHOLD,  # Consider objects closer than 3m as obstacles
            "min_valid_ratio": MIN_VALID_RATIO,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
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
                    valid_obstacles = [obs for obs in obstacles if obs.get('min_depth') is not None]
                    if valid_obstacles:
                        valid_obstacles.sort(key=lambda x: x.get('min_depth', float('inf')))
                        closest_obstacle = valid_obstacles[0]
                        
                        # Create simplified message with only obstacle name and distance
                        obstacle_name = closest_obstacle.get('name', 'object')
                        obstacle_depth = closest_obstacle.get('min_depth', min_depth)
                        
                        # Determine position based on bbox center
                        bbox = closest_obstacle.get('bbox_norm', [0, 0, 0, 0])
                        if len(bbox) == 4:
                            # Calculate center x-coordinate of the bbox
                            center_x = (bbox[0] + bbox[2]) / 2
                            
                            # Determine position (left, center, right)
                            if center_x < 0.33:
                                position = "左侧"
                            elif center_x < 0.67:
                                position = "前方"
                            else:
                                position = "右侧"
                            
                            message = f"{position} {obstacle_name} {obstacle_depth:.1f}米"
                        else:
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
                        message = "没有障碍物"
                    
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