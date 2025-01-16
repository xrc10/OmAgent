from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

@registry.register_worker()
class ObstacleSwitch(BaseWorker):
    """Worker that analyzes depth results and decides if obstacle detection is needed"""

    def _run(self, success: bool, min_depth: float, max_depth: float, avg_depth: float, *args, **kwargs):
        # Get depth values from input parameters
        if not success:
            return {"needs_detection": True, "error": "Previous depth task failed"}

        if min_depth is None or max_depth is None or avg_depth is None:
            return {"needs_detection": True, "error": "Missing depth parameters"}

        # Decision logic: 
        # If there's both near (<3m) and far (>6m) objects in the center region
        # it could indicate an obstacle
        has_near_object = min_depth < 3.0
        has_far_object = max_depth > 6.0
        # potential_obstacle = has_near_object and has_far_object
        potential_obstacle = has_near_object

        if potential_obstacle:
            # 发送初始警告(改为中文)
            warning_msg = (
                f"⚠️ 警告！检测到潜在障碍物。"
                f"最近点距离：{min_depth:.1f}米，"
                f"最远点距离：{max_depth:.1f}米"
            )
            # self.callback.send_answer(
            #     self.workflow_instance_id,
            #     msg=warning_msg
            # )
        else:
            self.callback.send_answer(
                self.workflow_instance_id,
                msg="未检测到障碍物"
            )

        return {
            "needs_detection": potential_obstacle,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "avg_depth": avg_depth
        } 