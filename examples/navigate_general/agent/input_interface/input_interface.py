from pathlib import Path
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

@registry.register_worker()
class NavigationInputInterface(BaseWorker):
    """Input interface processor that handles image input for navigation."""

    def _run(self, *args, **kwargs):
        # user_input = self.input.read_input(
        #     workflow_instance_id=self.workflow_instance_id,
        #     input_prompt="Please provide a image."
        # )

        user_input = self.input.read_first_input(
            workflow_instance_id=self.workflow_instance_id,
        )

        image_path = None
        # Extract image content from input message
        content = user_input["messages"][-1]["content"]
        for content_item in content:
            if content_item["type"] == "image_url":
                image_path = content_item["data"]

        logging.info(f"Image_path: {image_path}")

        # Store image URL in workflow shared memory
        if image_path:
            image_cache = {"<image_0>": image_path}
            self.stm(self.workflow_instance_id)["image_cache"] = image_cache

        return {"image_path": image_path} 