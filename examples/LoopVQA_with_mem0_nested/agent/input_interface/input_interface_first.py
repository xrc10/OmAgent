from pathlib import Path

from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry

CURRENT_PATH = Path(__file__).parents[0]


@registry.register_worker()
class InputInterfaceFirst(BaseWorker):
    """Input interface processor that handles user instructions and image input.

    This processor:
    1. Reads user input containing question and image via input interface
    2. Extracts text instruction and image path from the input
    3. Loads and caches the image in workflow storage
    4. Returns the user instruction for next steps
    """

    def _run(self, *args, **kwargs):
        # Get conversation history from STM
        conversation_history = self.stm(self.workflow_instance_id).get("conversation_history", [])
        
        # Format conversation history for display
        history_display = ""
        if conversation_history:
            history_display = "\n历史对话:\n"
            for i, conv in enumerate(conversation_history):
                history_display += f"问: {conv['question']}\n答: {conv['answer']}\n"
            
            # Display conversation history to user
            # self.callback.send_answer(
            #     self.workflow_instance_id,
            #     msg=history_display
            # )

        # Read user input through configured input interface

        # user_input = self.input.read_input(
        #     workflow_instance_id=self.workflow_instance_id,
        #     input_prompt="Please provide your question and image."
        # )

        user_input = self.input.read_first_input(
            workflow_instance_id=self.workflow_instance_id,
        )

        # Extract user_id from kwargs if present
        user_id = None
        if "kwargs" in user_input:
            for kwarg in user_input["kwargs"]:
                if kwarg["key"] == "userId":
                    user_id = kwarg["value"]
                    # Store user_id in shared memory
                    self.stm(self.workflow_instance_id)["user_id"] = user_id
                    break

        image_path = None
        # Extract text and image content from input message
        content = user_input["messages"][-1]["content"]
        for content_item in content:
            if content_item["type"] == "text":
                user_instruction = content_item["data"]
            elif content_item["type"] == "image_url":
                image_path = content_item["data"]

        logging.info(f"User_instruction: {user_instruction}\nImage_path: {image_path}")
        self.stm(self.workflow_instance_id)["user_instruction"] = user_instruction
        self.stm(self.workflow_instance_id)["image_url"] = image_path

        # Load image from file system
        if image_path:
            img = read_image(input_source=image_path)

            # Store image in workflow shared memory with standard key
            image_cache = {"<image_0>": img}
            self.stm(self.workflow_instance_id)["image_cache"] = image_cache

        # Determine if should exit
        should_exit = False
        if any(keyword in user_instruction.lower() for keyword in ["退出", "结束", "再见"]):
            should_exit = True

        return {"user_instruction": user_instruction, "user_id": user_id, "image_url": image_path, "should_exit": should_exit}