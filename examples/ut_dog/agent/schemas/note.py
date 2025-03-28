from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image

class VisionState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    image: Image.Image
    vyaw: float

class Step(BaseModel):
    plan: Optional[str] = Field(description="The plan of the step, describe what should be done.", default=None)
    execute_result: Optional[str] = Field(description="The result of the tool execution, describe what has been done.", default=None)
    tool_call: Optional[dict] = Field(description="The tool call details for this step", default=None)
    vision: List[VisionState] = Field(description="The current view of the robot, will be updated after the vision related tool is executed.", default=[])

class Task(BaseModel):
    instruction: str = Field(description="The instruction of the step, describe what should be done. Should be simple and feasible")
    steps: List[Step] = Field(description="The information about the steps that need to be executed to complete the task. Only return empty list when generating.", default=[])
    proof_of_completion: str = Field(description="Used to determine whether the step has been completed. It should be able to verify based on the current view of the robot. ")
    is_done: bool = Field(description="Whether the step is done.", literal=False)


class Note(BaseModel):
    origin_vision: List[VisionState] = Field(description="The initial environment of the robot. Only return empty list when generating.", default=[])
    content: List[Task]

    def __str__(self):
        result = []
        for task in self.content:
            checkbox = "[x]" if task.is_done else "[ ]" 
            result.append(f"## {checkbox} {task.instruction}")
            
            if task.steps:
                for i, step in enumerate(task.steps):
                    result.append(f"\n### Step {i+1}")
                    result.append(f"**Plan**: {step.plan}")
                    result.append(f"**Result**: {step.execute_result}")
            
            result.append("")  # Add empty line between tasks
        
        return "\n".join(result)
    
    def unfinished_tasks(self):
        return [task for task in self.content if not task.is_done]
    
    def finished_tasks(self):
        return [task for task in self.content if task.is_done]
    
    def current_task(self):
        return self.unfinished_tasks()[0]
    
    def current_step(self):
        return self.current_task().steps[-1]
    
    def previous_step(self):
        if len(self.current_task().steps) > 1:
            return self.current_task().steps[-2]
        else:
            return None
        
    def save(self, output_dir: str):
        """
        Save the Note as markdown files.
        
        Args:
            output_dir: The directory to save the note files
        """
        import os
        import base64
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create images directory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Create metadata directory to track changes
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Main markdown content
        main_content = []
        main_content.append("# Tasks\n")
        
        # Load existing metadata if available
        metadata_file = os.path.join(metadata_dir, "tasks_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"tasks": {}}
        
        # Process each task
        for task_idx, task in enumerate(self.content):
            task_id = f"task_{task_idx}"
            task_status = "completed" if task.is_done else "in_progress"
            
            # Check if task metadata exists and if content has changed
            task_changed = True
            if task_id in metadata["tasks"]:
                task_meta = metadata["tasks"][task_id]
                # Simple change detection based on instruction and status
                if (task_meta["instruction"] == task.instruction and 
                    task_meta["status"] == task_status):
                    task_changed = False
            
            # Update task metadata
            metadata["tasks"][task_id] = {
                "instruction": task.instruction,
                "status": task_status,
                "last_updated": datetime.now().isoformat()
            }
            
            # Task markdown filename
            task_filename = f"{task_id}_{task_status}.md"
            task_path = os.path.join(output_dir, task_filename)
            
            # Add link to main content
            checkbox = "[x]" if task.is_done else "[ ]"
            main_content.append(f"{checkbox} [{task.instruction}]({task_filename})\n")
            
            # Only update task file if changed
            if task_changed or not os.path.exists(task_path):
                # Create task content
                task_content = []
                task_content.append(f"# {task.instruction}\n")
                task_content.append(f"**Status**: {task_status}\n")
                
                if task.proof_of_completion:
                    task_content.append(f"**Proof of Completion**: {task.proof_of_completion}\n")
                
                # Process steps
                if task.steps:
                    task_content.append("\n## Steps\n")
                    
                    for step_idx, step in enumerate(task.steps):
                        step_content = []
                        step_content.append(f"### Step {step_idx+1}\n")
                        step_content.append(f"**Plan**: {step.plan}\n")
                        
                        if step.tool_call:
                            step_content.append("**Tool Call**:\n```json\n")
                            step_content.append(json.dumps(step.tool_call, indent=2))
                            step_content.append("\n```\n")
                        
                        if step.execute_result:
                            step_content.append(f"**Result**: {step.execute_result}\n")
                        
                        # Process vision data
                        if step.vision:
                            step_content.append("\n**Vision**:\n")
                            
                            for vision_idx, vision in enumerate(step.vision):
                                # Save the PIL image to a file
                                img_filename = f"{task_id}_step{step_idx}_vision{vision_idx}.jpg"
                                img_path = os.path.join(images_dir, img_filename)
                                
                                # Save image file if it's a PIL Image
                                if isinstance(vision.image, Image.Image):
                                    try:
                                        vision.image.save(img_path)
                                        # Add image reference to markdown
                                        relative_img_path = os.path.join("images", img_filename)
                                        step_content.append(f"![Vision {vision_idx}]({relative_img_path})\n")
                                        step_content.append(f"VYaw: {vision.vyaw}\n\n")
                                    except Exception as e:
                                        step_content.append(f"Error saving image: {str(e)}\n")
                                # Handle case where image might be a string path or base64
                                elif isinstance(vision.image, str):
                                    if vision.image.startswith(('data:image', 'base64')):
                                        # Extract the base64 content
                                        if ',' in vision.image:
                                            _, img_data = vision.image.split(',', 1)
                                        else:
                                            img_data = vision.image
                                        
                                        # Clean up base64 prefix if present
                                        if img_data.startswith('base64,'):
                                            img_data = img_data.replace('base64,', '', 1)
                                        
                                        # Save image file if it doesn't exist
                                        if not os.path.exists(img_path):
                                            try:
                                                img_data_decoded = base64.b64decode(img_data)
                                                with open(img_path, "wb") as img_file:
                                                    img_file.write(img_data_decoded)
                                            except Exception as e:
                                                step_content.append(f"Error saving image: {str(e)}\n")
                                        
                                        # Add image reference to markdown
                                        relative_img_path = os.path.join("images", img_filename)
                                        step_content.append(f"![Vision {vision_idx}]({relative_img_path})\n")
                                    else:
                                        # If it's already a path, just reference it
                                        step_content.append(f"![Vision {vision_idx}]({vision.image})\n")
                                    
                                    step_content.append(f"VYaw: {vision.vyaw}\n\n")
                        
                        task_content.extend(step_content)
                
                # Write task file
                with open(task_path, "w") as f:
                    f.write("\n".join(task_content))
        
        # Write main markdown file
        with open(os.path.join(output_dir, "index.md"), "w") as f:
            f.write("\n".join(main_content))
        
        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return output_dir