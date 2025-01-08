# Visual Question Answering with Memory Example

This example demonstrates how to use the framework for visual question answering (VQA) tasks with memory capabilities. The example code can be found in the `examples/VQA_with_mem0` directory.

```bash
cd examples/VQA_with_mem0
```

## Overview

This example implements a Visual Question Answering (VQA) workflow with memory that consists of two main components:

1. **Input Interface**
   - Handles user input containing questions about images
   - Processes and manages image data
   - Extracts the user's questions/instructions

2. **VQA with Memory Processing**
   - Takes the user input and image
   - Searches relevant memories if needed
   - Analyzes the image based on the user's question
   - Generates appropriate responses to visual queries
   - Stores new memories when relevant

The workflow follows a straightforward sequence:

## Prerequisites

- Python 3.10+
- Required packages installed (see requirements.txt)
- Access to OpenAI API or compatible endpoint (see configs/llms/gpt.yml)
- Redis server running locally or remotely
- Conductor server running locally or remotely

## Configuration

The container.yaml file is a configuration file that manages dependencies and settings for different components of the system, including Conductor connections, Redis connections, and other service configurations. To set up your configuration:

1. Generate the container.yaml file:
   ```bash
   python compile_container.py
   ```
   This will create a container.yaml file with default settings under `examples/VQA_with_mem0`.


2. Configure your LLM settings in `configs/llms/gpt.yml`:
   - Set your OpenAI API key or compatible endpoint through environment variable or by directly modifying the yml file
   ```bash
   export custom_openai_key="your_openai_api_key"
   export custom_openai_endpoint="your_openai_endpoint"
   ```
   - Configure other model settings like temperature as needed through environment variable or by directly modifying the yml file

3. Update settings in the generated `container.yaml`:
   - Modify Redis connection settings:
     - Set the host, port and credentials for your Redis instance
     - Configure both `redis_stream_client` and `redis_stm_client` sections
   - Update the Conductor server URL under conductor_config section
   - Adjust any other component settings as needed

## Running the Example

3. Run the VQA with memory example:

   For terminal/CLI usage:
   ```bash
   python run_cli.py
   ```

   For app/GUI usage:
   ```bash
   python run_app.py
   ```

## Troubleshooting

If you encounter issues:
- Verify Redis is running and accessible
- Check your OpenAI API key is valid
- Ensure all dependencies are installed correctly
- Review logs for any error messages


## Building the Example

Coming soon! This section will provide detailed instructions for building and packaging the VQA with memory example step by step.

