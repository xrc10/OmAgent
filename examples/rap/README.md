# Reasoning via Planning (RAP) Example

This example demonstrates how to use the RAP (Reasoning via Planning) operator for complex reasoning tasks. The RAP operator treats reasoning as a planning problem and uses Monte Carlo Tree Search (MCTS) to explore reasoning paths.

## Overview

The example implements a RAP workflow that consists of the following components:

1. **Input Interface**
   - Handles user input containing reasoning queries

2. **RAP Workflow**
   - Uses MCTS to explore reasoning paths
   - Leverages a language model as both world model and reasoning agent
   - Finds high-reward reasoning paths

3. **Conclude Task**
   - Provides final answer based on the reasoning process

### Workflow Diagram:

![RAP Workflow](./docs/images/rap_workflow_diagram.png)

## Prerequisites

- Python 3.10+
- Required packages installed (see requirements.txt)
- Access to OpenAI API or compatible endpoint (see configs/llms/*.yml)
- Redis server running locally or remotely
- Conductor server running locally or remotely

## Configuration

1. Generate the container.yaml file:
   ```bash
   python compile_container.py
   ```

2. Configure your LLM settings in `configs/llms/*.yml`:
   ```bash
   export custom_openai_key="your_openai_api_key"
   export custom_openai_endpoint="your_openai_endpoint"
   ```

3. Update settings in the generated `container.yaml`:
   - Modify Redis connection settings
   - Update the Conductor server URL
   - Adjust any other component settings as needed

## Running the Example

For terminal/CLI usage:
```bash
python run_cli.py
```

For app/GUI usage:
```bash
python run_app.py
```

For lite version:
```bash
python run_lite.py
```

## Example Usage

Here's a simple example of using the RAP operator:

```python
from omagent_core.advanced_components.workflow.rap import RAPWorkflow

# Initialize RAP workflow
workflow = RAPWorkflow()

# Set input query
workflow.set_input(query="What would happen if we doubled the Earth's gravity?")

# Get results
rap_structure = workflow.rap_structure
final_answer = workflow.final_answer
```

## Troubleshooting

If you encounter issues:
- Verify Redis is running and accessible
- Check your OpenAI API key is valid
- Ensure all dependencies are installed correctly
- Review logs for any error messages
