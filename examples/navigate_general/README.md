# Navigation Assistant for Blind People

This example implements a navigation assistance system that helps blind people detect and avoid obstacles. The system uses depth estimation and multimodal LLM technologies to provide real-time navigation guidance.

## Workflow

1. **Input Processing**: Handles incoming camera/image input
2. **Depth Analysis**: Processes image through depth estimation API
3. **Obstacle Detection**: 
   - Analyzes depth information for potential obstacles
   - Uses multimodal LLM for detailed obstacle description if needed
4. **User Feedback**: Provides clear audio/text feedback about obstacles and navigation

## Configuration

1. Set up your environment variables:
```bash
export custom_openai_key="your_openai_api_key"
export custom_openai_endpoint="your_openai_endpoint"
```

2. Configure the container:
```bash
python compile_container.py
```

## Usage

Run the navigation assistant:
```bash
python run_app.py
```

## Features

- Real-time depth estimation
- Intelligent obstacle detection
- Natural language descriptions of obstacles
- Memory of navigation context
- Clear and concise user feedback

## Safety Notice

This system is an assistive tool and should not be relied upon as the sole means of navigation. Always use proper mobility aids and techniques for safe navigation. 