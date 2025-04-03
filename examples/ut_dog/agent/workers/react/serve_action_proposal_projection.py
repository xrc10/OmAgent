import base64
import io
import os
import json
import datetime
from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
from add_action_proposal_projection_fixed import (
    find_boundary_points,
    filter_points_by_angle,
    draw_action_proposals,
    calculate_turning_degree
)

app = Flask(__name__)

# Global variables to store the loaded model
device = None
processor = None
model = None

# Directory to save logs and images
LOG_DIR = "/data3/xu_ruochen/vlmnav_log"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model(model_path):
    """Load the segmentation model once at startup."""
    global device, processor, model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
    
    return model.config.id2label

def generate_action_proposals_from_image(image_data, min_angle=15, number_size=15, min_path_length=50, draw_degree=False):
    """Process an image and generate action proposals."""
    global device, processor, model
    
    # Get class ids for navigable regions
    id2label = model.config.id2label
    navigability_class_ids = [id for id, label in id2label.items() 
                            if 'floor' in label.lower() or 'rug' in label.lower()]
    
    # Process the image
    inputs = processor(images=image_data, return_tensors="pt").to(device)
    
    # Generate segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the segmentation output
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_data.size[::-1]])[0].cpu().numpy()
    
    # Create navigability mask
    navigability_mask = np.isin(predicted_semantic_map, navigability_class_ids)
    
    # Convert PIL Image to numpy array
    img_np = np.array(image_data)
    
    # Find boundary points of the navigability mask
    boundary_points = find_boundary_points(navigability_mask)
    
    # Calculate the virtual starting point (outside the image)
    height, width = img_np.shape[:2]
    start_point = (width // 2, height + int(height * 0.2))  # Point below the image
    
    # Filter points based on minimum angle between them
    filtered_points = filter_points_by_angle(boundary_points, start_point, min_angle=min_angle)
    
    # Calculate turning degrees for each point
    action_info = []
    for i, point in enumerate(filtered_points):
        turning_degree = calculate_turning_degree(point, start_point)
        action_info.append({
            'action_number': i + 1,
            'turning_degree': round(turning_degree, 1)
        })
    
    # Add the "turn around" option as action 0
    action_info.insert(0, {
        'action_number': 0,
        'turning_degree': 180.0
    })
    
    # Create action visualization image
    output_image = draw_action_proposals(img_np, filtered_points, start_point, 
                                       number_size=number_size, 
                                       navigability_mask=navigability_mask,
                                       min_path_length=min_path_length,
                                       draw_degree=draw_degree)
    
    return output_image, action_info

@app.route('/generate_action_proposals', methods=['POST'])
def process_image():
    """API endpoint to process images and generate action proposals."""
    try:
        # Get request data
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get parameters
        min_angle = data.get('min_angle', 15)
        number_size = data.get('number_size', 15)
        min_path_length = data.get('min_path_length', 50)
        draw_degree = data.get('draw_degree', False)
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        # Process the image
        output_image, action_info = generate_action_proposals_from_image(
            image, 
            min_angle=min_angle, 
            number_size=number_size,
            min_path_length=min_path_length,
            draw_degree=draw_degree
        )
        
        # Convert output image to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        output_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': output_base64,
            'actions': action_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_result', methods=['POST'])
def save_result():
    """API endpoint to save VLM navigation results."""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract required fields
        if 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        if 'vlm_output' not in data:
            return jsonify({'error': 'VLM output is required'}), 400
        if 'action_number' not in data:
            return jsonify({'error': 'Action number is required'}), 400
            
        # Ensure the log directory exists
        ensure_directory_exists(LOG_DIR)
        
        # Generate a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save the image
        try:
            image_bytes = base64.b64decode(data['image'])
            image_path = os.path.join(LOG_DIR, f"{timestamp}.jpg")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            return jsonify({'error': f'Failed to save image: {str(e)}'}), 500
        
        # Prepare the log entry
        log_entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'vlm_output': data['vlm_output'],
            'action_number': data['action_number'],
        }
        
        # Add all other fields from data to log_entry
        for key, value in data.items():
            if key not in ['image', 'vlm_output', 'action_number'] and key not in log_entry:
                log_entry[key] = value
        
        # Append to the JSONL file
        log_file = os.path.join(LOG_DIR, 'vlmnav_results.jsonl')
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            return jsonify({'error': f'Failed to write to log file: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'message': 'Result saved successfully',
            'image_path': image_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Configuration
    model_path = os.environ.get(
        'SEGMENTATION_MODEL_PATH', 
        "/data3/xu_ruochen/my_checkpoints/mask2former-swin-small-ade-semantic"
    )
    port = int(os.environ.get('PORT', 8075))
    
    # Load model at startup
    print(f"Loading segmentation model from {model_path}...")
    id2label = load_model(model_path)
    print(f"Model loaded successfully with {len(id2label)} classes.")
    
    # Ensure log directory exists
    ensure_directory_exists(LOG_DIR)
    print(f"Log directory: {LOG_DIR}")
    
    # Start server
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
