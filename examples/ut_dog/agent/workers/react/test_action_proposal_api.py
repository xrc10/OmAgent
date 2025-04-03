import os
import base64
import requests
import glob
from PIL import Image
import io
import time

"""
Action Proposal API

Endpoint: /generate_action_proposals

Description:
    This API generates navigation action proposals based on an input image. 
    It analyzes the scene and suggests possible navigation paths with turning degrees.

Input:
    JSON payload with the following fields:
    - image (str): Base64-encoded image data
    - min_angle (int): Minimum angle difference between adjacent proposals (default: 40)
    - number_size (int): Size of the number markers in the output visualization (default: 30)
    - min_path_length (int): Minimum path length to consider for proposals (default: 200)

Output:
    JSON response with the following fields:
    - actions (list): List of action proposals, each containing a dictionary with the following keys :
        - action_number (int): Index of the action proposal
        - turning_degree (float): Suggested turning angle in degrees
    - image (str): Base64-encoded image with visualized action proposals
"""

def test_action_proposal_api(
    api_url="http://localhost:8075/generate_action_proposals",
    image_pattern=".MVIMG_*",
    output_dir="./output/",
    min_angle=40,
    number_size=30,
    min_path_length=200
):
    """
    Test the action proposal API by sending local images and saving the results.
    
    Args:
        api_url: URL of the API endpoint
        image_pattern: Glob pattern to find input images
        output_dir: Directory to save output images
        min_angle: Minimum angle between proposals
        number_size: Size of the number markers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching image files
    image_files = glob.glob(image_pattern)
    print(f"Found {len(image_files)} images matching pattern '{image_pattern}'")
    
    if not image_files:
        print("No images found. Please check the pattern.")
        return
    
    for image_path in image_files:
        # Get filename without path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {filename}...")
        
        try:
            # Read image and convert to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Prepare request payload
            payload = {
                "image": image_base64,
                "min_angle": min_angle,
                "number_size": number_size,
                "min_path_length": min_path_length
            }
            
            # Send request to API
            response = requests.post(api_url, json=payload, timeout=60)
            
            print(response.json().keys()) # dict_keys(['actions', 'image'])
            # actions [{'action_number': 0, 'turning_degree': 180.0}, {'action_number': 1, 'turning_degree': -56.9}, {'action_number': 2, 'turning_degree': -16.8}, {'action_number': 3, 'turning_degree': 23.3}, {'action_number': 4, 'turning_degree': 69.8}]
            # image is base64 encoded image in string

            if response.status_code == 200:
                # Get response data
                data = response.json()
                output_base64 = data["image"]
                actions = data["actions"]
                
                # Decode base64 image
                output_bytes = base64.b64decode(output_base64)
                
                # Save output image
                with open(output_path, "wb") as f:
                    f.write(output_bytes)
                
                print(f"✓ Saved output to {output_path}")
                print(f"  Actions: {actions}")
            else:
                print(f"✗ Error: API returned status code {response.status_code}")
                print(f"  Response: {response.text}")
        
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    # You can customize these parameters if needed
    API_URL = "http://localhost:8075/generate_action_proposals"
    IMAGE_PATTERN = "./MVIMG_*"  # Pattern to match input images
    OUTPUT_DIR = "./output/"
    MIN_ANGLE = 40
    NUMBER_SIZE = 20
    MIN_PATH_LENGTH = 50
    
    #iterate all images in the current directory
    for image_path in glob.glob(IMAGE_PATTERN):
        print(f"Processing {image_path}...")
        start_time = time.time()
        test_action_proposal_api(
            api_url=API_URL,
            image_pattern=image_path,
            output_dir=OUTPUT_DIR,
            min_angle=MIN_ANGLE,
            number_size=NUMBER_SIZE,
            min_path_length=MIN_PATH_LENGTH
        ) 
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")