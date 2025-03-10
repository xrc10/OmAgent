import requests
import base64
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import io
import time

# Add API key constant at the top of the file
API_KEY = '_meCERCZI4jhim5zm5Jh0yScxtTSGKFqWei2G0-boS0'

def test_health():
    """Test the health check endpoint"""
    headers = {'X-API-Key': API_KEY}
    response = requests.get('http://140.207.201.47:8085/health', headers=headers)
    print(f"Health check status: {response.json()}")

def test_depth_from_url(image_url):
    """Test the depth prediction endpoint using an image URL"""
    headers = {'X-API-Key': API_KEY}
    payload = {
        'url': image_url,
        "x1": 0.3,
        "y1": 0.3,
        "x2": 0.7,
        "y2": 0.7
    }
    
    response = requests.post('http://140.207.201.47:8085/predict', json=payload, headers=headers)
    return response.json()

def test_depth_from_file(image_path):
    """Test the depth prediction endpoint using a local image file"""
    headers = {'X-API-Key': API_KEY}
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    payload = {
        'image': image_data
    }
    
    response = requests.post('http://140.207.201.47:8085/predict', json=payload, headers=headers)
    return response.json()

def visualize_depth_map(depth_map):
    """Visualize the depth map using matplotlib"""
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()

def main():
    # Test health endpoint
    test_health()
    
    # Test with an image URL
    print("\nTesting with image URL...")
    image_url = "https://hd-node1.linker.cc/minio/hzlh/agiapp/temp/2025/01/14/20250114164125948_screenshot_1736844085464.jpg"
    start = time.time()
    result = test_depth_from_url(image_url)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    if result['status'] == 'success':
        print("Successfully processed image from URL")
        # Decode and visualize depth map
        print(result['depth_statistics']['min_depth'])
        print(result['depth_statistics']['max_depth'])
        print(result['depth_statistics']['avg_depth'])
    else:
        print(f"Error processing URL image: {result['message']}")
    
if __name__ == "__main__":
    main() 