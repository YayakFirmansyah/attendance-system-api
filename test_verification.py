import base64
import requests
import json

def test_verification(image_path):
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    url = 'http://localhost:5000/api/verify-face'
    payload = {
        'image': f'data:image/jpeg;base64,{image_data}'
    }
    
    # Send request
    response = requests.post(url, json=payload)
    
    # Print result
    print(json.dumps(response.json(), indent=2))

# Test dengan gambar
test_verification('test.jpg')