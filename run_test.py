import requests
import json
import base64
with open('test22.JPG', 'rb') as f:
  img_b64 = base64.b64encode(f.read()).decode('utf-8')
print('Sending request...')
response = requests.post('http://localhost:5000/api/verify-face', json={'image': img_b64})
print(json.dumps(response.json(), indent=2))
