import requests
import base64
import json

# Path to your test image
image_path = "crows_4.jpg"

# Encode image to base64
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

# Create payload
payload = {
    "image_base64": base64_image
}

# Send POST request to /predict endpoint
response = requests.post(
    "http://118.138.234.214:30003/predict",  
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

# Print the response
print("Status code:", response.status_code)
print("Response JSON:", response.json())