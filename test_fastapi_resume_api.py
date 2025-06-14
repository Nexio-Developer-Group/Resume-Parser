import requests
import os
import time

API_URL = "http://127.0.0.1:8000/process_resume"
API_KEY = os.getenv("API_KEY", "your_secret_api_key_here")  # Replace with your actual API key or set as env var

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def test_api(username, email, live_link):
    payload = {
        "username": username,
        "email": email,
        "live_link": live_link
    }
    start = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    end = time.time()
    print(f"Status Code: {response.status_code}")
    try:
        print("Response:", response.json())
    except Exception:
        print("Response content:", response.content)
    print(f"API response time: {end - start:.2f} seconds\n")

if __name__ == "__main__":
    # Replace with actual values for a real test
    print("--- Valid Input ---")
    test_api("naman Jain", "namanjain2002qw@gmail.com", "https://drive.google.com/uc?export=download&id=19Sac_hbXfUbGf6YeBXg9slbVcxpAd9l7")
