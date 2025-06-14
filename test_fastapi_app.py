import requests

API_URL = "http://127.0.0.1:8000/"

def test_api(name, email, link):
    payload = {
        "name": name,
        "email": email,
        "link": link
    }
    response = requests.post(API_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response:", response.json())

if __name__ == "__main__":
    print("--- Valid Input ---")
    test_api("John Doe", "john@example.com", "https://nexiotech.cloud/")