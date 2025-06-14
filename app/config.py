import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)
