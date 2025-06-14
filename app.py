from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI()

class InputData(BaseModel):
    name: str
    email: str
    link: str

def is_valid_email(email: str) -> bool:
    # Simple regex for email validation
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None

@app.post("/")
def process_input(data: InputData):
    if not is_valid_email(data.email):
        raise HTTPException(status_code=422, detail="Invalid email format.")
    if not (data.link.startswith("http://") or data.link.startswith("https://")):
        raise HTTPException(status_code=422, detail="Link must start with http:// or https://")
    return {
        "name": data.name,
        "email": data.email,
        "link": data.link
    }

