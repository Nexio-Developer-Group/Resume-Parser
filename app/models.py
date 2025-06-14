from pydantic import BaseModel, EmailStr, HttpUrl

class ResumeRequest(BaseModel):
    username: str
    email: EmailStr
    live_link: HttpUrl