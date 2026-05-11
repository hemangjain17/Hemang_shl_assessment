from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: Any
    duration: Any
    rationale: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

class ConstraintState(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    language: Optional[str] = None
    duration_max: Optional[int] = None
    remote_required: Optional[bool] = None
    explicit_nulls: List[str] = Field(default_factory=list)

class ExtractorResponse(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    language: Optional[str] = None
    duration_max: Optional[int] = None
    remote_required: Optional[bool] = None
    explicit_nulls_to_add: List[str] = Field(default_factory=list)

class RoleExpansion(BaseModel):
    roles: List[str] = Field(description="List of 3-5 concrete roles")
