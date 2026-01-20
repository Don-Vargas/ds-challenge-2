from pydantic import BaseModel, Field
from typing import List, Dict, Any


class TrainingRequest(BaseModel):
    version: str = Field(default="v1", example="v2")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
