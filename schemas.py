
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Pydantic Models 

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The long text to be summarized", min_length=50)
    max_words: int = Field(100, description="Target maximum words for the final summary", ge=1)

class SummarizeResponse(BaseModel):
    final_summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info like iteration count")

