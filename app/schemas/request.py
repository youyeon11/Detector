from pydantic import BaseModel, Field

"""
탐지 요청에 대한 스키마
"""
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")


"""
가공 요청에 대한 스키마
"""
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to normalize and inspect")

