from pydantic import BaseModel 


class SearchQuery(BaseModel):
    query: str

class ModelResponse(BaseModel):
    title: str
    author: str
    poem: str
    score: float
    time: float
