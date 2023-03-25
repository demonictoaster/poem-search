import time

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from embeddings.embeddings import PoemSearch
from schemas import SearchQuery, ModelResponse

# setup fastapi
app = FastAPI(title="Poem Search", openapi_url="/openapi.json")
api_router = APIRouter()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# setup semantic search
device = "cpu"
data_path = "./data/poems.csv"
emb_path = "./embeddings/embeddings.pt"

poem_search = PoemSearch(data_path, emb_path, device)

@api_router.get("/", status_code=200)
def root() -> dict:
    """
    Root GET
    """
    return {"msg": "Hello, World!"}

@api_router.post("/predict", response_model=ModelResponse)
async def predict(input: SearchQuery) -> ModelResponse:
    start = time.time()
    poem_id, score = poem_search.search(input.query)
    entry = poem_search.df.loc[poem_search.df['id']==poem_id]
    time_spent = time.time() - start
    response = ModelResponse(
        title = entry['title'].item(),
        author = entry['poet'].item(),
        poem = entry['poem'].item(),
        score = score,
        time = time_spent,
    )
    return response

app.include_router(api_router)

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")