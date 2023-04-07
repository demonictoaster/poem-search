# poem-search
A small web app that can perform semantic search over poems from the [Poetry Foundation](https://www.poetryfoundation.org/). Poems are embedded using Hugging Face's Transformer library and stored on the backend. At runtime, the query is embedded using the same model and compared against the poems' embeddings.
# Backend
To build the backend container:

`cd backend`

`docker build -t poem-search-api .`

To run the backend container run: 

`docker run -d --name mycontainer -p 8000:80 poem-search-api`