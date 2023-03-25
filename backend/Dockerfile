FROM python:3.10.9

WORKDIR /api
RUN mkdir -p /api/embeddings
RUN mkdir -p /api/data

# first we install dependencies so it is cached by docker
# saves time when rebuilding the image
RUN pip3 install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY ./api/requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# add the code near the end since this is what changes most frequently
COPY ./api/main.py /api
COPY ./api/schemas.py /api
COPY ./api/embeddings/embeddings.py /api/embeddings
COPY ./api/embeddings/embeddings.pt /api/embeddings
COPY ./api/data/poems.csv /api/data

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]