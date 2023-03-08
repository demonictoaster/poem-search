import os
import time

from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig


start = time.time()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
data_path = "./data/poems.csv"

#CLS Pooling - Take output from first token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

# tokenizer
def tokenize(texts):
    tokenized_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    tokenized_input = tokenized_input.to(device) 
    return tokenized_input

# encode text
def encode(texts):
    # tokenize
    tokenized_input = tokenize(texts)

    # Compute embeddings        
    with torch.no_grad():
        model_output = model(**tokenized_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output)

    return embeddings

def get_embeddings(text, path, from_disk):
    if not from_disk or not os.path.exists(path):
        dataloader = DataLoader(text, batch_size=8)
        embeddings = torch.Tensor().to(device)
        n_batches = len(dataloader)

        batch_nr = 1
        for batch in dataloader:
            print("Processing batch nr {} / {}".format(
                batch_nr, n_batches
            ))
            embs = encode(batch)
            embeddings = torch.cat([embeddings, embs])
            batch_nr += 1

        torch.save(embeddings, path)
    else:
        print("Loading embeddings from {}".format(path))
        embeddings = torch.load(path)
    return embeddings

data = pd.read_csv(data_path)['poem_clean'].tolist()
ds = Dataset.from_dict({'data': data}).with_format('torch')

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model = model.to(device)

emb_path = "./embeddings/embeddings.pt"
poem_emb = get_embeddings(data, emb_path, from_disk=False)

def get_scores(query, poem_emb):
    query_emb = encode(query)
    scores = torch.mm(query_emb, poem_emb.transpose(0, 1))[0].cpu().tolist()

    # combine and sort
    poem_score = list(zip(data, scores))
    poem_score = sorted(poem_score, key=lambda x: x[1], reverse=True)

    return poem_score


scores = get_scores("the world is an unfair place", poem_emb)

for poem, score in scores[:10]:
    print(score, poem)
    print('=' * 50)
print(len(scores))
print("time: {}".format(time.time() - start))