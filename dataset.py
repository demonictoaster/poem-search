import os

from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


device = "cuda:0"
data_path = "./data/poems.csv"

#CLS Pooling - Take output from first token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings        
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output)

    return embeddings

def get_embeddings(text, path, from_disk):
    if not from_disk or not os.path.exists(path):
        embeddings = encode(text)
        torch.save(embeddings, path)
    else:
        print("Loading embeddings from {}".format(path))
        embeddings = torch.load(path)
    return embeddings

dataset = pd.read_csv(data_path)['poem_clean'].tolist()[:500]

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

emb_path = "./embeddings/embeddings.pt"
poem_emb = get_embeddings(dataset, emb_path, from_disk=False)

def get_scores(query, poem_emb):
    query_emb = encode(query)
    scores = torch.mm(query_emb, poem_emb.transpose(0, 1))[0].cpu().tolist()

    # combine and sort
    poem_score = list(zip(dataset, scores))
    poem_score = sorted(poem_score, key=lambda x: x[1], reverse=True)

    return poem_score


scores = get_scores("a poem about a church", poem_emb)

for poem, score in scores:
    print(score, poem[:100])