import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


device = "cuda:0" if torch.cuda.is_available() else "cpu"
data_path = "./data/poems.csv"
emb_path = "./embeddings/embeddings.pt"

class PoemSearch:
    def __init__(self, data_path, emb_path, device):
        self.data_path = data_path
        self.emb_path = emb_path
        self.device = device

        self.batch_size = 8

        self.model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt)
        self.model = self.model.to(self.device)

        self.df = pd.read_csv(self.data_path)
        self.data = self.df['poem_clean'].tolist()

        self.embeddings = self.load_embeddings()

    def compute_embeddings(self):
        print("Computing poem embeddings...")
        dataloader = DataLoader(self.data, batch_size=self.batch_size)
        embeddings = torch.Tensor().to(self.device)
        n_batches = len(dataloader)

        batch_nr = 1
        for batch in dataloader:
            print("Processing batch nr {} / {}".format(
                batch_nr, n_batches
            ))
            embs = self.encode(batch)
            embeddings = torch.cat([embeddings, embs])
            batch_nr += 1

        torch.save(embeddings, self.emb_path)

        return embeddings

    def load_embeddings(self):
        if os.path.exists(self.emb_path):
            print("Loading embeddings from {}".format(self.emb_path))
            embeddings = torch.load(self.emb_path)
            embeddings = embeddings.to(self.device)
        else:
            print("No such file '{}'\nComputing embeddings from raw data.".format(self.emb_path))
            embeddings = self.compute_embeddings()

        return embeddings


    def encode(self, texts):
        # tokenize
        tokenized_input = self.tokenize(texts)

        # Compute embeddings        
        with torch.no_grad():
            model_output = self.model(**tokenized_input, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output)

        return embeddings

    def tokenize(self, texts):
        tokenized_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        tokenized_input = tokenized_input.to(self.device) 
        return tokenized_input

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:,0]
    
    def search(self, query):
        query_emb = self.encode(query)
        scores = torch.mm(query_emb, self.embeddings.transpose(0, 1))[0].cpu().tolist()

        # combine and sort
        poem_score = list(zip(self.df['id'].tolist(), scores))
        poem_score = sorted(poem_score, key=lambda x: x[1], reverse=True)

        for id, score in poem_score[:3]:
            print("Score = {}".format(score))
            print(self.df.loc[self.df['id']==id]['title'].item())
            print(self.df.loc[self.df['id']==id]['poem'].item())
            print("=" * 50)

poem_search = PoemSearch(data_path, emb_path, device)
poem_search.search("listing a bunch of items")