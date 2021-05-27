import json
from pprint import pprint
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# Initialize BERT model and tokenizer - variant is distilBERT, a distilled minor base version with no case rules.


class Semantic:
    """ Implements a semantic search pipeline over an index of search results from FAISS library """

    def __init__(self, documents, query, k):
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.documents = documents
        self.query = query
        self.k = k
        # self.embed = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')    # for sentence embeddings

    def encode(self, document: str) -> torch.Tensor:
        tokens = self.tokenizer(document, return_tensors='pt')
        vector = self.model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0)

    def build_index(self):
        """:returns an inverted index for the search documents"""
        vectors = [self.encode(document) for document in self.documents]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))  # dimensionality of vector space
        # Add document vectors into index after transforming into numpy arrays. IDs should match len(documents)
        index.add_with_ids(np.array([vec.numpy() for vec in vectors]), np.array(range(0, len(self.documents))))
        return index

    def search(self):
        encoded_query = self.encode(self.query).unsqueeze(dim=0).numpy()
        index = self.build_index()
        top_k = index.search(encoded_query, self.k)
        scores = top_k[0][0]
        results = [self.documents[_id] for _id in top_k[1][0]]
        return list(zip(results, scores))


