""" Implements a search function using sentence-transformers library to read long documents """
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util


class LongDocumentSearch:
    """ Implements a search on long documents in a database. Uses sentence transformers to embed sentence vectors.
    Args:
        query: string - search query in the front end
        k: int - top k results to display
    :returns search answers"""

    def __init__(self, query, k=5):
        self.query = query
        self.k = k
        self.documents = self.get_data()

    @classmethod
    def get_data(cls):
        with open('data/data.json', 'r') as f:
            documents = json.load(f)
        return documents

    def get_results(self):
        embed = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        # Compute sentence embeddings for every text n the documents
        corpus = [d['text'] for d in self.documents]
        corpus_embeddings = np.array(embed.encode(corpus, convert_to_tensor=True))
        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        index.add_with_ids(corpus_embeddings, np.array(range(0, len(corpus))))
        # Write index for future usage
        faiss.write_index(index, 'pandemics')
        encoded_query = embed.encode([self.query])
        top_k = index.search(encoded_query, self.k)
        answers = [corpus[_id] for _id in top_k[1].tolist()[0]]
        return answers
