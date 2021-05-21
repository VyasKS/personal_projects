"""This file performs Term Frequency - Inverse Document Frequency search on the corpus of text data"""
# Import dependencies
import json
import itertools
from collections import Counter
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Spacy language model
nlp = spacy.load("en_core_web_sm")
# Data
with open('final_data.json', 'r') as outfile:
    data = json.load(outfile)

""" Creation of a corpus vocabulary which is a list of unique tokens in the provided set of documents and their respective counts."""
tokenized_text = [index["tokenized_text"] for index in data]
# Flatten list of lists
vocab = list(itertools.chain(*tokenized_text))
# Remove duplicates
vocab = list(set(vocab))
# Save vocabulary
with open('vocab.json', 'w') as outfile:
    json.dump(vocab, outfile)
# Token counter
token_counter = []
for doc in data:
    tokenized_doc = doc['tokenized_text']
    token_counter.append(Counter(tokenized_doc))
# Unique tokens in the corpus. Frequency of tokens in documents in corpus vocabulary
number_of_times = {}
for token in vocab:
    count = sum([1 for doc in token_counter if token in doc.keys()])
    number_of_times[token] = count

# Test with token 'pandemic'
print(f"Number of times the word pandemic occurred is {number_of_times['ebola']}")

""" TF - IDF computation. TF gives frequency of a word in whole set of documents and IDF penalizes such occurances and helps give weightage to rare words. Result is a word score that is relevant
to study. For example, the usually words and, the & certain prepositions in a corpus are valued although they appear everywhere. On the other hand, unique words are given weightage and importance."""
for index, document in enumerate(token_counter):
    doc_length = len(document)
    tfidf_vec = []
    for token in vocab:
        # Term Frequency
        tf = document[token] / len(data[index]["tokenized_text"])
        # Log of inverse document frequency
        idf = np.log(len(data) / number_of_times[token])

        tfidf = tf * idf
        tfidf_vec.append(tfidf)
    # Write tfidf_vector to all json objects
    data[index]['tf_idf'] = tfidf_vec

# Update the final_data.json
with open('final_data.json', 'w') as json_file:
    json.dump(data, json_file)

"""Query Vectorization"""


def tokenizer(document):
    text_lowercased = nlp(document.lower())
    tokens_without_stopwords = [word for word in text_lowercased if not word.is_stop and not word.is_punct and len(word.dep_.strip()) != 0]
    token_lemmatized = [token.lemma_ for token in tokens_without_stopwords]
    return token_lemmatized


def vectorizer(query, vocabulary=vocab):
    tokenized_query = tokenizer(query)
    token_counter_query = Counter(tokenized_query)
    vector_query = []
    for token in vocabulary:
        tf = token_counter_query[token] / len(tokenized_query)
        idf = np.log(len(data) / number_of_times[token])
        tfidf = tf * idf
        vector_query.append(tfidf)
    return vector_query


""" sklearn document search"""


def search_with_tfidf(query, documents):
    # Vectorize query
    query_vector = vectorizer(query)
    query_array = np.array(query_vector)

    # List of results and cosine similarity scores
    ranks = []
    for doc in documents:
        doc_rank = {}
        doc_arr = np.array(doc['tf_idf'])
        rank = cosine_similarity(query_array.reshape(1, -1), doc_arr.reshape(1, -1))[0][0]
        if rank > 0:
            doc_rank['title'] = doc['title']
            doc_rank['rank'] = rank
            ranks.append(doc_rank)

    return sorted(ranks, key=lambda k: k['rank'], reverse=True)


# Individual Tf-Idf based search for search phrase
print(search_with_tfidf("ebola", data))

# Testing search phrase with title whether keyword 'ebola' exist or not
for s in data:
    if s['title'] == "Plague of Cyprian":
        print(s['text'])
