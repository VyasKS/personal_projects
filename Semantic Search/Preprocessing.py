# Load data from a JSON file
# Uncomment the lines to display output that helps understanding.
import spacy
import json
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
with open('data.json', 'r') as f:
    data = json.load(f)
print(data[0].keys()) # keys are ['title, 'text', 'url']
print(data)

""" Extract 'text' from the JSON file and store the entire text in a list for downstream NLP tasks"""
text_for_NLP = []
for part in data: # this is a JSON object among other objects in a list
    # now part is a dictionary with title, text, url as keys
    for attribute, value in part.items():
        if attribute == 'text':
            text_for_NLP.append(value)
print(text_for_NLP)

text = u"{}".format(text_for_NLP)

nlp = spacy.load("en_core_web_sm")

doc = nlp(text.lower())
for token in doc[:5]:
    print(type(token), token.text, token.pos_, token.dep_)

""" Unclassified tokens processing"""
unclassified_tokens = [(token.lemma_, token.dep_) for token in doc if token.dep_ == ""]
print(unclassified_tokens[:10])

"""Stop word and punctuation removal"""
tokens_without_sw = [word for word in doc if not word.is_stop and not word.is_punct]
print(tokens_without_sw[:10])

token_lemmas = [token.lemma_ for token in tokens_without_sw if token.dep_]
print(token_lemmas[:10])


def tokenizer(document):
    text_lowercased = nlp(document.lower())
    tokens_without_stopwords = [word for word in text_lowercased if not word.is_stop and not word.is_punct] #and len(word.dep_.strip()) != 0
    token_lemmatized = [token.lemma_ for token in tokens_without_stopwords if token.dep_]
    return token_lemmatized


for doc in data:
    doc['tokenized_text'] = tokenizer(doc['text'])

print(data[0]['tokenized_text'])

""" Save the tokenized text to a file"""
with open('final_data.json', 'w') as outfile:
    json.dump(data, outfile)
