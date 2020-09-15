""" Coding a chat-bit using Transformers neural networks from scratch. We can also use Transformer class from PyTorch. Coding here for a deeper
understanding of the architecture"""

# Importing necessary libraries
from collections import Counter          # for counting objects
import json                              # for java script object notation reading
import torch                             # Powerful PyTorch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import math
import torch.nn.functional as F

corpus_movie_conversations = 'data/movie_conversations.txt'
corpus_movie_lines = 'data/movie_lines.txt'

# We will load & feed data to network in batches and we define a maximum length parameter for matrix and pad shorter sentences
max_len = 25

# Read the data file
with open(corpus_movie_conversations, 'r', encoding='unicode_escape') as c:
    conv = c.readlines()            # inbuilt Python function to read lines from text

with open(corpus_movie_lines, 'r', encoding='unicode_escape') as lines:
    lines = lines.readlines()      # inbuilt Python function to read lines from text

# Check whether data loaded properly after encoding
print(conv)
print(lines)

# Lines are given as ['L523196', 'L523197'] and preceding series('u' & 'm') define their respective conversations
# First comes the question, then the reply. 'lines' defines the content and speaker

# Preprocessing the data by splitting between special characters with spaces [ +++$+++ ]
# They made dataset in such a way that makes it easy for extraction
lines_dict = {}
for line in lines:
    objects = line.split(' +++$+++ ')
    # key is line id
    lines_dict[objects[0]] = objects[-1]            # mapping values to keys

print(lines_dict)


# Remove punctuation
def remove_punctuation(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punctuation = ""                             # makes strings arrange within the quotes
    for char in string:
        if char not in punctuations:
            no_punctuation += char
    return no_punctuation.lower()                   # lower case characters


# We extract answers for every question in list of lines in the conversations.txt file after splitting them
pairs = []
for conv in conv:
    ids = eval(conv.split(' +++$+++ ')[-1])         # eval() will remove strings in a string object. split() will do what the name says
    for i in range(len(ids)):
        qa_pairs = []
        if i == len(ids) - 1:
            break                                   # break operation if index is going backward or equal
        # .strip() will remove extra spaces
        first = remove_punctuation(lines_dict[ids[i]]).strip()          # accessing questions
        second = remove_punctuation(lines_dict[ids[i + 1]]).strip()     # accessing answers
        qa_pairs.append(first.split()[:max_len])              # trim the words upto max_len
        qa_pairs.append(second.split()[:max_len])
        # now qa_pairs is a 2D list with questions & answers as dimensions
        pairs.append(qa_pairs)

print(len(pairs))                                   # 221616 pairs of questions and answers

"""
Construct word to index dictionary for each word with unique index & PyTorch maps this index to one-hot vector & when passed this vector into an
embedding layer, PyTorch will provide respective word embeddings. Gather unique words in dataset & get their count. Remove less than 5 times words to
reduce vocabulary eventually decreasing number of weights in the output layer """
word_freq = Counter()                               # counts unique objects
for pair in pairs:
    word_freq.update(pair[0])                       # counts number of times each word in questions occur
    word_freq.update(pair[1])                       # counts number of times each word in answer occur

min_word_frequency = 5
words = [w for w in word_freq.keys() if word_freq[w] > min_word_frequency]       # keys are words & their values are the count as per above action
word_map = {k: v + 1 for v, k in enumerate(words)}  # enumerate generates index & it's object, then maps object to an index starting from 1
# add also start, end, padding & unknown tokens
word_map['<unk>'] = len(word_map) + 1               # giving last index for unknown words
word_map['<start>'] = len(word_map) + 1             # index for start token
word_map['<end>'] = len(word_map) + 1               # index for end token
word_map['<pad>'] = 0                               # index for padded sequence(with shorter than 5 words) can be 0 since word_map starts with index 1

print("Total words are {}".format(len(word_map)))

with open('Wordmap_corpus.json', 'w') as j:
    json.dump(word_map, j)

def encode_question(words, word_map):           # get will give value of argument from object it passed upon, else returns other argument passed
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + \
            [word_map.get(word, word_map['<unk>']) for word in words] + \
            [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c


pairs_encoded = []
for pair in pairs:
    qus = encode_question(pair[0], word_map)        # Remember that 1st list if for questions
    ans = encode_reply(pair[1], word_map)           # And 2nd list is for replies
    pairs_encoded.append([qus, ans])

with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)

