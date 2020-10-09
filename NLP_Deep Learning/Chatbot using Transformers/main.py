""" Coding a chat-bot using Transformers neural networks from scratch. We can also use Transformer class from PyTorch. Coding here for a deeper
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Creating a Dataset class for operations on dataset
class Dataset(Dataset):

    def __init__(self):
        self.pairs = json.load(open('pairs_encoded.json'))      # Open json file to a variable
        self.dataset_size = len(self.pairs)

    def __getitem__(self, index):
        """ Retrieves a sample from dataset """
        """LongTensor converts discrete values into float64 dtypes. Most of the CPUs are x64 architecture. So, this optimizes for computation."""
        # Also encoded values have wide range of numbers(bits). Hence, it's better to encode in LongTensor
        question = torch.LongTensor(self.pairs[index][0])
        reply = torch.LongTensor(self.pairs[index][1])
        # Retrieve one pair per dataset. Later we shall pass in batches, with n pairs.
        return question, reply

    def __len(self):
        return self.dataset_size


train_loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=100, shuffle=True, pin_memory=True)

question, reply = next(iter(train_loader))

# question.shape is torch.Size([100, 25]) 100 is batch_size, 25 are maximum words in a sentence
# reply.shape is torch.Size([100, 27]) same as above, but 27 is including start & end tokens too


# Creating masks for input ( questions ) & output ( replies ). Note that transformer's decoder shouldn't see future words, hence masking.
def create_masks(question, reply_input, reply_target):
    """ Following are arguments
    sentence: <start> I slept last night <end>
    reply_input: <start> I slept last night
    reply_target: I slept last night <end>
    Here, reply_input is input to encoder & reply_target is input to decoder.
    Word 'I' predicts 'Slept', 'Slept' predicts 'last' & so on while feeding into a decoder"""
    # Subsequent masks will mask future positions of the sentence
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)  # Generates upper triangular elements of ones & transposes them as integers, then unsqueeze into 1 dimension
        # Unsqueeze 4D into 1D tensors. PyTorch will return 1s where there are elements to be masked, else 0s if the sentence is shorter & padded

    question_mask = (question != 0).to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words) unsqueeze two times

    reply_input_mask = reply_input != 0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    # before unsqueezing again, we've to combine('&') with subsequent masks because the decoder will mask not only padded words, but also future words
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data)
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target != 0  # (batch_size, max_words)
    # we don't have to unsqueeze reply_target as its for only loss calculation, not for attention
    return question_mask, reply_input_mask, reply_target_mask


# Generating word embeddings and positional encoding of those embeddings to feed into encoder of the transformer
class Embeddings(nn.Module):
    """ Word embeddings and their positional encoding"""

    def __init__(self, vocabulary_size, d_model, max_len=50):
        super(Embeddings, self).__init__()
        self.d_model = d_model                        # dimensionality of the model (# of features to represent)
        self.dropout = nn.Dropout(0.1)                              # regularization with 0.1 probability
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_len=max_len, d_model=d_model)


    def create_positional_encoding(self, max_len, d_model):
        """ Initialize a matrix of zeros"""
        positional_encoding = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):                      # for each position of the word
            for i in range(0, d_model, 2):              # for each dimension of each position
                positional_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        positional_encoding = positional_encoding.unsqueeze(0)      # this will add a dimension in 0(first dimension) this is for batch_size
        # Now this encoded matrix is in the form of (1, max_len, d_model). Here batch_size = 1, but will automatically expand when this matrix is
        # added to word embeddings, with same batch_size as encoded words.
        return positional_encoding

    def forward(self, encoded_words):
        """ Forward propagation """
        embeddings = self.embed(encoded_words) * math.sqrt(self.d_model)    # (batch_sie, max_len, d_model)
        # multiplying postionally encoded embeddings with sqrt of dimensionality to give more meanings to the embeddings
        """Following will trim positional encoding matrix to max_words = embeddings.size(1) and expand to same batch_size as embeddings
        """
        embeddings += self.pe[:, :embeddings.size(1)]
        embeddings = self.dropout(embeddings)
        return embeddings


""" Implementation of Multi-head attention. This applies to both encoder and decoder transformers. Encoder has self-attention, whereas decoder has
source attention and self attention."""


class MultiHeadAttention(nn.Module):
    """ Instantiates multi-head attention"""
    def __init__(self, heads, d_model):
        """
        :param heads: # of heads in a transformer (series of keys) should be a factor of dimensions we pass in
        :param d_model:
        """
        super(MultiHeadAttention, self).__init__()
        # Raise an assertion error if below condition is false
        assert d_model % heads == 0
        # Global variables or class variables
        self.d_k = d_model // heads     # dimensionality of heads
        self.dropout = nn.Dropout(0.1)  # as per the paper
        self.query = nn.Linear(d_model, d_model)        # just a fancy representation of dividing embeddings into heads for processing
        self.key = nn.Linear(d_model, d_model)          # all keys, values, queries are same. Just that they originate in different portions
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)       # we concatenate all the heads, so dimensions of this layer will also be same as embedding dim

    def forward(self, key, value, query, mask):
        """
        :param key: embeddings from encoder   ; shape: (batch_size, max_words, dimensionality of model=512)
        :param value: embeddings from encoder ; shape: (batch_size, max_words, dimensionality of model=512)
        :param query: embeddings from decoder ; shape: (batch_size, max_words, dimensionality of model=512)
        :param mask: applied to self attention(encoder), source-attention(decoder), masked_multi-head attention(decoder a.k.a self-attention)
        :param mask: shape of mask is as per above create_masks(), as (batch_size, 1, 1, max_words)
        :return: forward propagated pass
        In attention, we find similarity between queries and keys to generate weight matrix using softmax and multiply those weights with values
        In self-attention, all keys, values, queries come from model inputs
        In source-attention (inside decoder), values are encoded representation from encoder
        """
        # Project q, k, v using their linear layers. Shape stays the same
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        # Now reshape these according to dimensions of heads
        # Transform (batch_size, max_len, 512) --> (batch_size, max_len, 8, 64) --> (batch_size, number of heads, max_len, d_k=64)
        # Look carefully, we just broken down 512 to 8 heads of 64 dimensions then transposed along heads to match our shape
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)    # Transposed (swapped 2nd&3rd dimensions) 4D matrix
        key = key.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)    # Same for keys, all of them are same shape anyway
        value = value.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)    # Same for values, so not changing query part inside

        # (batch_size, 8, max_len, d_k) dot (batch_size, 8, d_k, max_len) --> (batch_size, 8, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))  # Can use sqrt(self.d_k) as well

        # Mask padded elements
        scores = scores.masked_fill(mask == 0, -1e9)        # fills a value we specify at given condition. Softmax avoids very tiny negative values.
        """
        Self-attention computes softmax for its own inputs (last argument - max_len)
        Source-attention computes softmax for same max_len, but before last argument one
        """
        # Compute softmax
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Now multiply our soft-maxed weights with values
        # (batch_size, 8, max_len, max_len) dot (batch_size, 8, max_len, d_k) --> (batch_size, 8, max_len, d_k)
        context = torch.matmul(weights, value)

        # Concatenate the attentions
        # batch_size, 8, max_len, d_K) --> (batch_size, max_len, 8, d_k) --> (batch_size, max_len, 8 * d_k)
        context = context.permute(0, 2, 1, 3).view(context.shape[0], -1, self.heads * self.d_k)
        interacted = self.concat(context)

        return interacted

""" This concludes the most important part of transformers, computing multi-head attentions. Following is construction of Feed-Forward NN"""

class FeedForward(nn.Module):

    def __init__(self, d_model, middle_dimension=2048):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model,middle_dimension)
        self.fc2 = nn.Linear(middle_dimension, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        :param x: output of multi-head attention for this feed-forward layers
        :return: output after ReLu & dropout on linear layers
        """
        out = F.relu(self.fc1(x))       # Paper used Relu activation in 1st linear layer
        out = self.fc2(self.dropout(out))
        return out


""" Assembly of everything we have created so far. Multi-head attention modules & Feed-Forward modules to create Encoder and Decoder and also apply
layer normalization after concatenating the attentions."""


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.self_multihead = MultiHeadAttention(heads=heads, d_model=d_model)      # Encoder has only self attention
        self.feed_forward = FeedForward(d_model=d_model, middle_dimension=2048)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, mask):
        """
        :param embeddings: input embeddings after positional encoding
        :param mask: source mask or question mask as defined above
        :return: concatenation of all the attention heads with projection
        """
        interacted = self.self_multihead(key=embeddings, value=embeddings, query=embeddings, mask=mask)
        interacted = self.layernorm(interacted + embeddings)

        # Applying layer normalisation for both attention & Feed Forward blocks
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)

        return encoded


class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, encoded, src_mask, target_mask):
        query = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded


class Transformer(nn.Module):

    def __init__(self, d_model, heads, num_layers, word_map):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.vocab_size = len(word_map)
        self.embed = Embeddings(self.vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)

    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings

    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim=2)
        return out

# Optimization settings

class AdamWarmup:

    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0

    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))

    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()


class LossWithLS(nn.Module):

    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)  # (batch_size * max_words)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)  # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss

""" Define the model, set the hyper parameters, train the model"""

# Hyperparameters

d_model = 512
heads = 8
num_layers = 1              # Paper uses 6 layers of transformers. Using lower configuration here.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1                  # Paper uses 25 epochs.

with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)

transformer = Transformer(d_model=d_model, heads=heads, num_layers=num_layers, word_map=word_map)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
criterion = LossWithLS(len(word_map), 0.1)


def train(train_loader, transformer, criterion, epoch):
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):

        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare Target Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create mask and add dimensions
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)

        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        sum_loss += loss.item() * samples
        count += samples

        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss / count))



# Evaluation


def evaluate(transformer, question, question_mask, max_len, word_map):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim=1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)  # (1,step+2)

    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    return sentence


for epoch in range(epochs):
    train(train_loader, transformer, criterion, epoch)

    state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
    torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')


load_checkpoint = True
ckpt_path = 'checkpoint.pth.tar'

if load_checkpoint:
    checkpoint = torch.load(ckpt_path)
    transformer = checkpoint['transformer']

while (1):
    question = input("Question: ")
    if question == 'quit':
        break
    max_len = input("Maximum Reply Length: ")
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
    question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)           # All un-padded words are unsqueezed into mask each of ( src, self)
    sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
    print(sentence)

