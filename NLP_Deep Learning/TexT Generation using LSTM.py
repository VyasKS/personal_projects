""" This python file contains source code for generating text given a text data for training where the model used is Long Short Term Memory networks.
This will generate text based on training data that we train the network upon. The data used is the book called 'Alice in Wonderland' by author Lewis
Carroll. One can use any text to generate a new body of text with the help of LSTM model, an improvement of Recurrent Neural Networks to overcome the
problem of vanishing gradients. We use a powerful deep learning framework called PyTorch. This project is dated on 6th of September 2020. Let us have
some fun. Friendly disclaimer --> [We have few names shadowed inside a class, method from outer scope, to understand better. Remember that names will
be born & destroyed inside a class, function etc.. So, not a big deal as long as we understand this to deal with Name Error if at all arises]"""
# Have path variable that defines the path of dataset of your choice to run this file
# Importing necessary libraries

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Dictionary(object):
    """ Stores the indices and words for their words and indices respectively. Also displays their length when called upon"""

    def __init__(self):
        self.word_to_idx = {}  # maps unique words to index
        self.idx_to_word = {}  # maps indices to unique words
        self.idx = 0           # acts like a counter

    def add_word(self, word):
        """ Adds a unique word to the dictionary word_to_idx"""
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.idx  # add if word is unique & set latest index from above
            self.idx_to_word[self.idx] = word  # map the same index to word
            self.idx += 1                      # counter will be incremented by 1

    def __len__(self):
        return len(self.word_to_idx)  # gets length of dictionary when called upon


class TextProcess(object):
    """ Preprocesses text data for modeling"""

    def __init__(self):
        self.dictionary = Dictionary()  # creating an object of dictionary here that we created above

    def get_data(self, path, batch_size=20):  # by default batch_size=20, renamed to size, to avoid shadowing outer scope
        """ Goes to a path and adds words to a dictionary and makes a tensor representation of tokens"""
        with open(path, 'r') as f:                  # reads data from the given path as some variable (f)
            tokens = 0                              # initializing no. of tokens as 0
            for line in f:                          # loop over every line in the text file
                words = line.split() + ['<eos>']    # python by default splits at white space. ['<eos>'] tells that its an end of sentence
                tokens += len(words)                # tokens will be updated as per word split
                for word in words:
                    self.dictionary.add_word(word)  # adds such words into a dictionary created in above class
        # We have tokenized all words from text file and stored words in a dictionary so far
        # Now, we need tensor of indices to lookup in lookup table for word embeddings
        # Create a 1 D tensor that contains the indices of all the words in the text file
        tensor_representation = torch.LongTensor(tokens)  # Tensor shape depends on size of 'tokens'. LongTensor is needed here as large text data.
        index = 0                                         # Initialize index as 0
        with open(path, 'r') as f:                        # Same, reads text data from given path
            for line in f:                                # loop over every line in text file
                words = line.split() + ['<eos>']          # split at white spaces and add an end of sentence token
                for word in words:                        # loop over all words
                    tensor_representation[index] = self.dictionary.word_to_idx[word]  # add the index of word by calling word_to_idx from dict class
                    index += 1                            # increment index by 1
        # For splitting data into batches, we can use DataLoader from PyTorch. But, we do manual implementation
        # Check number of batches needed for splitting data into batches
        num_batches = tensor_representation.shape[0] // batch_size  # ignore remainder
        # Ignore data that is left over
        tensor_representation = tensor_representation[:num_batches * batch_size]  # trims out left over data from above division
        # reshape it to (batch_size, num_batches) and return it
        tensor_representation = tensor_representation.view(batch_size, -1)  # -1 in PyTorch automatically checks for dimensions & display them
        return tensor_representation


# Defining hyper parameters
embedding_size = 128  # Input features (vectors) to the LSTM
hidden_size = 1024  # no. of LSTM units (hidden dimensions) when stacking them together
num_layers = 1  # 1 layered LSTM
num_epochs = 20  # how many times we show the entire data to the model
batch_size = 20  # in iterations of batches
time_steps = 30  # (no. of copies of LSTM models) We'll look 30 previous words to predict 31st word
learning_rate = 0.002  # self explanatory - learning_rate for optimization

# create an object of TextProcess class
corpus = TextProcess()

# tensor representation
in_tensors = corpus.get_data('alice.txt', batch_size)
print(in_tensors.shape)  # Original data has 1659 words for each row, but after trimming left over words in batching, we're left with 1484

vocabulary_size = len(corpus.dictionary)  # nested class inside an object. Prints no. of unique words in text data

num_batches = in_tensors.shape[1] // time_steps  # for 20 rows, (1484 words // 30). We're splitting every 30 words as 1 batch
print(num_batches)


class TextGenerator(nn.Module):
    """ Class to generate text after training. Inherits from PyTorch's nn.Module class"""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """ Initializes word embeddings, LSTM network, output layer with inputs from LSTM"""
        super(TextGenerator, self).__init__()  # Class inheritance
        self.embedding = nn.Embedding(vocab_size, embed_size)  # word embeddings
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM layers
        # If batch_first=True will give outputs same as inputs (batch_size*time_steps*embedding size), else then time_steps*batch_size*embedding_size
        self.linear = nn.Linear(hidden_size, vocab_size)  # Output layer

    def forward(self, x, h):
        """ Performs word embedding, forward propogation & generates output after decoding vectors to words back again"""
        x = self.embedding(x)
        # x = x.view(batch_size, time_steps, embedding_size)  # reshape input tensor # We specified batch_first=True, so not this line not required
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))  # reshape output from (samples, time_steps, output_features) to a shape appropriate
        # for the fully connected layer (FC) to feed into output layer as (batch_size * time_steps, hidden_size)
        out = self.linear(out)  # Decode hidden states from all time_steps into words
        return out, (h, c)


# Calling out methods for modeling
model = TextGenerator(vocab_size=vocabulary_size, embed_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

# Defining Cross Entropy loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer for convergence
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

""" If we have a tensor (z), z.detach() method will detaches it from its accumulated properties and returns a tensor that shares the same storage as 
previous (z), with deleted computational history. It doesn't know anything about how it was computed. In other words, we strip away the tensor (z)
from its past history. Following code performs truncated Backpropogation through time (TBTT). TBTT splits 1000-long sequence into some (for eg.50)
sequences each of length (for eg.20) and treats each such sequence as a separate training case. Kind of batch_size interpretation, but not literally.
This is a sensible approach that works well in practice, but is blind to temporal dependencies that span more than 20 time steps."""


def detach(states):
    return [state.detach() for state in states]


# Training
for epoch in range(num_epochs):
    # Initialize zero weights for hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size))

    for i in range(0, in_tensors.size(1) - time_steps, time_steps):
        # Generate mini-batch inputs and targets
        inputs = in_tensors[:, i: i + time_steps]
        targets = in_tensors[:, (i+1): (i+1) + time_steps]
        # This is what is happening in the background as an example of 12 timesteps
        # String : Black horse is here
        # input: Black horse, Output: lack horse i
        # Output here is a delay of input by 1 factor (character here)
        # So, we don't see 'B' in output and see extra (i) & this becomes input for the next & so on..
        outputs, _ = model(inputs, states)  # not interested in states it generates. Hence '_' was used
        loss = loss_function(outputs, targets.reshape(-1))  # have to change target to 1 dimension when using CrossEntropyLoss()
        # -1 tells PyTorch to infer the proper dimensions

        model.zero_grad()  # forget the gradients to avoid accumulation
        loss.backward()

        # Perform gradient clipping to a value (float or int) that is maximum allowed magnitude for gradient
        # Clipped in the range [ -clip_value, clip_value ] (min & max) to prevent Exploding Gradients Problem
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // time_steps
        if step % 100 == 0:
            print('Epoch [{} / {}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        """ Here, we didn't move our model to a GPU. We run tensors on CPU. CPU is fine if dealing with smaller text datasets. Not exactly fine, but 
        be managed in the absence of a GPU."""

# Testing
with torch.no_grad():  # No back propogation happens during testing
    with open('results.txt', 'w') as f:  # we open this file & write (w) it as 'results.txt'
        # Initializing hidden and cell states as zeros
        state = (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))

        # Random selection of one word ID and shaping into (1,1)
        input = torch.randint(0, vocabulary_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            output, _ = model(input, state)
            print(output.shape)
            # Sample a word ID from exponential of output to get probability
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            # Replace input with sampled word_id for next time step
            input.fill_(word_id)

            # Write results to a file
            word = corpus.dictionary.idx_to_word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print(' Sampled [{} / {}] words and save to {}'.format(i+1, 500, 'results.txt'))

""" For better results, use over 500 epochs, and a decay for learning rate, multiple layers of LSTM such as 2 or 3 instead of just 1
Also, use Dropout Regularization to avoid overfitting. These steps will eventually produce better results.

 --------This marks the end of the project-------"""
