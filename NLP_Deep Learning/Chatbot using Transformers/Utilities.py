import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import json

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


# Creating masks for input ( questions ) & output ( replies )
def create_masks(question, reply_input, reply_target):
    """ Following are arguments
    sentence: <start> I slept last night <end>
    reply_input: <start> I slept last night
    reply_target: I slept last night <end>
    Here, replY_input is input to encoder & reply_target is input to decoder.
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




















