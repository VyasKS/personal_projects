import torch
from torch.utils.data import Dataset

train_loader = torch.utils.data.DataLoader(dataset=Dataset(), batch_size=100, shuffle=True, pin_memory=True)

