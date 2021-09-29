""" Implementing a custom art classifier using a CNN model tradined on art paintings dataset to compare and visualize its performance on different art classes """

# Import necessary modules
import matplotlib.pyplot as plt
import math
import numpy as np
import pathlib
import sys, shutil, time
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


# PyTorch specific
import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from PIL import ImageFile

# %matplotlib inline (If using Jupyter enviroment to make plots more readable)

""" First step Setting up training mode based on CUDA capability."""
device = "CPU"                              # sets default value
train_on_gpu = torch.cuda.is_available()    #returns True if CUDA enabled GPU is available

if train_on_gpu:
    print('CUDA is available! Training on GPU...\n')
    print(torch.cudaa.get_device_properties(0))
    device = 'cuda'

""" Utitlity functions """
# Retrieve list of files in a directory
def get_files_in_directory(path_to_dir, extension="*.*"):
    if not isinstance(path_to_dir,pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return list(path_to_dir.glob(extension))


# Retrieves the list of folders with a directory
def get_folders_in_directory(path_to_dir, prefix = ""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([fld for fld in path_to_dir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])

# Retrieves the list of folders with a directory
def get_folder_names_in_directory(path_to_dir, prefix =""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([fld.name for fld in path_to_dir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])

""" Dataset preparation as follows:

2. Prepare the dataset

The folder structure created in data_processing.py should looks like this:
```
dataset/train/artCategory_1/file_01.jpg
dataset/train/artCategory_1/file_03.jpg
dataset/train/artCategory_1/file_06.jpg
...
dataset/test/artCategory_1/file_02.jpg
...
dataset/valid/artCategory_1/file_04.jpg
```

The root folder for training is `dataset/train` and the classes are the names of art types.
Likewise, `dataset/valid` and `dataset/test` for validation and testing respectively.
"""

# Define data directories and location of image-sets archive

# Sets root folder for image sets
path_to_dataset = pathlib.Path.cwd().joinpath('..', 'dataset')
path_to_train = path_to_dataset.joinpath('train')
path_to_test = path_to_dataset.joinpath('test')
path_to_valid = path_to_dataset.joinpath('valid')

# count and list art category
art_categories = get_folder_names_in_directory(path_to_train, ".")  # collects the list of folders

print("Total no. of categories = ", len(art_categories))
print("Categories: ", art_categories)

""" Transformations (same as in transfer_learning.py

Let's assume that our model expects a `224`-dim square image as input. Resizing will therefore be required for each art image to fit this model. These transformations applies on `train`, `test` & `valid`.
Use PyTorch's [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) class, which makes it very easy to load data from a directory.

***
A. Data-Augmentation***
Using data augmentation on training images.
1. Randomly rotating by 30 degress.
2. Randomly cropping and resizing it to 224-dim square image.
!!! Should use only one function that crops the given image to random size and aspect ratio.
3. Randomly flipping it horizontally.

> ** Note:** 
Data augmentation isn't applied on validation and testing set. These images are resized to 256 pixels and then cropped from center to make it 224-dim square images.  

Normalization: 
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. Same normalization was used by these pre-trained models for training.
"""
# Define the data-augmentation transforms including normalisations
train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# load and apply above transforms on dataset using ImageFolder
train_dataset = datasets.ImageFolder(path_to_train, transform=train_transforms)
test_dataset = datasets.ImageFolder(path_to_test, transform=test_transforms)
valid_dataset = datasets.ImageFolder(path_to_valid, transform=valid_transforms)

# Print out data stats
print('Length of training images: ', len(train_dataset))
print('Length of testing images: ', len(test_dataset))
print('Length of validation images:', len(valid_dataset))

"""
Data Loading
A [data loader](https://pytorch.org/docs/stable/data.html) is an iterable over a dataset. The parameters are:
* `batch`:  number of samples per batch to be loaded
* `shuffle`: set to True, data are reshuffled at every epoch.
* `num_workers`: number of subprocesses to use for data loading
"""

# Define dataloader parameters
batch_size = 16                 # Might have to increase the size to 32. This might raise an exception
num_workers = 0

# Prepare data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = True)
test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size=batch_size, num_workers=num_workers, shuffle = False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = False)

# Print the batches stats
print('Number of  training  batches:', len(train_dataloader))
print('Number of  testing   batches:', len(test_dataloader))
print('Number of validation batches:', len(valid_dataloader))

# """ Optional data visualization. Doesn't effect model training or testing"""
#
# # Ignore normalization and turn shuffle ON to visualize different classes together
# visual_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
#
# # Load and apply above transforms on dataset using ImageFolder
# # Use test directory images can be used for visualization
# visual_dataset = datasets.ImageFolder(path_to_test ,transform=visual_transforms)
#
# # Prepare data loaders
# visualization_dataloader = torch.utils.data.DataLoader(visual_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#
# # Obtain one batch of testing images
# data_iter = iter(visualization_dataloader)
# images, labels = data_iter.next()
#
# # Convert images to numpy for display
# images = images.numpy()
#
# # Plot the images in the batch along with the corresponding labels
# plotCols = 4
# plotRows = math.ceil(batch_size/plotCols) # SqRoot could be used as well: math.ceil(math.sqrt(batch_size))
# fig = plt.figure(figsize=(25, 25))
# for idx in np.arange(batch_size):
#     ax = fig.add_subplot(plotRows, plotCols, idx+1, xticks=[], yticks=[])
#     plt.imshow(np.transpose(images[idx], (1, 2, 0)))
#     ax.set_title(art_categories[labels[idx]])

""" 
Modelling

3.1. Definition

Define a model based on a class inherited from nn.Module which will take in a 224x224 dimensional image.

1. Initialise a CNN model  
This class function (`__init()__`) defines the architecture or flow of the model.  

	* 5 convolutional layers configured as such:  
		* input/output filters: 3, 8, 16, 32, 64, respectively
		* filter size: 224*224, 112*112, 56*56, 28*28, 14*14

		Each layer uses a kernel size of 3x3, stride and padding value of 1.

	* Max pooling layer with a kernel size of 2*2 and stride as 2. This layer takes the output of convolutional layer and decreases the dimensionality by half.
	* 3 fully connected layers with an output of 1024 and 256, `x` nodes, respectively. `x` represents the number of art classes processed.
	
	The example below shows how to create a single layer.
For ex: `self.conv1 = nn.Conv2d(3,  8, 3, stride = 1, padding = 1)` where the arguments are in_depth, out_depth, kernel_size, stride, padding. Here `self.conv1` is user define variable which denotes the given layer. In this manner, we can create multiple layers and assign each one to its respective variable name.

2. Model Architecture/Flow  
This class function (`forward()`) defines the architecture or flow of the model. Simply, it performs forward propogation.
	* Using the layer variables created in the above init() function, provide a sequential pathway for the tensors to pass through the layers define earlier.
	* Provide the activation function applied on the layer.  For ex: `x = F.relu(self.conv1(x))`.

	The flow is as follow:
	* Set the `relu` activation function to each convolutional layer
	* Set a max pooling layer after each convolutional layer
	* Set the `relu` activation function and dropout to each fully connected layers.

"""

# Define CNN architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  # takes in 224*224
        self.conv2 = nn.Conv2d(8, 24, 3, stride=1, padding=1)  # takes in 112*112
        self.conv3 = nn.Conv2d(24, 72, 3, stride=1, padding=1)  # takes in 56*56
        self.conv4 = nn.Conv2d(72, 144, 3, stride=1, padding=1)  # takes in 28*28
        self.conv5 = nn.Conv2d(144, 288, 3, stride=1, padding=1)  # takes in 14*14

        self.maxpool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(288 * 7 * 7, 4096)  # input to fc1 is (output filters of conv. part = 64) * (dimensions of each = 14*14)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 14)

    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))  # takes 224*224
        x = self.maxpool(x)  # gives 112*112
        x = F.relu(self.conv2(x))  # takes 112*112
        x = self.maxpool(x)  # gives 56*56
        x = F.relu(self.conv3(x))  # takes 56*56
        x = self.maxpool(x)  # gives 28*28
        x = F.relu(self.conv4(x))  # takes 28*28
        x = self.maxpool(x)  # gives 14*14
        x = F.relu(self.conv5(x))  # takes 14*14
        x = self.maxpool(x)  # gives 7*7

        x = x.view(-1, 288 * 7 * 7)  # flattening output of convolutional part

        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)  # dropout and activation function is not used on last layer

        return x


# instantiate CNN
model_scratch = Net()

# Move model to GPU if CUDA is available
if train_on_gpu == True:
    model_scratch.to(device)

print(model_scratch)

# Set model name and location
model_name = 'customised'
model_filename = 'trained_' + model_name + '.pt'
path_to_model = pathlib.Path.cwd().joinpath('..', 'models', model_filename)
print('File name for saved model: ', path_to_model)

"""
# Specify loss function and optimizer
Let's define a criterion and an optimizer that will work together. The criterion will be used to stop the algorithm when it approaches the optimum. We will use ADAM algorithm as optimizer and use
Cross-entropy loss function as criterion. Note that the optimizer accepts as input only the trainable parameters.
"""
# Select loss function
criterion = nn.CrossEntropyLoss()
# Select optimizer
optimizer = optim.Adam(model_scratch.parameters(), lr=0.001)

""" Model training
 Training prep as follows: Number of epochs, lower bound for validation loss and performance variables such as losses for training and validation, model accuracy.
"""

# Some images in dataset were truncated (maybe corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set number of epochs to train the model
n_epochs = 40

# Initialize tracker for minimum validation loss
valid_loss_min = np.Inf   # set initial "min" to infinity

# Initialise performances
train_losses, valid_losses, accuracies=[],[],[]
training_loss = 0.0
validation_loss = 0.0
accuracy = 0.0

a = time.time()  # Start-time for training

for epoch in range(1, n_epochs + 1):
    c = time.time()  # Start-time for epoch

    ###############
    # TRAIN MODEL #
    ###############

    # model by default is set to train
    for batch_i, (images, labels) in enumerate(train_dataloader):  # Getting one batch of training images and their corresponding true labels

        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            images = images.to(device)
            labels = labels.to(device)

        # clear the previous/buffer gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model_scratch.forward(images)

        # calculate the batch loss
        loss = criterion(outputs, labels)  # (y_hat, y)  or (our-prediction, true-label)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update training loss
        training_loss += loss.item()

    ##################
    # VALIDATE MODEL #
    ##################

    # validation loss and accuracy
    validation_loss = 0.0
    accuracy = 0.0

    model_scratch.eval()  # model is put to evaluation mode i.e. dropout is switched off

    with torch.no_grad():  # Turning off calculation of gradients (not required for validation and saves time)
        for images, labels in valid_dataloader:  # Getting one batch of validation images

            if train_on_gpu:  # moving data to GPU if available
                images = images.to(device)
                labels = labels.to(device)

            outputs = model_scratch.forward(images)

            # calculate the batch loss
            batch_loss = criterion(outputs, labels)
            validation_loss += batch_loss.item()

            # Calculating accuracy
            ps = torch.exp(outputs)  # Turning raw output values into probabilities using exponential function

            # getting top one probablilty and its corresponding class for batch of images
            top_p, top_class = ps.topk(1, dim=1)

            # Comparing our predictions to true labels
            equals = top_class == labels.view(*top_class.shape)  # equals is a list of values

            # incrementing values of 'accuracy' with equals
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # taking average of equals will give number of true-predictions
            # equals if of ByteTensor (boolean), changing it to FloatTensor for taking mean...

    train_losses.append(training_loss / len(train_dataloader))
    valid_losses.append(validation_loss / len(valid_dataloader))
    accuracies.append(((accuracy / len(valid_dataloader)) * 100.0))
    d = time.time()  # end-time for epoch

    print(f"Epoch {epoch} "
          f"Time: {int((d - c) / 60)} min {int(d - c) % 60} sec "
          f"Train loss: {training_loss / len(train_dataloader):.2f}.. "
          f"Validation loss: {validation_loss / len(valid_dataloader):.2f}.. "
          f"Validation accuracy: {((accuracy / len(valid_dataloader)) * 100.0):.2f}% "
          )

    training_loss = 0.0

    # save model if validation loss has decreased
    if (validation_loss / len(valid_dataloader) <= valid_loss_min):
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, validation_loss / len(valid_dataloader)))

        # saving model
        torch.save(model_scratch.state_dict(), path_to_model)

        # update minimum validation loss
        valid_loss_min = validation_loss / len(valid_dataloader)

    # After validation, model is put to training mode i.e. dropout is again switched on
    model_scratch.train()

    ################
    # END OF EPOCH #
    ################

b = time.time()  # end-time for training
print('\n\n\tTotal training time: ', int((b - a) / (60 * 60)), "hour(s) ", int(((b - a) % (60 * 60)) / 60), "minute(s) ", int(((b - a) % (60 * 60)) % 60), "second(s)")

# Load the model with lowest validation loss
model_scratch.load_state_dict(torch.load(path_to_model))

""" Testing preparation : params same as training (number of epochs, lower bound for validation loss, performance variables such as losses & model accuracy"""
test_loss = 0.0
counter = 0

class_correct = list(0. for i in range(len(art_categories)))
class_total = list(0. for i in range(len(art_categories)))
classes_accuracies=[]

# evaluation mode (switching off dropout)
model_scratch.eval()

y_true = []
y_pred = []

# iterate over test data - get one batch of data from test loader
for images, labels in test_dataloader:

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        images = images.to(device)
        labels = labels.to(device)

    # compute predicted outputs by passing inputs to the model
    output = model_scratch(images)

    # calculate the batch loss
    loss = criterion(output, labels)

    # update test loss
    test_loss += loss.item() * images.size(0)

    # Convert output probabilities to predicted class
    ps, pred = torch.max(output, 1)

    # Compare model's predictions to true labels
    for i in range(len(images)):
        y_true.append(art_categories[labels[i]])
        y_pred.append(art_categories[pred[i]])

        class_total[labels[i]] += 1
        if pred[i] == labels[i]:
            class_correct[pred[i]] += 1
    counter += 1

# calculate average test loss
test_loss = test_loss / len(test_dataloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(art_categories)):
    classes_accuracies.append(100 * class_correct[i] / class_total[i])
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            art_categories[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (art_categories[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %
      (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


""" Confusion matrix for performance evaluation"""

array = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(array, index = [i for i in art_categories],
                  columns = [i for i in art_categories])
plt.figure(figsize = (10,7))
mx = sn.heatmap(df_cm/np.sum(df_cm), annot=True, fmt='.2%', cmap='Blues')
del df_cm

""" Saving the best model """

checkpoint = {'training_losses': train_losses,
              'valid_losses': valid_losses,
              'accuracies': accuracies,
              'classes_accuracies':classes_accuracies,
              'state_dict': model_scratch.state_dict()}

torch.save(checkpoint, path_to_model)

""" This concludes custom training of art classifier in contrast to using pre-trained models in transfer_learning.py
As explained, problem is pre-trained models are trained upon image datasets. Finding a good generalization on our painting dataset is not a trivial task and poor performance is imperative
"""

