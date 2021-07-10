# Import dependencies
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import pathlib
import sys, shutil, time
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from PIL import ImageFile


# Set training mode based on  CUDA capability
device = 'cpu'
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('CUDA is available! Training on GPU....\n')
    print(torch.cuda.get_device_properties(0))
    device = 'cuda'
else:
    print('CUDA is not available! Training on CPU....\n')


# Define utility functions for data retrieval
def get_files_in_directory(path_to_dir, extension='*.*'):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)

    return list(path_to_dir.glob(extension))


# Retrieve list of folders within a directory
def get_folders_in_directory(path_to_dir, prefix=""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([folder for folder in path_to_dir.iterdir() if folder.is_dir() and not folder.name.lower().startswith(prefix)])


# Retrieve list of folder names within a directory
def get_folders_names_in_directory(path_to_dir, prefix=""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)
    return sorted([folder.name for folder in path_to_dir.iterdir() if folder.is_dir() and not folder.name.lower().startswith(prefix)])


# Dataset roots {train: dataset/train, test:dataset/test, valid:dataset/valid}
path_to_dataset = pathlib.Path.cwd().joinpath('..', 'dataset')
path_to_train = path_to_dataset.joinpath('train')
path_to_test = path_to_dataset.joinpath('test')
path_to_valid = path_to_dataset.joinpath('valid')

# Folders as a list to count
categories = get_folders_names_in_directory(path_to_train, '.')
print(f'Total number of categories are {len(categories)}\n'
      f'Categories are {categories}')

"""
 Transformations : Let's assume that model expects a 224 dimension square image as input.Therefore, resizing is required
  for each art image to fit this model
    Data Augmentation is done by rotating training set by 30 degrees, randomly cropping and resizing to 224 dim-square
    image and flipping it horizontally. Data augmentation isn't applied on validation and testing set. These images are 
    resized to 256 pixels and then cropped from center to make it 224-dim square images
    Normalization of pixels within a range of [0,1] to avoid uneven preference of the network when updating weights.

"""
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

# Load and apply transforms on dataset using ImageFolder
train_dataset = datasets.ImageFolder(path_to_train, transform=train_transforms)
test_dataset = datasets.ImageFolder(path_to_test, transform=test_transforms)
valid_dataset = datasets.ImageFolder(path_to_valid, transform=valid_transforms)

# Print out data stats
# print('Training  images: ', len(train_dataset))
# print('Testing   images: ', len(test_dataset))
# print('Validation images:', len(valid_dataset))

""" 
Data Loader: Iterable over a dataset with parameters (batch size, shuffle=True/False, num_workers for loading data)

"""
# Define dataloader parameters
batch_size = 16  # You might want to increase the size to 32. This might raise an exception
num_workers = 0

# Prepare data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Print the batches stats
print('Number of  training  batches:', len(train_dataloader))
print('Number of  testing   batches:', len(test_dataloader))
print('Number of validation batches:', len(valid_dataloader))
""" Following is optional data visualization"""
# Ignore normalization and turn shuffle ON to visualize different classes together
visual_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])

# Load and apply above transforms on dataset using ImageFolder
# Use test directory images can be used for visualization
visual_dataset = datasets.ImageFolder(path_to_test, transform=visual_transforms)

# Prepare data loaders
visualization_dataloader = torch.utils.data.DataLoader(visual_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# Obtain one batch of testing images
data_iter = iter(visualization_dataloader)
images, labels = data_iter.next()

# Convert images to numpy for display
images = images.numpy()

# Plot the images in the batch along with the corresponding labels
plotCols = 4
plotRows = math.ceil(batch_size/plotCols) # SqRoot could be used as well: math.ceil(math.sqrt(batch_size))
fig = plt.figure(figsize=(25, 25))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(plotRows, plotCols, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(categories[labels[idx]])

"""" 
Modelling: 
Several pre-trained models are available and can be used directly (`resnet50`, `alextnet` and `vgg19`)
Download, load & set different parameters. Model will be fine-tuned to fit specific requirements related to art classification.
"""
# Select the model
model_name = 'resnet50'  # 'alexnet', 'resnet50', 'vgg19'
model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
print(model)

# Freeze parameters to make network act as a fixed feature extractor. Feature parameters don't change during training for the pre-trained part.
# Freeze training of all feature layers
if model_name == 'resnet50':  # 'alexnet', 'resnet50', 'vgg19'
    for param in model.parameters():
        param.requires_grad = False
else:
    for param in model.features.parameters():
        param.requires_grad = False
"""
Fine-tuning :
Replace the last layer with a custom linear classifier layer with desirable output nodes
After having the pre-trained feature extractor, modifying final-fully-connected classifier layer
    > This layer should produce an appropriate number of outputs for this painting classification task.
    Access any layer in a pre-trained network by name and (sometimes) number.
    * For Alexnet/Vgg19: model`.classifier[6]` is the sixth layer in a group of layers named "classifier".
    * For ResNet: `resnet50.fc` is the only fully connected Linear layer.
"""
# Input features
n_inputs = None
last_layer = None

if model_name == 'resnet50':  # ResNet50 model
    print(model.fc.in_features)
    print(model.fc.out_features)
    n_inputs = model.fc.in_features
    last_layer = nn.Linear(n_inputs, len(categories))  # Add last linear layer (n_inputs --to---> n painting classes)
    model.fc = last_layer
else:  # AlexNet, Vgg19
    print(model.classifier[6].in_features)
    print(model.classifier[6].out_features)
    n_inputs = model.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(categories))  # Add last linear layer (n_inputs --to---> n painting classes)
    model.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    model = model.to(device)

# Print the new model architecture
print(model)

# Check to see that your last layer produces the expected number of outputs
if model_name == 'resnet50':
    print(model.fc.out_features)
else:
    print(model.classifier[6].out_features)

# Set model name
model_filename = 'trained_' + model_name + '.pt'
path_to_model = pathlib.Path.cwd().joinpath('..', 'models', model_filename)
print('File name for saved model: ', path_to_model)

"""
Loss function & Optimizer:
Cross-entropy loss and stochastic gradient descent. Optimizer accepts only trainable parameters.
AlexNet is made up of 2 parts->  Features (Convolutional and max-pooling layers) and Classifier (Fully-connected layers).
In transfer-learning, only the Classifier part is trained.
"""
criterion = nn.CrossEntropyLoss()

# optimizer and learning rate
if model_name == 'resnet50':
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

"""
Model training:
number of epochs to train, minimum value for validation loss to converge at, performance variables - loss & accuracy are defined
"""
# Some images in dataset were truncated (maybe corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set number of epochs to train the model
n_epochs = 40

# Initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

# Initialise performances
train_losses, valid_losses, accuracies = [], [], []
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

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model.forward(images)

        # calculate the batch loss
        loss = criterion(outputs, labels)  # (y_hat, y)  or (our-prediction, true-label)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # clear the previous/buffer gradients of all optimized variables
        optimizer.zero_grad()

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

    # model is put to evaluation mode i.e. dropout is switched off
    model.eval()

    # turning off calculation of gradients (not required for validation)  {saves time}
    with torch.no_grad():

        # getting one batch of validation images
        for images, labels in valid_dataloader:

            # moving data to GPU if available
            if train_on_gpu:
                images = images.to(device)
                labels = labels.to(device)

            outputs = model.forward(images)

            # calculate the batch loss
            batch_loss = criterion(outputs, labels)
            validation_loss += batch_loss.item()

            # calculating accuracy - turn raw output values into probabilities using exponential function
            ps = torch.exp(outputs)

            # getting top one probability and its corresponding class for batch of images
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

    # Set epoch end-time
    d = time.time()

    print(f"Epoch {epoch} "
          f"Time: {int((d - c) / 60)} min {int(d - c) % 60} sec "
          f"Train loss: {training_loss / len(train_dataloader):.2f}.. "
          f"Validation loss: {validation_loss / len(valid_dataloader):.2f}.. "
          f"Validation accuracy: {((accuracy / len(valid_dataloader)) * 100.0):.2f}% "
          )

    training_loss = 0.0

    # save model if validation loss has decreased
    if validation_loss / len(valid_dataloader) <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, validation_loss / len(valid_dataloader)))

        # saving model
        torch.save(model.state_dict(), path_to_model)

        # update minimum validation loss
        valid_loss_min = validation_loss / len(valid_dataloader)

    # after validation, model is put to training mode i.e. dropout is again switched on
    model.train()

    ################
    # END OF EPOCH #
    ################

# end-time for training
b = time.time()  # end-time for training
print('\n\n\tTotal training time: ', int((b - a) / (60 * 60)),
      "hour(s) ", int(((b - a) % (60 * 60)) / 60),
      "minute(s) ", int(((b - a) % (60 * 60)) % 60),
      "second(s)")

# Load model with lowest validation loss
model.load_state_dict(torch.load(path_to_model))

""" Testing: same as training, setting params & performance variables """
test_loss = 0.0
counter = 0
class_correct = list(0. for i in range(len(categories)))
class_total = list(0. for i in range(len(categories)))
classes_accuracies = []

# evaluation mode (switching off dropout)
model.eval()

# Send a set of images to model, collect output, evaluate loss & calculate test accuracy
y_true = []
y_pred = []

# iterate over test data - get one batch of data from testloader
for images, labels in test_dataloader:

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        images = images.to(device)
        labels = labels.to(device)

    # compute predicted outputs by passing inputs to the model
    output = model(images)

    # calculate the batch loss
    loss = criterion(output, labels)

    # update test loss
    test_loss += loss.item() * images.size(0)

    # convert output probabilities to predicted class
    ps, pred = torch.max(output, 1)

    # compare model's predictions to true labels
    for i in range(len(images)):
        y_true.append(artCategories[labels[i]])
        y_pred.append(artCategories[pred[i]])

        class_total[labels[i]] += 1
        if pred[i] == labels[i]:
            class_correct[pred[i]] += 1
    counter += 1

# calculate avg test loss
test_loss = test_loss / len(test_dataloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(categories)):
    classes_accuracies.append(100 * class_correct[i] / class_total[i])
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            categories[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (categories[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' %
      (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))

# Confusion matrix
array = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(array, index=[i for i in categories], columns=[i for i in categories])
plt.figure(figsize=(10, 7))
mx = sn.heatmap(df_cm/np.sum(df_cm), annot=True, fmt='.2%', cmap='Blues')
del df_cm

# Save model along with loss (train, validation), accuracy & class accuracy
checkpoint = {'training_losses': train_losses,
              'valid_losses': valid_losses,
              'accuracies': accuracies,
              'classes_accuracies':classes_accuracies,
              'state_dict': model.state_dict()}

torch.save(checkpoint, path_to_model)