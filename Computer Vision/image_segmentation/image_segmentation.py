""" APPLICATION OF SEMANTIC SEGMENTATION OF THE ART PAINTINGS DATASET
OBJECTIVES:
* Applying Semantic Segmentation on images from Art Paintings dataset using pre-trained DeepLabV3 model to detect various object boundaries.
* Train pre-trained DeepLabv3 model using transfer learning on given art images dataset.
Using only 'genre painting' here and it applies to other art categories as well.

Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). Image segmentation is typically used to locate objects and
boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain
characteristics. The first section of this notebook will use pre-trained segmentation models. Globally, these models tend to perform well but the segmentation can be improved by using additional art images.
The second section will show how these pre-trained models can be refined.

"""

# Import necessary packages
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import pathlib
import sys, time
import warnings

# PyTorch modules/packages
import torch
import torch.nn

from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# %matplotlib inline (If using Jupyter environment)
warnings.filterwarnings('ignore')

"""Setting up training mode based on CUDA """

device = 'cpu'      # sets the default value
train_on_gpu = torch.cuda.is_available()        # returns True if CUDA enabled GPU is available

if train_on_gpu:
    print('CUDA is available! Training on GPU...\n')
    print(torch.cuda.get_device_properties(0))
    device = 'cuda'

# Utility functions for files and folders
# Retrieves the list of files with a directory
def getFilesInDirectory(pathToDir, extension = "*.*"):
    if not isinstance(pathToDir, pathlib.PurePath):
        pathToDir = pathlib.Path(pathToDir)

    return list(pathToDir.glob(extension))

# Retrieves the list of folders with a directory
def get_folders_in_directory(path_to_dir, prefix =""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)

    return sorted([fld for fld in path_to_dir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])

# Retrieves the list of folders with a directory
def get_folder_names_in_directory(path_to_dir, prefix =""):
    if not isinstance(path_to_dir, pathlib.PurePath):
        path_to_dir = pathlib.Path(path_to_dir)

    return sorted([fld.name for fld in path_to_dir.iterdir() if fld.is_dir() and not fld.name.lower().startswith(prefix)])

# Calculates the Intersection Over Union (IOU)
def iou(pred, target, n_classes = 3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)

# Data Preparation - define directories and location of image sets archives

# sets the root folder for image sets
pathToDataset = pathlib.Path.cwd().joinpath('..','dataset')
pathToTest = pathToDataset.joinpath('test')

# Check 'segmented' folder exists
pathToSegmented = pathToDataset.joinpath('segmented')
if not pathToSegmented.exists():
    pathToSegmented.mkdir()

pathToSegmentedImg = pathToSegmented.joinpath('genre', 'train')
if not pathToSegmentedImg.exists():
    pathToSegmentedImg.mkdir()

pathToSegmentedOutput1 = pathToSegmented.joinpath('outputs_s1')
if not pathToSegmentedOutput1.exists():
    pathToSegmentedOutput1.mkdir()

"""
Semantic Segmentation:

`DeepLabV3 with Resnet-101 backbone` model will be used to deal with image semantic segmentation. This pre-trained model have been trained on a subset of COCO train2017, on the 20 categories that are 
present in the Pascal VOC dataset. The pre-trained model was trained on `21` classes and thus our output will have `21` channels ! We need to convert this 21 channeled outputs into a 
`2D` image (or a `1` channeled image), where each pixel of that image corresponds to a class.

Segmentation Mapping:

At some point, we will need to convert to a segmentation map an image where each pixel corresponds to a class label. Each class label will be converted into a `RGB` color.
The purpose is to visualise easier the segmentation. Let's define a function (called `decode_segmap`) that would accept an `2D` image and the colors for each categories handled by the model.

The function should now create an `RGB` image from the `2D` image passed. To do so, the function creates empty `2D` matrices for all 3 channels. So, `r`, `g`, and `b` are arrays which will form the
`RGB` channels for the final image. And each are of shape `[H x W]` (which is same as the shape of `image` passed in). The function will then loop over each class color we stored in `label_colors`,
get the indexes in the image where that particular class label is present and  for each channel, it puts its corresponding color to those pixels where that class label is present. Finally the function
stacks the 3 separate channels to form a `RGB` image.
"""


def decode_segmap(image, label_colors):
    len_categories = len(label_colors)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, len_categories):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

"""Image Pre-processing

The images should be preprocessed to facilitate the segmentation process. Let's define a function that will transform the images according to the model input expectations. 

The function will first `label_colors` stores the colors for each of the classes, according to the index
    The first class which is `background` is stored at the `0th` position (index)
    The second class which is `aeroplane` is stored at the `1st` position (index)

Open the file and convert it to RGB  

The function should unsqueeze the image so that it becomes `[1 x C x H x W]` from `[C x H x W]`  

The `2D` image, (of shape `[H x W]`) will have each pixel corresponding to a class label. So, each `(x, y)` will correspond to a number between `0 - 20` representing a class (`[1 x 21 x H x W]`). The function should take a max index for each pixel position.

**Note:** The pre-trained model that will be used has been trained on `21 categories` (20 categories + background (black)).
The classes that the pre-trained model outputs are the following, in order:  
['__background__' , 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

The function will require a transformation which:  
* resize the image to `(256 x 256)`
* center and crop it to `(224 x 224)`
* convert it to a Tensor (all the values in the image becomes between `[0, 1]` from `[0, 255]`)
* normalize it with the Imagenet specific values `mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]`
"""
# Define the data-augmentation transforms including normalisations
segment_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

label_colors = np.array([
    (0, 0, 0),  # 0=background
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),  # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),  # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),  # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)  # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
])


def segment(seg_model, fileName, outputPath, showImage=True):
    local_img = Image.open(fileName).convert('RGB')
    n_img = np.asarray(local_img)

    if len(n_img.shape) != 3:
        print('ERROR! ', fileName, ' is grayscale and not in RGB format. Cannot implement segmentation. Ignoring it!')
    else:
        print('Implementing segmentation on ', fileName)

        inp = segment_transforms(local_img).unsqueeze(0).to(device)  # inp= Size([1, 3, 224, 224])
        #        trans = transforms.ToPILImage(mode='RGB')
        out = seg_model.to(device)(inp)['out']  # out= Size([1, 21, 224, 224])
        image = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        rgb = decode_segmap(image, label_colors)

        if (showImage == True):
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.imshow(local_img)
            plt.subplot(122)
            plt.imshow(rgb)
            plt.axis('off');
            plt.show()

"""Model-based Semantic Segmentation
1. Load in a pre-trained model & display its structure 
2. Process each file for each art category under `Test` folder, 
"""
# Set the model
deep_lab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Print out the model
# print( deep_lab )

# Collect the images
files = getFilesInDirectory(pathToSegmentedImg, '*.jpg')
for file in files:
    # Apply semantic segmentation
    segment(deep_lab, file, pathToSegmentedOutput1, showImage=False)


""" Transfer learning based on pre-trained model"""

import glob, os
from torch.utils.data.dataset import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, img_path, mask_path, mode):
        super(SegmentationDataset, self).__init__()

        # Collect files
        self.img_files = glob.glob(os.path.join(img_path, '*.jpg'))
        self.msk_files = glob.glob(os.path.join(mask_path, '*.png'))

        # Data augmentation and normalization for training
        # Just normalization for validation (='V')
        if "V" == mode:
            self.transforms = transforms.Compose([
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        msk_path = self.msk_files[index]

        image = Image.open(img_path)
        mask = Image.open(msk_path)

        # Convert to arrays
        image_np = np.asarray(image)
        mask_np = np.asarray(mask)

        # Convert to tuple (256, 256, 4)
        new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)

        # Convert to ndarray (256, 256, 4)
        combined_np = np.zeros(new_shape, image_np.dtype)

        # Concatenate image and mask so transformation is applied on both
        combined_np[:, :, 0:3] = image_np
        combined_np[:, :, 3] = mask_np

        # Convert to PIL
        combined = Image.fromarray(combined_np)

        # Apply transformation and get a Tensor [4, 224, 224]
        combined = self.transforms(combined)

        # Extract image Tensor ([3, 224, 224]) and mask Tensor ([1, 224, 224])
        image = combined[0:3, :, :]
        mask = combined[3, :, :].unsqueeze(0)

        # Normalize back from [0, 1] to [0, 255]
        mask = mask * 255

        #  Convert to int64 and remove second dimension - Tensor ([224, 224])
        mask = mask.long().squeeze()

        return image, mask


    # Path definition - define path for ground truth masks.
    # Masks folder
    pathToSegmentedMask = pathToSegmented.joinpath('masks', 'train')
    if not pathToSegmentedMask.exists():
        pathToSegmentedMask.mkdir()

"""Transformation & Data loaders
Some data loader's parameters will be set: the size of the batch to 4 and the number of workers to 0.

We will first instantiate the `SegmentationDataset` class with the appropriate parameters and, then, create a dataloader using the dataset instance.

**Note**: Don't forget to set to `True` the **drop_last** parameter of the Dataloader constructor. """
batch_size = 4
num_workers = 0

# 'T'rain & 'V'alidation transformations & data loaders
tv_datasets = {x: SegmentationDataset(pathToSegmentedImg, pathToSegmentedMask, x) for x in ['T', 'V']}
tv_dataloaders = {x: DataLoader(tv_datasets[x], batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers) for x in ['T', 'V']}

""" Modelling
Let's set the pretrained model based on which the transfer learning will be performed. Then we will set the different parameters of the features.
The model will then be customized to fit specific requirements related to Semantic Segmentation.This is very similar to what we have seen in the previous milestones.
"""
# Set the number of output channels (= number of classes)
num_classes = 7

# Model definition
seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

# The auxiliary classifier is removed, and the pretrained weights are frozen
seg_model.aux_classifier = None
for param in seg_model.parameters():
    param.requires_grad = False

# The pretrained classifier is replaced by a new one with a custom number of classes.
# Since it comes after the freeze, its weights won't be frozen. They are the ones that we will fine-tune.
seg_model.classifier = DeepLabHead(2048, num_classes)

# Model serialisation
model_filename = 'custom_segmented.pt'
pathToModel = pathlib.Path.cwd().joinpath('..', 'models', model_filename)
print('File name for saved model: ', pathToModel)

# Loss function
criterion = torch.nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss(), well suited for multiclass classification problems

# Optimizer definition
optimizer = torch.optim.SGD(seg_model.parameters(), lr=0.001, momentum=0.9)

# Use cpu/gpu based on availability
seg_model.to(device)

# Setup variables using during Training & Validation
n_epochs = 50
counter = 0

# Initialise epoch performances
epoch_loss = 0.0
epoch_acc, epoch_bestacc = 0.0, 0.0

# Initialise training performances
train_iou_means, train_losses = [], []
train_loss, train_iou_mean = 0.0, 0.0

# Initialise validation performances
val_acc_history = []

""" Training and Validation """
a = time.time()  # Start-time for training

for epoch in range(1, n_epochs + 1):

    # Start-time for epoch
    c = time.time()

    for phase in ['T', 'V']:
        if phase == 'T':
            # Set to training mode
            seg_model.train()
        else:
            # Set to evaluation mode
            seg_model.eval()

        train_loss = 0.0
        train_iou_means = []

        # Iterate over data - Getting one batch of training images and their corresponding true labels
        for inputs, masks in tv_dataloaders[phase]:

            images = inputs.to(device)  # Array of tensors - size: [3, 224, 224]
            masks = masks.to(device)  # Array of tensors - size: [224, 224]

            # zero the parameter gradients
            optimizer.zero_grad()

            # Turning on calculation of gradients (not required for validation)  {saves time}
            with torch.set_grad_enabled(phase == 'T'):

                outputs = seg_model(images)['out']  # returns an Array of tensors - size: [3, 224, 224]

                # Calculate the train loss
                train_criterion = criterion(outputs, masks)  # (prediction, true-label)
                train_loss += train_criterion.item() * inputs.size(0)

                # Returns the maximum values (tuple: values, indices) of each row of the `outputs` in the dimension `1`
                _, preds = torch.max(outputs, 1)

            if (phase == 'T'):
                # backward pass: compute gradient of the loss based on model parameters
                train_criterion.backward()

                # perform a single optimization step
                optimizer.step()

            # Collect the Intersection Over Union (IOU)
            train_iou_mean = iou(preds, masks, num_classes).mean()
            train_iou_means.append(train_iou_mean)
            train_losses.append(train_loss)

            # Increment counter
            counter = counter + 1

        # Displays statistics
        epoch_loss = train_loss / len(tv_dataloaders[phase].dataset)
        if (train_iou_means is not None):
            epoch_acc = np.array(train_iou_means).mean()
        else:
            epoch_acc = 0.

        d = time.time()  # end-time for epoch

        print(f"Epoch: {epoch} | "
              f"Time: {int((d - c) / 60)} min {int(d - c) % 60} sec | "
              f"Phase: {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
              )

        # save model if validation loss has decreased
        if ((phase == 'V') and (epoch_acc > epoch_bestacc)):
            print('Epoch accuracy increased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_bestacc, epoch_acc))

            # saving model
            torch.save(seg_model.state_dict(), pathToModel)

            # update minimum validation loss
            epoch_bestacc = epoch_acc

        if (phase == 'V'):
            val_acc_history.append(epoch_acc)

################
# END OF EPOCH #
################
b = time.time()
print('\n\n\tTotal training time: ', int((b - a) / (60 * 60)), "hour(s) ", int(((b - a) % (60 * 60)) / 60), "minute(s) ", int(((b - a) % (60 * 60)) % 60), "second(s)")


# Load the model with lowest validation loss
seg_model.load_state_dict(torch.load(pathToModel))

"""Testing

Preparation:
    * The following variables will be defined:
    * The path where the test images are located
    * The transformation that will be applied on the test image
"""

# Test & Valid image folder
pathToSegmentedTest = pathToSegmented.joinpath('genre', 'test')
if not pathToSegmentedTest.exists():
    pathToSegmentedTest.mkdir()

# Test transform & data loader
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of labels to be identified
label_colors = np.array([
            (0, 0, 0),  # 0= background
            (128, 0, 0), # 1= chair
            (0, 128, 0), # 2= door_window
            (128, 128, 0), # 3= person
            (0, 0, 128),  # 4= table
            (128, 0, 128), # 5= animal
            (0, 128, 128) # 6= bird
])

# Set the model to evaluate mode
seg_model.eval()

images = getFilesInDirectory(pathToSegmentedTest, '*.jpg')

for idx in range(0, len(images) - 1):

    image_orig = Image.open(images[idx]).convert("RGB")
    n_img = np.asarray(image_orig)

    if len(n_img.shape) != 3:
        print('ERROR! ', file, ' is grayscale and not in RGB format. Cannot implement segmentation. Ignoring it!')
    else:
        print('Implementing segmentation on ', images[idx])

        # Apply the transformation to the image and add a dimension to fit into the model input
        inp = test_transforms(image_orig).unsqueeze(0).to(device)  # inp= Size([1, 3, 256, 256])

        # Execute the model and return the output (`out`)
        out = seg_model.to(device)(inp)['out']  # out= Size([1, 7, 256, 256])

        # Returns the indices of the maximum values of a tensor across a dimension.
        image = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

        # Colorise the image with the predefined labels using the `decode_segmap`
        rgb = decode_segmap(image, label_colors)

        # Plotting
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(image_orig)
        plt.subplot(122)
        plt.imshow(rgb)
        plt.axis('off');
        plt.show()

""" This marks the end of semantic segmentation project. Plotted results are visualized and can be seen how semantic segmentation divides various objects in an image into different colours
(segmentation)"""