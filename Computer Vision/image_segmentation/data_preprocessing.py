"""Objective - Prepare the image sets for modelling"""
# Import libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib, os, shutil
import random
import requests
import warnings

from zipfile import ZipFile
from PIL import Image

from google_images_download import google_images_download
import reader
warnings.filterwarnings('ignore')

# Data Collection

"""
# Download from Google
def download_google_images(query, ext="jpg", limit=10):
    response = google_images_download.googleimagesdownload()
    # aspect ratio denotes the height width ratio of images to download. ("tall, square, wide, panoramic")
    # specific_site
    # usage_rights: labeled-for-reuse
    arguments = {"keywords": query,     # keywords is the search query
                 "format": ext,         # format: svg, png, jpg
                 "limit": limit,        # limit: is the # of images to download
                 "print_urls": False,   # print_urls : prints image file url
                 "silent_mode": True,
                 "usage_rights": "labeled-for-reuse",
                 "size": "large"}       # size: large, medium, icon

    try:
        path = response.download(arguments)
    except FileNotFoundError:
        print("Exception raised")


# Download for the following categories
search = ["genre painting", "history painting", "literary painting", "mythological painting", "portrait", "religious painting", "self-portrait", "still life", "vanities"]

pathToDataset = pathlib.Path.cwd().joinpath('google_downloads')
for i in search:
    download_google_images(i, ext="jpg", limit=40)

print("Download completed !\n\n")"""


# Direct download - Download images from a list of urls in the txt files
def download_listed_images(filepath):
    # Check 'downloads' folder exists
    download_path = pathlib.Path.cwd().joinpath('downloads')
    if not download_path.exists():
        download_path.mkdir()

    # Check Art Category folder exists
    download_path = download_path.joinpath(filepath[:-4])
    if not download_path.exists():
        download_path.mkdir()

    # grab the list of URLs from the input file, then initialize the total number of images downloaded so far
    urls = open(filepath).read().strip().split("\n")
    url_counter = 0

    # loop the URLs
    for url in urls:

        try:
            # try to download the image
            req = requests.get(url, timeout=60)

            # save the image to disk
            downloaded_image_path = download_path.joinpath("{}.jpg".format(str(url_counter).zfill(8)))
            downloaded_image = open(downloaded_image_path, "wb")
            downloaded_image.write(req.content)
            downloaded_image.close()

            # update the counter
            print("[INFO] downloaded: {}".format(downloaded_image_path))
            url_counter += 1

        # Handle exceptions in download process
        except:
            print(f"[INFO] download error...skipping {downloaded_image_path}")


# Data Preparation
# Utility functions for retrieving files and folders within a directory
def get_files_in_directory(path_to_dir, extension="*.*"):
    return list(pathlib.Path(path_to_dir).glob(extension))


def get_folders_in_directory(path_to_dir, prefix=""):
    path_to_directory = pathlib.Path(path_to_dir)
    folders = [f.name for f in os.scandir(path_to_directory) if f.is_dir()]
    return folders


# Prepare the images
path_to_dataset = pathlib.Path.cwd().parent / 'dataset'
path_to_download = path_to_dataset / 'downloads'

path_to_train = path_to_dataset / 'train'
if not path_to_train.exists():
    path_to_train.mkdir()

path_to_test = path_to_dataset / 'test'
if not path_to_test.exists():
    path_to_test.mkdir()

path_to_valid = path_to_dataset / 'valid'
if not path_to_valid.exits():
    path_to_valid.mkdir()

art_categories = get_folders_in_directory(path_to_download, ".")  # collects list folders
print(f"Total number of art categories : {len(art_categories)} and such categories are : {art_categories}")

# For each category in downloads, split images to test (20%) and valid (20%)
for art in art_categories:

    # set source folder
    source = path_to_download.joinpath(art_categories)

    # set datasets
    images = get_files_in_directory(source, "*.jpg")

    # split 20%
    index = int(len(images) / 5)
    split = np.split(images, [3*index, 4*index, 5*index])

    # target folder
    train_target_path = path_to_train / art

    if not train_target_path.exists():
        train_target_path.mkdir()
    for image in split[0]:
        shutil.move(image, train_target_path / image.name)

    test_target_path = path_to_test / art
    if not test_target_path.exists():
        test_target_path.mkdir()
    for image in split[1]:
        shutil.move(image, test_target_path / image.name)

    valid_target_path = path_to_valid / art
    if not valid_target_path.exists():
        valid_target_path.mkdir()
    for image in split[2]:
        shutil.move(image, valid_target_path / image.name)


data = ['train', 'test', 'valid']

for folder in data:
    for art in art_categories:
        current = path_to_dataset / folder / art
        print(f'Directory {current} has {len(os.listdir(current))} images')


""" Clean corrupted files """


def clean_images(location):
    categories = get_folders_in_directory(location, ".")
    # loop over art categories
    for a in categories:
        # set source
        source_path = path_to_train.joinpath(a)
        # set dataset
        files = get_files_in_directory(source_path, "*.jpg")
        for file in files:
            try:
                img = Image.open(file)
            except IOError:
                print(file)
                os.remove(file)


path_to_train = pathlib.Path.cwd().joinpath('..', 'dataset', 'train')
clean_images(path_to_train)

path_to_test = pathlib.Path.cwd().joinpath('..', 'dataset', 'test')
clean_images(path_to_test)

path_to_valid = pathlib.Path.cwd().joinpath('..', 'dataset', 'valid')
clean_images(path_to_valid)


"""This concludes the data pre-processing for further downstream modeling and testing tasks"""
