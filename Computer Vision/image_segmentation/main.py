import data_preprocessing
import transfer_learning
import image_segmentation

def main():
    # Prefer downloading from google images directly by uncommenting the 'download_google_images()' method
    # Below modules constitute the whole project pipeline

    data_preprocessing          # Download data, clean & store in respective directories
    transfer_learning           # Augment data, normalize, import resnet50, fine-tune, train for a few epochs, save model checkpoint
    image_segmentation		# Trains DeepLabv3 model, segments & maps images, fine-tune model, validate & save final output

if __name__ == '__main__':
    main()
