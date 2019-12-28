# Image Captioning with Deep Learning

## Problem Statement
The aim of this project is to predict a caption for an image. The problem can be broken down into two parts -
1. Image classification using CNN
2. Caption Generation using RNN Bimodal Architecture

## Dataset and Initial Setup
You can using a Image Captioning and Semantic Segmentation dataset like MSCOCO. 
1. Run `$ pip install -r requirements.txt` for installing the respective packages. 
2. Add all the images into the directory path `./data/images`. We are performing a 60-20-20 split of the dataset for training, validation and testing respectively. 
3. Add the instances and captions file as `data/image_data/instances.json` and `data/image_data/captions.json` respectively. 
4. Run `$ python split_data.py` for splitting the images of `data/images` into train, validation and testing directories.

## Model Architecture
You can find the CNN and RNN models at `cnn_model.py` and `rnn_bimodel.py` respectively.

## Training
To train the models run the files in the following manner - 
1. Run `$ python construct_image_categories.py <directory_name>` for each image directory (train/test/validation). This constructs the necessary metadata required for training the CNN model on the images. 
2. Run `$ python extract_labels.py` for saving the image arrays and their labels as dask arrays for the final preprocessing before CNN training.
3. Run `$ python image_captioning_cnn.py` for training the images using the training and validation datasets.
4. Run `$ python extract_captions.py` for saving the captions of the images for each of the directories.
5. Run `$ python text_preprocessing.py` for saving image embeddings and the captions into dask arrays for final preprocessing before the RNN Bimodal training. 
6. Run `$ python image_captioning_rnn.py` for training the RNN Bimodal model.

## Testing
Open `test_cnn_model.ipynb` for testing the CNN and `test_rnn_model.ipynb` for testing the RNN.