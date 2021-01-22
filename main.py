# -*- coding: utf-8 -*-

import video as vid
import train as trn
import torch
from pathlib import Path

'''

This program trains a pre-trained resnet18 model to classify images
of cats, dogs or objects which are neither of them. The program is run from
the main.py file. The two datasets used are specified below, and should be 
extracted into data/. Two further included modules are imported (see below).
After training the model, predictions may be made either on a Youtube video 
from its url or on all images in a specified directory. If the model output 
for a cat (or dog) is high enough, a cat is "detected". In a video, the
detection is a discrete function between 0 to 1 of the elapsed video time, 
which may be plotted.


main.py (this file) contains some calculation hyperparameters and features 
which may be turned on and off, such as saving or loading a trained model from 
disk. This file also contains directories and the youtube url for material
to be evaluated by a trained model (prediction)

Some comments on hyperparameters:
- Dataset size: due to the transferred weights from the pre-trained model, 
relatively small dataset sizes are sufficient. Fractions of 0.1-0.4 have been tested.
- Image resolution: may be changed, possibly with a decrease in batch size
- THRESHOLD: specifies how strong an output a image must yield, during prediction,
in order to classify the image as cat or dog (cat/dog specified by LOOK_FOR)
- DO_FIVECROP: further splits each video frame into five smaller chunks, increasing 
the likelihood of detecting a cat (dataset enhancement, six times as costly).
- Number of epochs: an optimizer scheduler decreases the learning rate by
0.1 every 5th epoch (train.py). Typically, only a few epochs a needed for decent
predictive power
- ngpu: specifies if one gpu (1) or the cpu (0) should be used

train.py contains the NN parts, basically a residual convolutional neural network
which are loaded from a resnet18 model pretrained on the imagenet 1000 dataset.
Further hyperparameters are defined here, including learning rate, momentum etc.
Also, modifications to the model structure may be made (e.g. dropout).

video.py contains a pytorch dataset class implementation that loads an mp4 video file
into a pytorch dataset. This dataset is called upon by main.py. 
video.py also contains parts that download the youtube mp4, decodes and splits it into 
frames of a given time resolution (defined in main.py). Videos may be cropped after a 
certain time. 
Furthermore, the file contains a VideoScanner class which contains auxiliary functions 
to perform a detection scan on a video from a trained model


10000 images, Dogs & Cats dataset
https://www.kaggle.com/chetankv/dogs-cats-images

ca. 38000 misc images, ImageNet 1000 dataset
Class indices 151-268 (dogs) and 281-285 (cats) removed
https://www.kaggle.com/ifigotin/imagenetmini-1000

Extract datasets into data/

'''


# Program control
save = False
load = False
train_model = True

show_video_frames = False   # plot each scanned video frame (for performance testing)
predict_on_images = False   # make predictions on all images in a directory
plot_prediction = True      # plot the detection of the target vs the seconds

# HYPERPARAMETERS
SEED = 999
IMAGE_SIZE = 224
# Training
BATCH_SIZE = 16
NUM_EPOCHS = 5
DS_FRACTION_CAT_DOG = 0.2   # fraction of cat_dog dataset? (1.0 -> full dataset)
DS_FRACTION_NEITHER = 0.2   # fraction of neither dataset?
# Video analysis
SCAN_RES = 0.1              # timestep between video frames
VIDEO_CROP = None           # video cropping after x seconds
THRESHOLD = 0.50            # sensitivity parameter towards detection
LOOK_FOR = 0                # 0: cats, 1: dogs
DO_FIVECROP = True          # also crop video frames into five smaller pieces?
                            # -> increases probability of detection
# Processing unit
ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) 
                      else 'cpu')

# DIRECTORIES/FILES
# Dataset image directories
data_dir_cat_dog = Path('data/cats_and_dogs')
data_dir_neither = Path('data/imagenet-mini')

# This youtube video is downloaded and scanned
yt_video_url='https://www.youtube.com/watch?v=olRi5-AOn1A'

# If predict_on_images: All images in this directory will be scanned
image_predictions_dir = Path('predictions/img/' )

# If save or load: Path to save or load model
savepath = Path('saved_model')



if __name__=="__main__":
    
    # Setup (and train?) model
    model = trn.setupModel(SEED, 
                           batch_size=BATCH_SIZE, 
                           num_epochs=NUM_EPOCHS, 
                           data_dir1=data_dir_cat_dog, 
                           data_dir2=data_dir_neither, 
                           image_size=IMAGE_SIZE, 
                           dataset_fraction_cats_dogs=DS_FRACTION_CAT_DOG, 
                           dataset_fraction_neither=DS_FRACTION_NEITHER,
                           save=save,
                           load=load,
                           savepath=savepath,
                           train_model=train_model,
                           )
    
    
    
    # # Optionally: Dataset from video file
    # prediction_videofile = Path('predictions/videos/some_video_file.mp4')
    # video_dataset = vid.VideoDataset(prediction_videofile, 
    #                                   IMAGE_SIZE, 
    #                                   SCAN_RES, 
    #                                   video_crop=VIDEO_CROP,
    #                                   do_fivecrop=DO_FIVECROP)
    
    # Dataset from youtube url
    video_dataset, video_path = vid.datasetFromYoutubeUrl(yt_video_url, 
                                      IMAGE_SIZE, 
                                      SCAN_RES, 
                                      video_crop=VIDEO_CROP,
                                      do_fivecrop=DO_FIVECROP)
    
    video_dataloader = torch.utils.data.DataLoader(video_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
    
    
    
    # PREDICTION OPTIONS
    
    if predict_on_images:
        # Visualize results on images in image prediction folder
        trn.imagePredictions(model=model, 
                             image_size=IMAGE_SIZE, 
                              pred_dir=image_predictions_dir, 
                              pred_threshold=THRESHOLD,
                              pred_label_considered=LOOK_FOR,
                              batch_size=BATCH_SIZE)
    
    
    if show_video_frames:
        # Visualize the frames of video
        trn.visualizeModel(model, 
                           video_dataloader, 
                            pred_threshold=THRESHOLD, 
                            pred_label_considered=LOOK_FOR)
    
    
    
    if plot_prediction:
        # Plot target detection as a function of time
        # Use VideoScanner class for relevant functions
        video_scanner = vid.VideoScanner(video_dataloader)
        preds_out, time_seconds_out = video_scanner.scanVideo(model, device)
        video_scanner.plotTargetDetection(trn, 
                                          preds_out, 
                                          time_seconds_out, 
                                          THRESHOLD, 
                                          LOOK_FOR)
    
    
    
    
