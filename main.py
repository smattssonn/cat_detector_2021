# -*- coding: utf-8 -*-

import video as vid
import train as trn
import sys
import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np




# Run video dataset through a trained model
def scanVideo(dataloader, model):
    since = time.time()
    
    model.eval()   # Set model to evaluate mode
    
    preds_out = []
    timestamps_out = []

    for i, data in enumerate(dataloader):
            
            
        frames, timestamps = data
        frames = frames.to(device)
        
        preds = model(frames).tolist()

        # Add each frame output (f) to the list of results (preds_out)
        for f in preds:
            preds_out.append(f)
        
        timestamps_out += timestamps.tolist()
        
        # Phase status bar
        sys.stdout.write('\r')
        j = (i + 1) / len(dataloader)
        sys.stdout.write('[%-20s] %d%% ' % ('='*int(20*j), 100*j))
        sys.stdout.flush()
                

    time_elapsed = time.time() - since
    print('Video scan completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # If a frame has several cropped portions (e.g. FiveCrop)
    # get maximum predicted output for each class out of the cropped
    # frames
    preds_out, timestamps_out = getFrameMaximum(preds_out, timestamps_out)

    return preds_out, timestamps_out

""" 
Takes the best score from all predictions at a certain timestamp
This is necessary to consider five-cropped images which 
run several times through the model
"""
def getFrameMaximum(preds_out, timestamps_out):

    outzip = list(zip(timestamps_out, preds_out))
    
    # Make list of only unique timestamps
    unique_ts = []
    for t in timestamps_out:
        if t not in unique_ts:
            unique_ts.append(t)
    
    # Get the best scores for each class
    best_score = []
    for t in unique_ts:
        tmplist = [ps for ts, ps in outzip if ts == t]
        best_score.append(max(tmplist))
    
    return best_score, unique_ts


# program control
save = False
load = True
train_model = False

## hyperparameters
SEED = 999
IMAGE_SIZE = 224
# training
BATCH_SIZE = 16
NUM_EPOCHS = 5
DS_FRACTION_CAT_DOG = 0.8
DS_FRACTION_NEITHER = 0.4
# video analysis
SCAN_RES = 0.1
VIDEO_CROP = 120
THRESHOLD = 0.25
LOOK_FOR = 0    # 0: cats, 1: dogs

# directories/files
data_dir_cat_dog = 'data/cats_and_dogs'
data_dir_neither = 'data/imagenet-mini'
videofile='predictions/videos/dido_emilia.mp4'
savepath = 'saved_model_336px'
# yt_video_url='https://www.youtube.com/watch?v=VXT1Nqr_qQs'  persian pokemon
yt_video_url='https://www.youtube.com/watch?v=DdX7IsBlOC4' # funny animals
# yt_video_url='https://www.youtube.com/watch?v=RxozAnVBWio' 

ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')


if __name__=="__main__":
    

    model = trn.setupPipeline(SEED, 
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
    
    
    
    # # Dataset from video file
    # video_dataset = vid.VideoDataset(videofile, 
    #                                   IMAGE_SIZE, 
    #                                   SCAN_RES, 
    #                                   video_crop=VIDEO_CROP,
    #                                   do_fivecrop=True)
    
    # Dataset from youtube url
    video_dataset, video_path = vid.datasetFromYoutubeUrl(yt_video_url, 
                                      IMAGE_SIZE, 
                                      SCAN_RES, 
                                      video_crop=VIDEO_CROP,
                                      do_fivecrop=True)
    # print("Scanning:", video_path)
    
    video_dataloader = torch.utils.data.DataLoader(video_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
    
    
    
    # # Visualize results on images in image prediction folder
    # trn.imagePredictions(model=model, image_size=IMAGE_SIZE, 
    #                       pred_dir='predictions/img/', 
    #                       pred_threshold=THRESHOLD,
    #                       pred_label_considered=LOOK_FOR,
    #                       batch_size=BATCH_SIZE)
    
    
    # # Visualize the frames of video
    # trn.visualizeModel(model, video_dataloader, 
    #                     pred_threshold=THRESHOLD, 
    #                     pred_label_considered=LOOK_FOR)
    
    
    # Plot target detection as a function of time
    preds_out, timestamps_out = scanVideo(video_dataloader, model)
    
    
    _, pred_target = trn.getPredictedLabel(preds_out, 
                                            threshold=THRESHOLD, 
                                            label_considered=LOOK_FOR)
    
    pred_target_plot = []
    for i in pred_target:
        if i == LOOK_FOR:
            pred_target_plot.append(1) 
        else:
            pred_target_plot.append(0)
    
    plt.plot(np.array(timestamps_out), pred_target_plot, label=trn.class_names[LOOK_FOR])
    plt.xticks(np.arange(0, video_dataset.scan_length+1, 10))
    plt.tick_params(axis ='x', labelsize = 8, rotation = 90)


    # preds_out, timestamps_out = getFrameMaximum(preds_out, timestamps_out)
    
    # _, pred_target = trn.getPredictedLabel(preds_out, 
    #                                         threshold=THRESHOLD, 
    #                                         label_considered=LOOK_FOR)
    
    # pred_target_plot = []
    # for i in pred_target:
    #     if i == LOOK_FOR:
    #         pred_target_plot.append(1) 
    #     else:
    #         pred_target_plot.append(0)
    
    # plt.plot(np.array(timestamps_out), pred_target_plot, label=trn.class_names[LOOK_FOR])

    plt.show()
    
    
    
    
