# -*- coding: utf-8 -*-

import video as vid
import train as trn
import sys
import time
import torch
import matplotlib.pyplot as plt
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


    return (preds_out, timestamps_out)

# program control
save = False
load = True
train = False

## hyperparameters
SEED = 999
IMAGE_SIZE = 448
# training
BATCH_SIZE = 8
NUM_EPOCHS = 5
DS_FRACTION_CAT_DOG = 0.1
DS_FRACTION_NEITHER = 0.1
# video analysis
SCAN_RES = 1
VIDEO_CROP = 300
THRESHOLD = 0.5

# directories/files
data_dir_cat_dog = 'data/cats_and_dogs'
data_dir_neither = 'data/imagenet-mini'
videofile='predictions/videos/dido_emilia.mp4'
savepath = 'saved_model_448px'


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
                              train=train,
                              )
    

    video_dataset = vid.VideoDataset(videofile, IMAGE_SIZE, SCAN_RES, video_crop=VIDEO_CROP)
    video_dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=False)
    
    # Visualize the frames of video
    trn.visualizeModel(model, video_dataloader, pred_threshold=THRESHOLD)
    
    # # Visualize results on images in image prediction folder
    # trn.imagePredictions(model=model, image_size=224, 
    #                      pred_dir='predictions/img', pred_threshold=0.25)
    
    
    # Plot cat detection as a function of time
    preds_out, timestamps_out = scanVideo(video_dataloader, model)
    
    _, pred_cat = trn.getPredictedLabel(preds_out, threshold=THRESHOLD, label_considered=0)
    pred_catplot = []
    for i in pred_cat:
        if i == 0:
            pred_catplot.append(1) 
        else:
            pred_catplot.append(0) 
    plt.plot(np.array(timestamps_out), pred_catplot, label='cat')
    plt.xticks(np.arange(0, video_dataset.scan_length+1, 10))
    plt.tick_params(axis ='x', labelsize = 8, rotation = 90 )
    plt.show()
    # plt.ion()
    
    # # # Show video frames
    # # for img, title in video_dataset:
    # #     trn.imshow(img, title)
        
    # cat_preds = [i[0] for i in preds_out]
    # dog_preds = [i[1] for i in preds_out]
    # neither_preds = [i[2] for i in preds_out]
    
    # plt.plot(timestamps_out, cat_preds, label='cat')
    # plt.plot(timestamps_out, dog_preds, label='dog')
    # plt.plot(timestamps_out, neither_preds, label='neither')
    # plt.legend()
    
    # plt.ioff()
    # plt.show()
