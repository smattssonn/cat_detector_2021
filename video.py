# -*- coding: utf-8 -*-

import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import pytube
from urllib.parse import urlparse
import numpy as np
from pathlib import Path
import time
import sys


''' 
Dataset class consistent with pytorch
- The length of the video may be cropped
- The video may be cropped into smaller pieces using torchvision's FiveCrop
method, meaning the change of detecting e.g. a cat increases
'''   
class VideoDataset(Dataset):

    def __init__(self, 
                 video_filepath, 
                 image_size, 
                 scan_res, 
                 video_crop=None, 
                 do_fivecrop=False):
        
        self.video_filepath = video_filepath
        self.image_size = image_size
        self.scan_res = scan_res
        self.video_crop = video_crop
        self.do_fivecrop = do_fivecrop
        
        self.video_transforms = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(self.image_size),
             transforms.CenterCrop(self.image_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
              ])

        cap = cv2.VideoCapture(str(self.video_filepath))
        
        # Get full screen (fs) frames
        self.frames, self.time_seconds = self._splitVideo(cap, self.scan_res, self.video_crop)
    
    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        frame = self.video_transforms(self.frames[idx])
        time_second = self.time_seconds[idx]
        sample = (frame, time_second)
        return sample
    
    # Return list of image frames separated by the given resolution
    # Format: numpy image
    def _splitVideo(self, videocapture, scan_res, video_crop=None):
        fr = videocapture.get(cv2.CAP_PROP_FPS)
        
        # Read first frame
        ret, frame = videocapture.read()
        assert ret == True, 'Empty video provided'
        
        # frames contains image pixels and time_seconds the time at 
        # which they occur in the video
        frames, time_seconds = [], []
        t = 0.

        # Start adding frames every "scan_res-th" second
        while ret:
            t += 1./fr
            ret, frame = videocapture.read()
            
            # Makes certain no non-returned frames make it into the output list
            if ret == False:
                break
            
            # Break loop after certain time?
            if video_crop is not None:
                if t > float(video_crop):
                    break
                
            # Check if next frame (or first frame) should be appended
            if t % scan_res < (t - 1./fr) % scan_res or t == 1./fr:
                if not self.do_fivecrop:
                    # Don't fivecrop frame
                    print(frame)
                    frames.append(self._cv2frameToNumpyImage(frame))
                    time_seconds.append(t)
                else:
                    # Fivecrop frame
                    crop_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.image_size*2),
                        transforms.FiveCrop(self.image_size),
                        ])
                    fivecropped_frames = crop_transforms(frame)
                    # Append whole and five-cropped frames, in total six frames
                    # with the same t
                    frames.append(self._cv2frameToNumpyImage(frame))
                    time_seconds.append(t)
                    for f in fivecropped_frames:
                        frames.append(self._cv2frameToNumpyImage(np.asarray(f)))
                        time_seconds.append(t)
        
        self.scan_length = t        
        
        return frames, time_seconds
    
    # Convert cv2 video output to numpy Image, changing from BGR to RGB format
    def _cv2frameToNumpyImage(self, frame_bgr):
        frame = frame_bgr[...,::-1]
        return frame

     # Convert cv2 video output to numpy Image, changing from BGR to RGB format
    def saveFrames(self, outdir):
        for frame, t in zip(self.frames, self.time_seconds):
            im = Image.fromarray(frame)
            path = Path('{}/{}.jpg'.format(outdir, round(t, 2)))
            im.save(path)

    # Plot frames of the video (mainly debugging or performance testing)
    def showFrames(self):
        for frame, t in zip(self.frames, self.time_seconds):
            im = Image.fromarray(frame)
            plt.imshow(im)
            plt.show()
    

# Class with functions to scan an input video
class VideoScanner():
    
    # Initialize with data from a VideoDataset instance
    def __init__(self, dataloader):
        self.dataloader = dataloader


    # Run video dataset through a trained model
    def scanVideo(self, model, device):
        since = time.time()
        
        model.eval()   # Set model to evaluate mode
        
        preds_out = []
        time_seconds_out = []
        
        print('Scanning video...')
    
        for i, data in enumerate(self.dataloader):
            frames, time_seconds = data
            frames = frames.to(device)
            
            preds = model(frames).tolist()
    
            # Add each frame output (f) to the list of results (preds_out)
            for f in preds:
                preds_out.append(f)
            
            time_seconds_out += time_seconds.tolist()
            
            # Phase status bar
            sys.stdout.write('\r')
            j = (i + 1) / len(self.dataloader)
            sys.stdout.write('[%-20s] %d%% ' % ('='*int(20*j), 100*j))
            sys.stdout.flush()
                    
    
        time_elapsed = time.time() - since
        print('Video scan completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
        # Account for if a frame has several cropped portions (FiveCrop),
        preds_out, time_seconds_out = self._getFrameMaximum(preds_out, time_seconds_out)
    
        return preds_out, time_seconds_out


    '''
    Takes the best score from all predictions at a certain time_second
    This is necessary to consider five-cropped images which 
    run several times through the model
    '''
    def _getFrameMaximum(self, preds_out, time_seconds_out):
    
        outzip = list(zip(time_seconds_out, preds_out))
        
        # Make list of only unique time_seconds
        unique_ts = []
        for t in time_seconds_out:
            if t not in unique_ts:
                unique_ts.append(t)
        
        # Get the best scores for each class
        best_score = []
        for t in unique_ts:
            tmplist = [ps for ts, ps in outzip if ts == t]
            best_score.append(max(tmplist))
        
        return best_score, unique_ts


    # Plot target detection as a function of time from model predictions
    # Outputs y = 1 for detection and y = 0 for no detection
    def plotTargetDetection(self,
                            trn, 
                            preds_out, 
                            time_seconds_out,
                            threshold,
                            label_considered=0):
        
        # Get predicted target (highest model output)
        # Borrow function from train module
        _, pred_target = trn.getPredictedLabel(preds_out, 
                                               threshold=threshold, 
                                               label_considered=label_considered)
        
        # Plot
        pred_target_plot = []
        for i in pred_target:
            if i == label_considered:
                pred_target_plot.append(1) 
            else:
                pred_target_plot.append(0)
        plt.plot(np.array(time_seconds_out), pred_target_plot, label=trn.class_names[label_considered])
        plt.xticks(np.arange(0, max(time_seconds_out)+1, 10))
        plt.tick_params(axis ='x', labelsize = 8, rotation = 90)
        plt.show()
    

# Download youtube video and create VideoDataset class from the downloaded file
def datasetFromYoutubeUrl(url, 
                          image_size=128, 
                          scan_res=1., 
                          video_crop=None,
                          do_fivecrop=False):

    dest = urlparse(url)
    # Video id is given in the url query after "v=" and before any "&"
    vid_id = dest.query.split('v=')[1].split('&')[0]
    # Directory to save video
    vid_dir = Path('predictions/videos/youtube')

    yt = pytube.YouTube(dest.geturl())
    yt.check_availability()
    
    # Path to mp4 file
    video_filepath = vid_dir / (vid_id + '.mp4')

    if not os.path.isfile(video_filepath):
        print('Downloading video...')
        yt.streams.first().download(vid_dir, vid_id)
    
    
    dataset = VideoDataset(video_filepath, image_size, scan_res, video_crop, do_fivecrop)
    
    return dataset, video_filepath

        
