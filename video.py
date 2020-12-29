# -*- coding: utf-8 -*-

import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms



    
# Dataset class consistent with torch
class VideoDataset(Dataset):

    def __init__(self, video_filepath, image_size, scan_res, video_crop=None):
        self.video_filepath = video_filepath
        self.image_size = image_size
        self.scan_res = scan_res
        self.video_crop = video_crop
        
        
        self.video_transforms = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(self.image_size),
             transforms.CenterCrop(self.image_size),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
              ])

        cap = cv2.VideoCapture(self.video_filepath)
        
        # Get full screen (fs) frames
        self.frames_fs, self.timestamps_fs = self._splitVideo(cap, self.scan_res, self.video_crop)
    
    def __len__(self):
        return len(self.frames_fs)
        
    def __getitem__(self, idx):

        frame = self.video_transforms(self.frames_fs[idx])
        timestamp = self.timestamps_fs[idx]
        sample = (frame, timestamp)
        
        return sample
    
    # Return list of image frames separated by the given resolution
    # Format: numpy image
    def _splitVideo(self, videocapture, scan_res, max_t=None):
        fr = videocapture.get(cv2.CAP_PROP_FPS)
        
        # Read first frame
        ret, frame = videocapture.read()
        assert ret == True, "Empty video provided"
        
        # frames contains image pixels and timestamps the time at which they occur in the video
        frames, timestamps = [], []
        t = 0.
        frames.append(self._cv2frameToNumpyImage(frame))
        timestamps.append(t)

        # Start adding frames every "scan_res-th" second
        while ret:
            t += 1./fr
            ret, frame = videocapture.read()
    
            # Break loop after certain time
            if max_t is not None:
                if t > float(max_t):
                    break
            # Check if next frame should be appended depending on the remainder operator of this and the previous frame
            if t % scan_res < (t - 1./fr) % scan_res:
                frames.append(self._cv2frameToNumpyImage(frame))
                timestamps.append(t)
        
        self.scan_length = t        
        
        return (frames, timestamps)
    
    # Convert cv2 video output to numpy Image, changing from BGR to RGB format
    def _cv2frameToNumpyImage(self, frame):
        frame[:,:,[0,2]] = frame[:,:,[2,0]]
        return frame

        
        
if __name__ == "__main__":

    videofile="predictions/videos/cat_video.mp4"
    scan_res = 10. # frames with this distance (in seconds) will be considered
    
    dataset = VideoDataset(videofile, 128, scan_res, video_crop=50)
    


