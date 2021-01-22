# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import sys

# parameters
ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
class_names = ['cat', 'dog', 'neither']

"""
Instead of random initializaion, we initialize the network 
with a pretrained network, like the one that is trained on 
imagenet 1000 dataset
"""

def imshow(inp, title=None, label=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    if label is not None:
        plt.text(inp.shape[0]*0.5, 20, label, 
                 horizontalalignment='center', color='r', 
                 fontsize=18)
        
        
    plt.pause(0.01)  # pause a bit so that plots are updated


def train(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()
    
    training_tracker = TrainingTracker()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
                # Phase status bar
                sys.stdout.write('\r')
                j = (i + 1) / len(dataloaders[phase])
                sys.stdout.write('[%-20s] %d%% ' % ('='*int(20*j), 100*j))
                sys.stdout.flush()
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            training_tracker.add(phase, epoch_loss, epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    training_tracker.plot()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model


# Prepare a dataloader for visualization
def imagePredictions(model, image_size, pred_dir, pred_threshold=None, pred_label_considered=0, batch_size=None):
    # transforms are the same as 'val' transforms during train(...)
    data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    pred_dataset = datasets.ImageFolder(pred_dir, data_transforms)
    
    if batch_size is None:
        batch_size = len(pred_dataset)

    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=1)

    visualizeModel(model, pred_dataloader, pred_threshold)


# Run model and make predictions
def visualizeModel(model, dataloader, pred_threshold=None, pred_label_considered=0):
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            vals, preds = getPredictedLabel(outputs, threshold=pred_threshold, label_considered=pred_label_considered)

            for j in range(inputs.size()[0]):
                label = vals[j].item()
                # include the predicted label if it is cat or dog
                if class_names[preds[j]] != 'neither':
                    label = class_names[preds[j]] 
                else:
                    label = None
                    
                imshow(inputs.cpu().data[j], label=label)


""" 
Get the predicted label for a model output
threshold decides which one of two modes to be used:
1) If threshold is not set, the maximum value of 'cat', 'dog' and 'neither' 
is returned.
2) If threshold is set, label_considered (default 0 ('cat')) is returned if 
the output value is above threshold, otherwise 2 ('neither') is returned
This would act as a "cat detector" and return neither ALSO if dog has the 
highest output value.
"""
def getPredictedLabel(model_out, threshold=None, label_considered=0):
    # mode 1 (only for debugging)
    if threshold is None:
        vals, preds = torch.max(model_out, 1)
    # mode 2
    else:
        vals = torch.zeros(len(model_out))
        preds = torch.zeros(len(model_out)).int()
        for i, v in enumerate(model_out):
            if v[label_considered] > threshold:
                vals[i] = v[label_considered]
                preds[i] = label_considered
            else:
                vals[i] = v[2]
                preds[i] = 2
        
    return vals, preds
        

# Custom class for "neither cat nor dog" target
class ImageFolderCatAndDogLess(datasets.ImageFolder):
    

    # # Remove images with the given class indices (e.g. indices of cats or dogs)
    # def dropIndices(self, idxs_drop: list) -> None:
    #     self.imgs = [[path, class_idx] for path, class_idx in self.imgs if class_idx not in list(idxs_drop)]
    #     self.class_to_idx = {class_str: idx for class_str, idx in self.class_to_idx.items() if idx not in list(idxs_drop)}
    #     self.classes = [class_str for class_str in self.classes if class_str in self.class_to_idx.keys()]
    #     self.targets = [idx for idx in self.targets if idx not in list(idxs_drop)]
    #     self.samples = [(path, idx) for path, idx in self.samples if idx not in list(idxs_drop)]
       
    # Reduce number of targets to a single integer ("neither cat nor dog")
    def unifyTargets(self, out_target: int) -> None:
        self.samples = [(path, out_target) for path, _ in self.samples]
        self.targets = [s[1] for s in self.samples]        
            
# Keep track of performance during training and plot it afterwards
class TrainingTracker:
    
    def __init__(self):
        self.losses = {x: [] for x in ['train', 'val']}
        self.accuracies  = {x: [] for x in ['train', 'val']}
    
    def add(self, phase, loss, accuracy):
        self.losses[phase].append(loss)
        self.accuracies[phase].append(accuracy)
    
    def plot(self):
        # Train and test loss
        fig, ax = plt.subplots()
        ax.plot(self.losses['train'], color='b', label='train')
        ax.plot(self.losses['val'], color='r', label='val')
        ax.legend()
        ax.set_title('Loss')
        
        # Train and test accuracy
        fig2, ax2 = plt.subplots()
        ax2.plot(self.accuracies['train'], color='b', label='train')
        ax2.plot(self.accuracies['val'], color='r', label='val')
        ax2.legend()
        ax2.set_title('Accuracy')
        
        plt.show()


def setupPipeline(seed, 
                  batch_size, 
                  num_epochs, 
                  data_dir1, 
                  data_dir2, 
                  image_size, 
                  dataset_fraction_cats_dogs, 
                  dataset_fraction_neither,
                  save,
                  load,
                  savepath,
                  train_model
                  ):
    random.seed(seed)
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(int(image_size*1.2)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomApply(p=0.1, transforms=
                [transforms.Pad(int(image_size*0.33), padding_mode='constant')]
                ),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1,
                                    contrast=0.1, 
                                    saturation=0.1, 
                                    hue=0.1),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    
    
    
    # train determines if training or validation datasets are loaded
    # pick False if you are only interested in predictions on video
    if train_model:
        
        # Load cats_and_dogs and imagenet 1000 datasets separately
        image_datasets_cats_dogs = {x: datasets.ImageFolder(os.path.join(data_dir1, x),
                                                  data_transforms[x])
                                    for x in ['train', 'val']}
        
        dataset_sizes_cats_dogs = {x: len(image_datasets_cats_dogs[x]) for x in ['train', 'val']}
        
        image_datasets_neither = {x: ImageFolderCatAndDogLess(os.path.join(data_dir2, x),
                                                  transform=data_transforms[x])
                                  for x in ['train', 'val']}
        
        dataset_sizes_neither = {x: len(image_datasets_neither[x]) for x in ['train', 'val']}
        
    
    
        # Set all imagenet images as same "neither" target
        for x in ['train', 'val']:
            image_datasets_neither[x].unifyTargets(2)
    
    
    
        # If desired, random subsets of either dataset may be used (use 1. for all samples )
        for x in ['train', 'val']:
            # Cats and dogs
            selected_samples = random.sample(range(dataset_sizes_cats_dogs[x]), 
                                              int(dataset_sizes_cats_dogs[x] * dataset_fraction_cats_dogs))
            image_datasets_cats_dogs[x] = torch.utils.data.Subset(image_datasets_cats_dogs[x], selected_samples)
            # Neither
            selected_samples = random.sample(range(dataset_sizes_neither[x]), 
                                              int(dataset_sizes_neither[x] * dataset_fraction_neither))
            image_datasets_neither[x] = torch.utils.data.Subset(image_datasets_neither[x], selected_samples)




        print('Combining %d cat_and_dog and %d neither training images' % (
            len(image_datasets_cats_dogs['train']), len(image_datasets_neither['train'])))
    
    
        # Combine datasets
        image_datasets = {x: torch.utils.data.ConcatDataset([image_datasets_cats_dogs[x], 
                                                image_datasets_neither[x]])
                          for x in ['train', 'val']}
    
        
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=1)
                      for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        
        
        # Plot some training images
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs, nrow=4)
        imshow(out)
        
    
        print('Dataset sizes (train/val): {}/{}'.format(
            dataset_sizes['train'], dataset_sizes['val']))
    
    
    # Define model regardless if train or video predictions
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)
    

    # Load previous model?
    if load:
        model_ft.load_state_dict(torch.load(savepath))     
        print('Model loaded')
    
    if train_model:
        
        criterion = nn.CrossEntropyLoss()
    
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        if not save:
            print('Training WITHOUT saving model')
        
        
        # Train model
        oom = False
        try:
            model_ft = train(dataloaders, dataset_sizes, model_ft, 
                             criterion, optimizer_ft, 
                             exp_lr_scheduler, device, 
                             num_epochs)
        except RuntimeError: # Out of memory
            oom = True
        
        # If out of memory, try a smaller batch size
        if oom:
            batch_size = int(batch_size/2)
            
            print('OOM - trying smaller batch size %d' %(batch_size))
                
            # Redefine smaller batch dataloader
            for x in ['train', 'val']: 
                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                              shuffle=True, num_workers=1)
                              for x in ['train', 'val']}
                
            model_ft = train(dataloaders, dataset_sizes, model_ft, 
                             criterion, optimizer_ft, 
                             exp_lr_scheduler, device, 
                             num_epochs)
        
    # Save model?
    if save:
        torch.save(model_ft.state_dict(), savepath)
        print('Model saved')
    
    
    return model_ft


# # Parameters
# seed = 998
# batch_size = 32
# num_epochs = 5
# pred_dir = 'predictions/'
# data_dir1 = 'data/cats_and_dogs'
# data_dir2 = 'data/imagenet-mini'
# image_size = 224
# # val_image_resize = 255
# ngpu = 1
# DS_FRACTION_CATS_DOGS = 0.1
# DS_FRACTION_NEITHER = 0.1


# if __name__=='__main__':
#     random.seed(seed)
    
#     plt.ion()   # interactive mode


#     # Data augmentation and normalization for training
#     # Just normalization for validation
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
    
    
#     # Load cats_and_dogs and imagenet 1000 datasets separately
#     image_datasets_cats_dogs = {x: datasets.ImageFolder(os.path.join(data_dir1, x),
#                                               data_transforms[x])
#                       for x in ['train', 'val']}
    
#     dataset_sizes_cats_dogs = {x: len(image_datasets_cats_dogs[x]) for x in ['train', 'val']}
    
#     image_datasets_neither = {x: ImageFolderCatAndDogLess(os.path.join(data_dir2, x),
#                                               transform=data_transforms[x])
#                       for x in ['train', 'val']}
    
#     dataset_sizes_neither = {x: len(image_datasets_neither[x]) for x in ['train', 'val']}
    
#     class_names = image_datasets_cats_dogs['train'].classes + ['neither']
   
    
#     # # Remove dogs and cats from imagenet dataset
#     # dog_idxs = range(151,269)
#     # cat_idxs = range(281,286)

        
#     for x in ['train', 'val']:
#         # for i in [dog_idxs, cat_idxs]:
#         #     image_datasets_neither[x].dropIndices(i)
#         # Set all remaining images as same "neither" target
#         image_datasets_neither[x].unifyTargets(2)
    
    
    
    
#     # If desired, random subsets of either dataset may be used (use 1. for all samples )
#     dataset_fraction_cats_dogs = 0.1
#     dataset_fraction_neither = 0.1
#     for x in ['train', 'val']:
#         # Cats and dogs
#         selected_samples = random.sample(range(dataset_sizes_cats_dogs[x]), 
#                                           int(dataset_sizes_cats_dogs[x] * dataset_fraction_cats_dogs))
#         image_datasets_cats_dogs[x] = torch.utils.data.Subset(image_datasets_cats_dogs[x], selected_samples)
#         # Neither
#         selected_samples = random.sample(range(dataset_sizes_neither[x]), 
#                                           int(dataset_sizes_neither[x] * dataset_fraction_neither))
#         image_datasets_neither[x] = torch.utils.data.Subset(image_datasets_neither[x], selected_samples)
    
    
    
#     print('Combining %d cat_and_dog and %d neither training images' % (
#         len(image_datasets_cats_dogs['train']), len(image_datasets_neither['train'])))
    
    
#     # Combine datasets
#     image_datasets = {x: torch.utils.data.ConcatDataset([image_datasets_cats_dogs[x], 
#                                             image_datasets_neither[x]])
#                       for x in ['train', 'val']}

    
    
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                                   shuffle=True, num_workers=1)
#                   for x in ['train', 'val']}
    
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    
    
#     device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    
#     print('Dataset sizes (train/val): {}/{} \t Classes: {}'.format(
#         dataset_sizes['train'], dataset_sizes['val'], 
#         [i for i in class_names]))
    
    
#     # Plot some training images
#     # Get a batch of training data
#     inputs, classes = next(iter(dataloaders['train']))
#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out)
#     # imshow(out, title=[class_names[x] for x in classes])

    
    
#     model_ft = models.resnet18(pretrained=True)
#     num_ftrs = model_ft.fc.in_features
#     # Here the size of each output sample is set to 2.
#     # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#     model_ft.fc = nn.Sequential(
#         nn.Dropout(0.0),
#         nn.Linear(num_ftrs, len(class_names))
#         )
    
#     model_ft = model_ft.to(device)
    
#     criterion = nn.CrossEntropyLoss()
    
#     # Observe that all parameters are being optimized
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
#     # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    
#     # Load previous model?
#     if load:
#         model.load_state_dict(torch.load(savepath))     
#         print('Model loaded')

    # model_ft = train(dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, 
    #                  exp_lr_scheduler, device, num_epochs=num_epochs)
    
#     # Save model?
#     if save:
#         torch.save(model_ft.state_dict(), savepath)
#         print('Model saved')

#     visualizeModel(model_ft, dataloader=dataloaders['val'])
    
    
#     # # Test on noise
#     noise = torch.randn((1,3,image_size,image_size))
#     img_noise = torchvision.utils.make_grid(noise)
#     imshow(img_noise)
#     print(model_ft(noise.to(device)))
    
    
#     # Make predictions on images in folder
#     pred_dataset = datasets.ImageFolder(pred_dir, data_transforms['val'])
    
#     pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=len(pred_dataset),
#                                                   shuffle=False, num_workers=1)

#     visualizeModel(model_ft, pred_dataloader)


#     plt.ioff()
#     plt.show()