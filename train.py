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

# Show sample images on input Tensors
def imshow(inp, title=None, label=None):
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
        

# Train model
def train(dataloaders, 
          dataset_sizes, 
          model, 
          criterion, 
          optimizer, 
          scheduler, 
          device, 
          num_epochs):
    
    since = time.time()
    
    # Class to keep track of loss and accuracy 
    # (to be replaced with a Tensorboard implementation)
    training_tracker = TrainingTracker()

    # Keep track of best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Calculate grad only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize if in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            
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

            # Backup and copy the model if it's superior
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
def imagePredictions(model, 
                     image_size, 
                     pred_dir, 
                     pred_threshold=None, 
                     pred_label_considered=0, 
                     batch_size=None):
    # Transforms are the same as 'val' transforms during train(...)
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
            vals, preds = getPredictedLabel(outputs, 
                                            threshold=pred_threshold, 
                                            label_considered=pred_label_considered)

            for j in range(inputs.size()[0]):
                label = vals[j].item()
                # include the predicted label if it is cat or dog
                if class_names[preds[j]] != 'neither':
                    label = class_names[preds[j]] 
                else:
                    label = None
                    
                imshow(inputs.cpu().data[j], label=label)


'''
Get the predicted label for a model output
threshold decides which one of two modes to be used:
1) If threshold is not set, the maximum value of "cat", "dog" and "neither" 
is returned (used for debugging)
2) If threshold is set, label_considered (default 0 ("cat")) is returned if 
the output value is above threshold, otherwise 2 ("neither") is returned
This would act as a "cat detector" and return neither ALSO if dog has the 
highest output value.
'''
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
        

# simple custom class for "neither cat nor dog" target
class ImageFolderCatAndDogLess(datasets.ImageFolder):
    
    # Reduce number of targets to a single integer ("neither cat nor dog")
    def unifyTargets(self, out_target: int) -> None:
        self.samples = [(path, out_target) for path, _ in self.samples]
        self.targets = [s[1] for s in self.samples]        
            
# Keep track of performance during training and plot it afterwards
class TrainingTracker:
    
    def __init__(self):
        self.losses = {x: [] for x in ['train', 'val']}
        self.accuracies  = {x: [] for x in ['train', 'val']}
    
    # Add statistics
    def add(self, phase, loss, accuracy):
        self.losses[phase].append(loss)
        self.accuracies[phase].append(accuracy)
    
    # Plot statistics
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


# Useful function to perform all steps in setting up a model, possibly also
# training it
def setupModel(seed, 
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
    
    
    
    
    # train_model determines if training or validation datasets are loaded
    # Pick False if you are only interested in predictions on video
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
            n_samples_cats_dogs = int(dataset_sizes_cats_dogs[x]*dataset_fraction_cats_dogs)
            selected_samples = random.sample(range(dataset_sizes_cats_dogs[x]), 
                                             n_samples_cats_dogs)
            image_datasets_cats_dogs[x] = torch.utils.data.Subset(image_datasets_cats_dogs[x], 
                                                                  selected_samples)
            
            # Neither
            n_samples_neither = int(dataset_sizes_neither[x]*dataset_fraction_neither)
            selected_samples = random.sample(range(dataset_sizes_neither[x]), 
                                             n_samples_neither)
            image_datasets_neither[x] = torch.utils.data.Subset(image_datasets_neither[x], 
                                                                selected_samples)



        print('Combining {} cat_and_dog and {} neither training images'.format(
            len(image_datasets_cats_dogs['train']), len(image_datasets_neither['train'])))
    
    
        # Combine datasets
        image_datasets = {x: torch.utils.data.ConcatDataset([image_datasets_cats_dogs[x], 
                                                             image_datasets_neither[x]])
                          for x in ['train', 'val']}
    
        
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=batch_size,
                                                      shuffle=True, 
                                                      num_workers=1)
                      for x in ['train', 'val']}
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        
        
        # Plot some training images
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs, nrow=4)
        imshow(out)
        
    
        print('Dataset sizes (train/val): {}/{}'.format(
            dataset_sizes['train'], dataset_sizes['val']))
    
    
    # Define model regardless if train or video predictions
    # Transfer weights from a pre-trained resnet18 model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Output layer is the number of classes
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    
    model_ft = model_ft.to(device)
    

    # Load previous model?
    if load:
        model_ft.load_state_dict(torch.load(savepath))     
        print('Model loaded')
    
    
    if train_model:
        
        # Use cross entropy loss for classification problem
        criterion = nn.CrossEntropyLoss()
        # Use stochastic gradient descent optimizer
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        
        # Decay learning rate by a factor of 0.1 every 5 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        
        # Warning in order to not waste calculation time
        if not save:
            print('Training WITHOUT saving model')
        
        # TRAINING START
        # If we run out of memory...
        oom = False
        try:
            model_ft = train(dataloaders, 
                             dataset_sizes, 
                             model_ft, 
                             criterion, 
                             optimizer_ft, 
                             exp_lr_scheduler, 
                             device, 
                             num_epochs)
        except RuntimeError: # Out of memory
            oom = True
        
        # ...try with smaller batch size
        if oom:
            batch_size = int(batch_size/2)
            
            print('OOM - trying smaller batch size {}'.format(batch_size))
                
            # Redefine smaller batch dataloader
            for x in ['train', 'val']: 
                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                              batch_size=batch_size,
                                                              shuffle=True, 
                                                              num_workers=1)
                               for x in ['train', 'val']}
                
            model_ft = train(dataloaders, 
                             dataset_sizes, 
                             model_ft, 
                             criterion, 
                             optimizer_ft, 
                             exp_lr_scheduler, 
                             device, 
                             num_epochs)
            
        # TRAINING FINISH
    
    
    # Save model?
    if save:
        torch.save(model_ft.state_dict(), savepath)
        print('Model saved')
    
    
    return model_ft
