
import cv2
import numpy as np
import json
import math
import os
import argparse
import time
import copy

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from dataset import DATASET

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
parser.add_argument('--DATA_ROOT', help='Location to root directory for dataset reading') # Data/contiguous_videos
parser.add_argument('--SAVE_ROOT', help='Location to root directory for saving checkpoint models') # Data/contiguous_videos
parser.add_argument('--GROUP', default='contiguous_videos', 
                    type=str, help='Group types are contiguous_videos, short_gap, and long_gap')

parser.add_argument('--MODE', default='train',
                    help='MODE can be train, val, and test')
#  data loading argumnets
parser.add_argument('-b','--BATCH_SIZE', default=4, 
                    type=int, help='Batch size for training')
parser.add_argument('--VAL_BATCH_SIZE', default=1, 
                    type=int, help='Batch size for testing')
parser.add_argument('--TEST_BATCH_SIZE', default=1, 
                    type=int, help='Batch size for testing')
# Number of worker to load data in parllel
parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                    type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--OPTIM', default='SGD', 
                    type=str, help='Optimiser type')
parser.add_argument('--MAX_EPOCHS', default=1, 
                    type=int, help='Number of training epoc')
parser.add_argument('-l','--learning_rate', 
                    default=0.001, type=float, help='initial learning rate')
parser.add_argument('--device', default='cuda', type=str, help='device type CPU or CUDA')

## Parse arguments
args = parser.parse_args()





def train(model, device, train_loader, criterion,optimizer_ft,exp_lr_scheduler, epoch):
        
    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(train_loader):
        # distribute data to device
        print(X.shape)
        print(y)
        print(vid_names)
        print(frame_nums)
        break
        #X, y = X.to(device), y.to(device).view(-1, )

        
    return model

def val(model, device, val_loader, optimizer, epoch):
        
    for batch_idx,(X, y,vid_names,frame_nums)in enumerate(val_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        
    return losses, scores


def test(model, device, test_loader, optimizer, epoch):
        
    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        
    return losses, scores



img_x, img_y = 300, 300
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_params = {'batch_size': args.BATCH_SIZE, 'shuffle': True, 'num_workers': args.NUM_WORKERS}
train_set = DATASET(args, 'train_split',transform=transform)
train_loader = data.DataLoader(train_set, **train_params)

val_params = {'batch_size': args.VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS}
val_set = DATASET(args,'validation_split',transform=transform)
val_loader = data.DataLoader(val_set, **val_params)
 

test_params = {'batch_size': args.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS}
test_set = DATASET(args,'test_split',transform=transform)
test_loader = data.DataLoader(test_set, **test_params)
 
numofClasses = train_set.numofClasses
classesList = train_set.allLabels



epochs = args.MAX_EPOCHS
device = torch.device(args.device)
learning_rate = args.learning_rate


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, numofClasses)

model = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



# start training
for epoch in range(epochs):
    # train, test model
    model = train(model, device, train_loader, criterion,optimizer_ft,exp_lr_scheduler, epoch)
    
#epoch_test_loss, epoch_test_score = validation(ROADNet, device, optimizer, valid_loader,epoch)
