
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
                    help='MODE can be all, train, val, and test')
#  data loading argumnets
parser.add_argument('-b','--BATCH_SIZE', default=64, 
                    type=int, help='Batch size for training')
parser.add_argument('--VAL_BATCH_SIZE', default=64, 
                    type=int, help='Batch size for testing')
parser.add_argument('--TEST_BATCH_SIZE', default=64, 
                    type=int, help='Batch size for testing')
# Number of worker to load data in parllel
parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                    type=int, help='Number of workers used in dataloading')

parser.add_argument('--MAX_EPOCHS', default=1, 
                    type=int, help='Number of training epoc')
parser.add_argument('--VAL_EPOCHS', default=1, 
                    type=int, help='Number of training epoc')
parser.add_argument('-l','--learning_rate', 
                    default=0.001, type=float, help='initial learning rate')
parser.add_argument('--device', default='cuda', type=str, help='device type CPU or CUDA')

## Parse arguments
args = parser.parse_args()

class AutoDict(dict):
    def __missing__(self, k):
        self[k] = AutoDict()
        return self[k]

def train(model, device, train_loader, criterion,optimizer,scheduler,train_len,nb_classes, num_epochs):
    log_file = open(args.SAVE_ROOT+"/"+args.GROUP+"_training.log","w",1)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1,num_epochs+1):
        confusion_matrix = np.zeros((nb_classes, nb_classes),dtype=int)
        print('Training Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        log_file.write('Training Epoch {}/{}\n'.format(epoch, num_epochs))
        log_file.write('-' * 10+ '\n')
        
        running_loss = 0.0
        running_corrects = 0
        model.train()

        for batch_idx, (X, y,vid_names,frame_nums) in enumerate(train_loader):
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                X, y = X.cuda(), y.cuda()
                y = torch.squeeze(y)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
            # print(running_corrects)
            
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1
            #break
        scheduler.step()
        epoch_loss = running_loss / train_len
        epoch_acc = running_corrects.double() / train_len
        print('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Train', epoch_loss, epoch_acc))
        print(confusion_matrix)
        log_file.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Train', epoch_loss, epoch_acc))
        log_file.write(str(confusion_matrix))
        log_file.write('\n')

    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)    
    return model,optimizer,scheduler

def val(model, device, val_loader, criterion,optimizer,scheduler,val_len,nb_classes, num_epochs):
    log_file = open(args.SAVE_ROOT+"/"+args.GROUP+"_validation.log","w",1)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1,num_epochs+1):
        confusion_matrix = np.zeros((nb_classes, nb_classes),dtype=int)
        print('Validation/Self Training Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        log_file.write('Validation/Self Training Epoch {}/{}\n'.format(epoch, num_epochs))
        log_file.write('-' * 10+ '\n')
        
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
            
            model.eval()
            optimizer.zero_grad()
            sudo_outputs = model(X)
            _, sudo_y = torch.max(sudo_outputs, 1)
            model.train()
            with torch.set_grad_enabled(True):
                X, y = X.cuda(), sudo_y.cuda()
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            
            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
            # print(running_corrects)
            
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1

        scheduler.step()
        epoch_loss = running_loss / val_len
        epoch_acc = running_corrects.double() / val_len
        print('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Val', epoch_loss, epoch_acc))
        print(confusion_matrix)
        log_file.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Val', epoch_loss, epoch_acc))
        log_file.write(str(confusion_matrix))
        log_file.write('\n')
    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)    
    val_json = AutoDict()
    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
    
        model.eval()
        optimizer.zero_grad()
        outputs = model(X)
        _, label = torch.max(outputs, 1)
        for i in range(len(frame_nums)):
            val_json['validation_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = label[i].item()

    #print(val_json)
    with open(args.SAVE_ROOT+'/'+args.GROUP+'_validation_results.json', 'w') as outfile:
        json.dump(val_json, outfile)

    return model,optimizer,scheduler


def test(model, device, test_loader, test_len,nb_classes):
   
    test_json = AutoDict()
    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
    
        model.eval()
        optimizer.zero_grad()
        outputs = model(X)
        _, label = torch.max(outputs, 1)
        for i in range(len(frame_nums)):
            test_json['test_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = label[i].item()

    with open(args.SAVE_ROOT+'/'+args.GROUP+'_test_results.json', 'w') as outfile:
        json.dump(test_json, outfile)

    return model



img_x, img_y = 224, 224
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_params = {'batch_size': args.BATCH_SIZE, 'shuffle': True, 'num_workers': args.NUM_WORKERS}
train_set = DATASET(args, 'train_split',transform=transform)
train_loader = data.DataLoader(train_set, **train_params)
train_len = train_set.dataLen

val_params = {'batch_size': args.VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS}
val_set = DATASET(args,'validation_split',transform=transform)
val_loader = data.DataLoader(val_set, **val_params)
val_len = val_set.dataLen

test_params = {'batch_size': args.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS}
test_set = DATASET(args,'test_split',transform=transform)
test_loader = data.DataLoader(test_set, **test_params)
test_len = test_set.dataLen

numofClasses = train_set.numofClasses
classesList = train_set.allLabels



epochs = args.MAX_EPOCHS
val_epochs = args.VAL_EPOCHS
device = torch.device(args.device)
learning_rate = args.learning_rate


model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, numofClasses)

model = model_ft.to(device)
print(model)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = nn.DataParallel(model)

# start training
# train, validation/self training and testing model

if args.MODE == 'all' or args.MODE == 'train':
    model,optimizer,scheduler = train(model, device, train_loader, criterion,optimizer,scheduler,train_len,numofClasses,epochs)
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, args.SAVE_ROOT+'/'+args.GROUP+'_trained_model.pth')


if args.MODE == 'all' or args.MODE == 'val':
    checkpoint = torch.load(args.SAVE_ROOT+'/'+args.GROUP+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model,optimizer,scheduler = val(model, device, val_loader, criterion,optimizer,scheduler,val_len,numofClasses,val_epochs)
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, args.SAVE_ROOT+'/'+args.GROUP+'_validation_self_trained_model.pth')

if args.MODE == 'all' or args.MODE == 'test':
    checkpoint = torch.load(args.SAVE_ROOT+'/'+args.GROUP+'_validation_self_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = test(model, device, test_loader, test_len,numofClasses)

