
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
from efficientnet_pytorch import EfficientNet

from losses import CB_loss
from dataset import DATASET
from dataset import DATASET_VAL_TEST

Mtype = "bg_weights_cb"


parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
parser.add_argument('--DATA_ROOT', help='Location to root directory for dataset reading') # Data/contiguous_videos
parser.add_argument('--SAVE_ROOT', help='Location to root directory for saving checkpoint models') # Data/contiguous_videos

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


def train(model, device, train_loader, criterion,optimizer,scheduler,train_len,nb_classes,samples_per_cls, num_epochs):
    log_file = open(args.SAVE_ROOT+"/"+Mtype+"_training.log","w",1)
    since = time.time()
    beta = 0.9999
    gamma = 2.0
    loss_type = "focal"

    for epoch in range(1,num_epochs+1):
        confusion_matrix = np.zeros((nb_classes, nb_classes),dtype=int)
        print('Training Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        log_file.write('Training Epoch {}/{}\n'.format(epoch, num_epochs))
        log_file.write('-' * 10+ '\n')
        running_loss = 0.0
        running_corrects = 0
        model.train()
        #print(len(train_loader))
        for batch_idx, (X, y,vid_names,frame_nums,_) in enumerate(train_loader):
            # print(batch_idx)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                X, y = X.cuda(), y.cuda()
                y = torch.squeeze(y)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                loss = CB_loss(y.cpu(), outputs.cpu(), samples_per_cls, nb_classes,loss_type, beta, gamma)
                #loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
            # print(running_corrects)
            
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t, p] += 1
            
        scheduler.step()
        epoch_loss = running_loss / train_len
        epoch_acc = running_corrects.double() / train_len
        print('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Train', epoch_loss, epoch_acc))
        print(confusion_matrix)
        log_file.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format('Train', epoch_loss, epoch_acc))
        log_file.write(str(confusion_matrix))
        log_file.write('\n')
   
    return model,optimizer,scheduler

def evaluate_val(model, device, criterion,optimizer,scheduler,val_loader, num_epochs):
    val_json = AutoDict()
    for epoch in range(1,num_epochs+1):
        print('Evaluation without self-training Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        for batch_idx, (X, y,vid_names,frame_nums,category) in enumerate(val_loader):
            model.eval()
            optimizer.zero_grad()
            outputs = model(X)
            _, label = torch.max(outputs, 1)
            for i in range(len(frame_nums)):
                lab = label[i].item()
                val_json[category[i]]['validation_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab
        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_val_predictions_nost.json', 'w') as outfile:
            json.dump(val_json, outfile)   
    return model

def evaluate_test(model, device, criterion,optimizer,scheduler,test_loader, num_epochs):
    test_json = AutoDict()
    for epoch in range(1,num_epochs+1):
        print('Evaluation without self-training Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        for batch_idx, (X, y,vid_names,frame_nums,category) in enumerate(test_loader):
            model.eval()
            optimizer.zero_grad()
            outputs = model(X)
            _, label = torch.max(outputs, 1)
            for i in range(len(frame_nums)):
                lab = label[i].item()
                test_json[category[i]]['test_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab
        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_test_predictions_nost.json', 'w') as outfile:
            json.dump(test_json, outfile)
    return model


def self_val(model, device, criterion,optimizer,scheduler,val_params,transform,nb_classes, samples_per_cls,num_epochs,thresh=0.4):
    val_json = AutoDict()
    beta = 0.9999
    gamma = 2.0
    loss_type = "focal"
    
    for epoch in range(1,num_epochs+1):
        print('Self Training Val-Fold Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        with open(args.DATA_ROOT +'/Annotation.json') as json_file:
            annoData = json.load(json_file)
        for category in ['contiguous_videos', 'short_gap', 'long_gap']:
            val_videos = annoData[category]['validation_split']['videos']
            for val_vid_name in val_videos:
                val_model = copy.deepcopy(model)
                num_sessions = 5
                val_sessions_list = []
                val_len = len(os.listdir(os.path.join(args.DATA_ROOT,category,'rgb-images',val_vid_name)))
                ses = int(9000/num_sessions)
                for i in range(1,9000,ses):
                    val_sessions_list.append(i)
                val_sessions_list.append(val_len)
                print("Validation video: ", val_vid_name)
                ##########Validation
                for val_inc in range(len(val_sessions_list)-1):
                    print("Session: ", val_inc)
                    val_start_sess =  val_sessions_list[val_inc]
                    val_end_sess =  val_sessions_list[val_inc+1]
                    val_set = DATASET_VAL_TEST(args,category,val_vid_name,val_start_sess,val_end_sess,transform=transform)
                    val_loader = data.DataLoader(val_set, **val_params)
                    #######Self-training
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        sudo_outputs = val_model(X)
                        _, sudo_y = torch.max(sudo_outputs, 1)
                        for i in range(len(frame_nums)):
                            if sudo_outputs[i][sudo_y[i].item()] > thresh:                           
                                val_model.train()
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(True):
                                    X_ = torch.unsqueeze(X[i].cuda(),0)
                                    y_ = torch.tensor([sudo_y[i].item()]).cuda()
                                    outputs = val_model(X_)
                                    _, preds = torch.max(outputs, 1)
                                    loss = CB_loss(y_.cpu(), outputs.cpu(), samples_per_cls, nb_classes,loss_type, beta, gamma)
                                    #loss = criterion(outputs, y_)
                                    loss.backward()
                                    optimizer.step()
                    #######Evaluation
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        outputs = val_model(X)
                        _, label = torch.max(outputs, 1)
                        for i in range(len(frame_nums)):
                            lab = label[i].item()
                            val_json[category]['validation_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab
        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_self_val_predictions_st.json', 'w') as outfile:
            json.dump(val_json, outfile)
    return model


def self_test(model, device, criterion,optimizer,scheduler,test_params,transform,nb_classes, samples_per_cls,num_epochs,thresh=0.4):
    test_json = AutoDict()
    beta = 0.9999
    gamma = 2.0
    loss_type = "focal"
    for epoch in range(1,num_epochs+1):
        print('Self Training Test-Fold Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        with open(args.DATA_ROOT +'/Annotation.json') as json_file:
            annoData = json.load(json_file)
        for category in ['contiguous_videos', 'short_gap', 'long_gap']:
            test_videos = annoData[category]['test_split']['videos']
            for test_vid_name in test_videos:
                test_model = copy.deepcopy(model)
                num_sessions = 5
                test_sessions_list = []
                test_len = len(os.listdir(os.path.join(args.DATA_ROOT,category,'rgb-images',test_vid_name)))
                ses = int(9000/num_sessions)
                for i in range(1,9000,ses):
                    test_sessions_list.append(i)
                test_sessions_list.append(test_len)
                ##########Testing
                print("Testing video: ", test_vid_name)   
                for test_inc in range(len(test_sessions_list)-1):
                    print("Session: ", test_inc)
                    test_start_sess =  test_sessions_list[test_inc]
                    test_end_sess =  test_sessions_list[test_inc+1]
                    test_set = DATASET_VAL_TEST(args,category,test_vid_name,test_start_sess,test_end_sess,transform=transform)
                    test_loader = data.DataLoader(test_set, **test_params)
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
                        test_model.eval()
                        optimizer.zero_grad()
                        sudo_outputs = test_model(X)
                        _, sudo_y = torch.max(sudo_outputs, 1)
                        for i in range(len(frame_nums)):
                            if sudo_outputs[i][sudo_y[i].item()] > thresh:                           
                                test_model.train()
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(True):
                                    X_ = torch.unsqueeze(X[i].cuda(),0)
                                    y_ = torch.tensor([sudo_y[i].item()]).cuda()
                                    outputs = test_model(X_)
                                    _, preds = torch.max(outputs, 1)
                                    loss = CB_loss(y_.cpu(), outputs.cpu(), samples_per_cls, nb_classes,loss_type, beta, gamma)
                                    #loss = criterion(outputs, y_)
                                    loss.backward()
                                    optimizer.step()
                    #######Evaluation
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
                        test_model.eval()
                        optimizer.zero_grad()
                        outputs = test_model(X)
                        _, label = torch.max(outputs, 1)
                        for i in range(len(frame_nums)):
                            lab = label[i].item()
                            test_json[category]['test_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab
        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_self_test_predictions_st.json', 'w') as outfile:
            json.dump(test_json, outfile)                
    return model


def self_val_test_combine(model, device, criterion,optimizer,scheduler,val_params,test_params,transform,nb_classes, samples_per_cls,num_epochs,thresh=0.4):
    log_file = open(args.SAVE_ROOT+"/"+Mtype+"_validation.log","w",1)
    val_json = AutoDict()
    test_json = AutoDict()
    beta = 0.9999
    gamma = 2.0
    loss_type = "focal"
    
    for epoch in range(1,num_epochs+1):
        confusion_matrix = np.zeros((nb_classes, nb_classes),dtype=int)
        print('Self Training Validation/Test Fold combine Epoch {}/{}\n'.format(epoch, num_epochs))
        print('-' * 10)
        log_file.write('Validation/Self Training Epoch {}/{}\n'.format(epoch, num_epochs))
        log_file.write('-' * 10+ '\n')
        running_loss = 0.0
        running_corrects = 0
        with open(args.DATA_ROOT +'/Annotation.json') as json_file:
            annoData = json.load(json_file)
        for category in ['contiguous_videos', 'short_gap', 'long_gap']:
            val_videos = annoData[category]['validation_split']['videos']
            test_videos = annoData[category]['test_split']['videos']
            
            for val_vid_name,test_vid_name in zip(val_videos,test_videos):
                val_model = copy.deepcopy(model)
                num_sessions = 5
                val_sessions_list = []
                test_sessions_list = []
                val_len = len(os.listdir(os.path.join(args.DATA_ROOT,category,'rgb-images',val_vid_name)))
                test_len = len(os.listdir(os.path.join(args.DATA_ROOT,category,'rgb-images',test_vid_name)))
                ses = int(9000/num_sessions)
                for i in range(1,9000,ses):
                    val_sessions_list.append(i)
                    test_sessions_list.append(i)
                val_sessions_list.append(val_len)
                test_sessions_list.append(test_len)
                print("Validation video: ", val_vid_name)
                ##########Validation
                for val_inc in range(len(val_sessions_list)-1):
                    print("Session: ", val_inc)
                    val_start_sess =  val_sessions_list[val_inc]
                    val_end_sess =  val_sessions_list[val_inc+1]
                    val_set = DATASET_VAL_TEST(args,category,val_vid_name,val_start_sess,val_end_sess,transform=transform)
                    val_loader = data.DataLoader(val_set, **val_params)
                    #######Self-training
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        sudo_outputs = val_model(X)
                        _, sudo_y = torch.max(sudo_outputs, 1)
                        for i in range(len(frame_nums)):
                            if sudo_outputs[i][sudo_y[i].item()] > thresh:                           
                                val_model.train()
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(True):
                                    X_ = torch.unsqueeze(X[i].cuda(),0)
                                    y_ = torch.tensor([sudo_y[i].item()]).cuda()
                                    outputs = val_model(X_)
                                    _, preds = torch.max(outputs, 1)
                                    loss = CB_loss(y_.cpu(), outputs.cpu(), samples_per_cls, nb_classes,loss_type, beta, gamma)
                                    #loss = criterion(outputs, y_)
                                    loss.backward()
                                    optimizer.step()
                    #######Evaluation
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(val_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        outputs = val_model(X)
                        _, label = torch.max(outputs, 1)
                        for i in range(len(frame_nums)):
                            lab = label[i].item()
                            val_json[category]['validation_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab

                ##########Testing
                print("Testing video: ", test_vid_name)   
                for test_inc in range(len(test_sessions_list)-1):
                    print("Session: ", test_inc)
                    test_start_sess =  test_sessions_list[test_inc]
                    test_end_sess =  test_sessions_list[test_inc+1]
                    test_set = DATASET_VAL_TEST(args,category,test_vid_name,test_start_sess,test_end_sess,transform=transform)
                    test_loader = data.DataLoader(test_set, **test_params)
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        sudo_outputs = val_model(X)
                        _, sudo_y = torch.max(sudo_outputs, 1)
                        for i in range(len(frame_nums)):
                            if sudo_outputs[i][sudo_y[i].item()] > thresh:                           
                                val_model.train()
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(True):
                                    X_ = torch.unsqueeze(X[i].cuda(),0)
                                    y_ = torch.tensor([sudo_y[i].item()]).cuda()
                                    outputs = val_model(X_)
                                    _, preds = torch.max(outputs, 1)
                                    loss = CB_loss(y_.cpu(), outputs.cpu(), samples_per_cls, nb_classes,loss_type, beta, gamma)
                                    #loss = criterion(outputs, y_)
                                    loss.backward()
                                    optimizer.step()
                    #######Evaluation
                    for batch_idx, (X, y,vid_names,frame_nums) in enumerate(test_loader):
                        val_model.eval()
                        optimizer.zero_grad()
                        outputs = val_model(X)
                        _, label = torch.max(outputs, 1)
                        for i in range(len(frame_nums)):
                            lab = label[i].item()
                            test_json[category]['test_split']['videos'][vid_names[i]]['frames']['{:05d}.jpg'.format(frame_nums[i].item())] = lab


        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_self_test_combine_predictions_st.json', 'w') as outfile:
            json.dump(test_json, outfile)
        with open(args.SAVE_ROOT+'/'+Mtype+'_Activity_challenge_self_val_combine_predictions_st.json', 'w') as outfile:
            json.dump(val_json, outfile)
        print(confusion_matrix)
                
    return model


img_x, img_y = 600, 600
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_params = {'batch_size': args.BATCH_SIZE, 'shuffle': True, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
train_set = DATASET(args, 'train_split',transform=transform)
train_loader = data.DataLoader(train_set, **train_params)
train_len = train_set.dataLen

numofClasses = train_set.numofClasses
classesList = train_set.allLabels


epochs = args.MAX_EPOCHS
val_epochs = args.VAL_EPOCHS
device = torch.device(args.device)
learning_rate = args.learning_rate


# model_ft = models.resnet152(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, numofClasses)

model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=numofClasses)
model = model.to(device)
print(model)

samples_per_cls = train_set.nb_samples
nb_samples = train_set.nb_samples
print(nb_samples)
weights = 1. / nb_samples
print(weights)

class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = nn.DataParallel(model)

if not os.path.exists(args.SAVE_ROOT):
    os.makedirs(args.SAVE_ROOT)

# start training
# train, validation/self training and testing model

if args.MODE == 'all' or args.MODE == 'train':
    model,optimizer,scheduler = train(model, device, train_loader, criterion,optimizer,scheduler,train_len,numofClasses,samples_per_cls,epochs)
    torch.save({'model_state_dict':model.state_dict()}, args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')



if args.MODE == 'all' or args.MODE == 'evaluate_val':
    val_params = {'batch_size': args.VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    val_set = DATASET(args,'validation_split',transform=transform)
    val_loader = data.DataLoader(val_set, **val_params)

    checkpoint = torch.load(args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  
    model =evaluate_val(model,device, criterion,optimizer,scheduler,val_loader,val_epochs) 

if args.MODE == 'all' or args.MODE == 'evaluate_test':
    test_params = {'batch_size': args.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    test_set = DATASET(args,'test_split',transform=transform)
    test_loader = data.DataLoader(test_set, **test_params)

    checkpoint = torch.load(args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  
    model =evaluate_test(model,device, criterion,optimizer,scheduler,test_loader,val_epochs) 


if args.MODE == 'all' or args.MODE == 'self_val':
    val_params = {'batch_size': args.VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    checkpoint = torch.load(args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])                          
    model = self_val(model,device, criterion,optimizer,scheduler,val_params,transform,numofClasses,samples_per_cls,val_epochs) 


if args.MODE == 'all' or args.MODE == 'self_test':
    test_params = {'batch_size': args.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    checkpoint = torch.load(args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])                          
    model =self_test(model,device, criterion,optimizer,scheduler,test_params,transform,numofClasses,samples_per_cls,val_epochs) 

if args.MODE == 'all' or args.MODE == 'self_val_test_combine':
    val_params = {'batch_size': args.VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    test_params = {'batch_size': args.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': args.NUM_WORKERS, 'drop_last': True}
    checkpoint = torch.load(args.SAVE_ROOT+'/'+Mtype+'_trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])                          
    model =self_val_test_combine(model,device, criterion,optimizer,scheduler,val_params,test_params,transform,numofClasses,samples_per_cls,val_epochs) 


