import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import json

class DATASET(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, args,split_type,transform=None):
        "Initialization"
        self.root_path = args.DATA_ROOT
        self.split_type = split_type
        self.transform = transform
        self.ids = list()
        self.nb_samples = np.zeros(9,dtype=int)
        self.video_list = list()
        self._make_lists()
        self.dataLen = len(self.ids)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self):
        with open(self.root_path+'/Annotation.json') as json_file:
            self.annoData = json.load(json_file)
        self.allLabels = self.annoData['contiguous_videos']['all_labels']
        self.numofClasses = len(self.allLabels)
        print(self.allLabels)
        print(self.numofClasses)
        inc = 0 

        for category in ['contiguous_videos', 'short_gap', 'long_gap']:
            videos = self.annoData[category][self.split_type]['videos']
            print(len(videos))
            for videoname in videos:
                print(videoname)
                self.video_list.append(videoname)
                list = os.listdir(os.path.join(self.root_path,category,'rgb-images',videoname))
                number_files = len(list)
                print(number_files)            
                for frame_num in range(1,number_files+1):

                    if self.split_type == 'train_split':
                        label = self.annoData[category][self.split_type]['videos'][videoname]['frames']['{:05d}.jpg'.format(frame_num)]
                        # if label !=0:
                        lab = label
                        self.nb_samples[lab] = self.nb_samples[lab]+1
                        self.ids.append([category,videoname,lab, frame_num])
                        #print(label)
                    else:
                        label = []
                        self.ids.append([category,videoname,label, frame_num])

                    

    def __getitem__(self, index):
        #print(index)
        "Generates one sample of data"
        # Select sample
        category,videoname,label, frame_num = self.ids[index]
        X = []
        image = Image.open(os.path.join(self.root_path,category,'rgb-images', videoname, '{:05d}.jpg'.format(frame_num)))
        if self.transform is not None:
            image = self.transform(image)
        if self.split_type == 'train_split':
            y = torch.LongTensor([int(label)])
        else:
            y = torch.LongTensor([label])
        return image, y,videoname,frame_num,category


    def custum_collate(batch):
        
        images = []
        labels = []
        videonames = []
        frame_nums = []
        categories = []
 
        for sample in batch:
            images.append(sample[0])
            labels.append(sample[1])
            videonames.append(sample[2])
            frame_nums.append(sample[3])
            categories.append(sample[4])          
        return images, labels,videonames,frame_nums,categories

class DATASET_VAL_TEST(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, args,category,vid_name,start_sess,end_sess,transform=None):
        "Initialization"
        self.root_path = args.DATA_ROOT
        self.category = category
        self.vid_name = vid_name
        self.start_sess = start_sess
        self.end_sess = end_sess
        self.transform = transform
        self.ids = list()
        self._make_lists()
        self.dataLen = len(self.ids)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self):    
        for frame_num in range(self.start_sess,self.end_sess):
            label = []
            self.ids.append([self.category,self.vid_name,label, frame_num])

    def __getitem__(self, index):
        #print(index)
        "Generates one sample of data"
        # Select sample
        category,videoname,label, frame_num = self.ids[index]
        X = []
        # print(category)
        # print(videoname)
        image = Image.open(os.path.join(self.root_path,category,'rgb-images', videoname, '{:05d}.jpg'.format(frame_num)))
        #print(image.size)
        if self.transform is not None:
            image = self.transform(image)
        y = torch.LongTensor([label])
        return image, y,videoname,frame_num


    def custum_collate(batch):
        
        images = []
        labels = []
        videonames = []
        frame_nums = []

        for sample in batch:
            images.append(sample[0])
            labels.append(sample[1])
            videonames.append(sample[2])
            frame_nums.append(sample[3])

            
        return images, labels,videonames,frame_nums




