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
        self.group = args.GROUP
        self.split_type = split_type
        self.transform = transform
        self.ids = list()
        self.video_list = list()
        self._make_lists()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def _make_lists(self):
        with open(self.root_path+'/contiguous_videos.json') as json_file:
            self.annoData = json.load(json_file)
        self.allLabels = self.annoData[self.group]['all_labels']
        self.numofClasses = len(self.allLabels )
        print(self.allLabels)
        print(self.numofClasses)
        self.videos = self.annoData[self.group][self.split_type]['videos']
        print(len(self.videos))
        
        for videoname in self.videos:
            print(videoname)
            self.video_list.append(videoname)
            list = os.listdir(os.path.join(self.root_path,'rgb-images',videoname))
            number_files = len(list)
            print(number_files)            
            for frame_num in range(1,number_files+1):
                if self.split_type == 'train_split':
                    label = self.annoData[self.group][self.split_type]['videos'][videoname]['frames']['{:05d}.jpg'.format(frame_num)]
                    #print(label)
                else:
                    label = []
                self.ids.append([videoname,label, frame_num])

    def __getitem__(self, index):
        #print(index)
        "Generates one sample of data"
        # Select sample
        videoname,label, frame_num = self.ids[index]
        X = []
        image = Image.open(os.path.join(self.root_path,'rgb-images', videoname, '{:05d}.jpg'.format(frame_num)))
        if self.transform is not None:
            image = self.transform(image)
        y = torch.LongTensor([int(label)])
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
