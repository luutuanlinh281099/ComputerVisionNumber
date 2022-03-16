import torch
from PIL import Image
import cv2
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
path_cur=os.path.dirname(os.path.abspath(__file__))

class num_ocr:
    def __init__(self,device=torch.device("cuda")):
        self.output_dim=10
        self.device=device
     
        self.load_model()
        self.maps1={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
        self.maps1=dict([(value, key) for key, value in self.maps1.items()]) 
     

    def create_resnet9_model(self,output_dim: int = 1) -> nn.Module:
        model = ResNet(BasicBlock, [1, 1, 1, 1])
        in_features = model.fc.in_features
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(in_features, output_dim)
        return model
    
    def create_model(self, output_dim: int = 10) -> nn.Module:
        model = models.quantization.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_dim)
        # model.to(device)
        return model

    def load_model(self):
        self.model = self.create_model(output_dim=self.output_dim)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(path_cur,'weights/num_last.pth')))
        self.model.eval()
        maps=open(os.path.join(path_cur,'weights/maps.txt'),"r").read().split("\n")
        self.maps={x.split(" ")[0] : x.split(" ")[1] for x in maps} # id : labels


    def predict(self,data):
        output=F.softmax(self.model(data.to(self.device)))
        texts=""
        for x  in output:
            texts+=self.maps[self.maps1[int(x.argmax(0))]]
        return texts
   
