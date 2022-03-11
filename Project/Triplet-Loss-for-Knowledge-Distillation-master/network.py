import torch
import torch.nn as nn
# from torchvision.models import resnet, densenet, vgg, inception
from torchvision import models
import numpy as np
# from teacher_train import Only_teacher

class Net_teacher(nn.Module):
    def __init__(self):
        super(Net_teacher,self).__init__()
        
#         resnet = models.resnet50(pretrained=True)
#         num_input_ftrs = resnet.fc.in_features

#         modules = list(resnet.children())[:-1]
#         resnet = nn.Sequential(*modules)
#         for param in resnet.parameters():
#             param.requires_grad = False   
                
#         self.net = resnet
#         self.net.fc = nn.Linear(num_input_ftrs , 10)
#         for param in self.net.fc.parameters():
#             param.requires_grad = True

        self.net = Only_teacher(10)
        self.net.load_state_dict(torch.load("only_teacher_model.pth"))
            
    def forward(self, x):
        return self.net(x)

class Net_student(nn.Module):
    def __init__(self):
        super(Net_student,self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        num_input_ftrs = resnet.fc.in_features

#         modules = list(resnet.children())[:-1]
#         resnet = nn.Sequential(*modules)
        for param in resnet.parameters():
            param.requires_grad = True   
                
        self.net = resnet
        self.net.fc = nn.Linear(num_input_ftrs , 10)
        for param in self.net.fc.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.net(x)
    
class Only_teacher(nn.Module):
    def __init__(self,n_class):
        self.n_class = n_class
        super(Only_teacher,self).__init__()

        resnet = models.resnet50(pretrained=True)
        num_input_ftrs = resnet.fc.in_features
        self.pretrained_model = nn.Sequential(*(list(resnet.children())[:-1]))
        self.linear = nn.Linear(num_input_ftrs , self.n_class)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False        
        
        for param in self.linear.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        features = self.pretrained_model(x)
        features = features.reshape(features.size(0),-1)
        output = self.linear(features)
        return output
  
 # class Net_teacher(nn.Module):
#     def __init__(self):
#         super(Net_teacher, self).__init__()
#         self.conv1 = nn.Conv2d(3,32,3, padding=1)
#         self.conv2 = nn.Conv2d(32,32,3, padding=1)
#         self.conv3 = nn.Conv2d(32,64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64,64, 3, padding=1)
#         self.conv5 = nn.Conv2d(64,128, 3, padding=1)
#         self.fc1 = nn.Linear(128*4*4, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 10)

#         self.batchnorm1 = nn.BatchNorm2d(32)
#         self.batchnorm2 = nn.BatchNorm2d(32)
#         self.batchnorm3 = nn.BatchNorm2d(64)
#         self.batchnorm4 = nn.BatchNorm2d(64)
#         self.batchnorm5 = nn.BatchNorm2d(128)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()


#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.batchnorm1(x)
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.batchnorm2(x)
#         x = self.pool(x)

#         x = self.relu(self.conv3(x))
#         x = self.batchnorm3(x)
#         x = self.relu(self.conv4(x))
#         x = self.batchnorm4(x)
#         x = self.relu(self.conv5(x))
#         x = self.batchnorm5(x)
#         x = self.pool(x)

#         x = x.view(-1, 128 * 4 * 4)
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.dropout(self.relu(self.fc2(x)))
#         x = self.fc3(x)
#         return x


# class Net_student(nn.Module):
#     def __init__(self):
#         super(Net_student, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.fc1 = nn.Linear(64 * 4 * 4, 128)
#         self.fc2 = nn.Linear(128, 10)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)

#         x = self.relu(self.conv2(x))
#         x = self.pool(x)

#         x = self.relu(self.conv3(x))
#         x = self.pool(x)
        
#         x = x.view(-1, 64 * 4 * 4)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x