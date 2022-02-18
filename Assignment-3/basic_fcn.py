import torch.nn as nn
from torchvision import models




class FCN_mxp(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.drop    = nn.Dropout(p=0.5, inplace=False)
        self.relu    = nn.LeakyReLU(0.1)
        self.maxpool1= nn.MaxPool2d(2, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.maxpool2= nn.MaxPool2d(3, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.maxpool3= nn.MaxPool2d(4, padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.unpool1 = nn.MaxUnpool2d(2, padding=0)
        self.unpool2 = nn.MaxUnpool2d(3, padding=0)
        self.unpool3 = nn.MaxUnpool2d(4, padding=0)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.deconv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):

        x1  = self.bnd1(self.relu(self.conv1(x)))
        print("x1---->", x1.shape)
        x2  = self.bnd2(self.relu(self.conv2(x1)))
        print("x2---->", x2.shape)
        xm1,ind1 = self.maxpool1(x2)
        print("xm1---->", xm1.shape)
        x3  = self.bnd3(self.relu(self.conv3(xm1)))
        print("x3---->", x3.shape)
        x4  = self.bnd4(self.relu(self.conv4(x3)))
        print("x4---->", x4.shape)
        xm2,ind2 = self.maxpool1(x4)
        print("xm2---->", xm2.shape)
        xd1 = self.drop(xm2)
        print("xd1---->", xd1.shape)
        out_encoder = self.bnd5(self.relu(self.conv5(xd1)))
        print("out encoder ---->", out_encoder.shape)
        y2 = self.bn2(self.relu(self.deconv2(out_encoder)))
        print("y2---->", y2.shape)
        ym1= self.unpool1(y2,ind2) 
        print("ym1--->", ym1.shape)
        y3 = self.bn3(self.relu(self.deconv3(ym1)))
        print("y3---->", y3.shape)
        xd2 = self.drop(y3)
        print("xd2---->", xd2.shape)
        y4 = self.bn4(self.relu(self.deconv4(xd2)))
        print("y4---->", y4.shape)
        ym2= self.unpool1(y4,ind1)
        print("ym2---->", ym2.shape)
        y5 = self.bn5(self.relu(self.deconv5(ym2)))
        print("y5---->", y5.shape)
        out_decoder = self.bn6(self.relu(self.deconv6(y5)))
        print("out_decoder---->", out_decoder.shape)
        score = self.classifier(out_decoder)                   
        print("score---->", score.shape)
        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):

        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        out_encoder = self.bnd5(self.relu(self.conv5(x4)))
        # Complete the forward function for the rest of the encoder

        y1 = self.bn1(self.relu(self.deconv1(out_encoder)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))
#         out_decoder = self.bn5(self.relu(self.deconv5(y5)))
        # Complete the forward function for the rest of the decoder

        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN_2(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.lkyrlu  = nn.LeakyReLU(0.1)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):

        x1 = self.bnd1(self.lkyrlu(self.conv1(x)))
        x2 = self.bnd2(self.lkyrlu(self.conv2(x1)))
        x3 = self.bnd3(self.lkyrlu(self.conv3(x2)))
        x4 = self.bnd4(self.lkyrlu(self.conv4(x3)))
        out_encoder = self.bnd5(self.lkyrlu(self.conv5(x4)))
        # Complete the forward function for the rest of the encoder

        y1 = self.bn1(self.lkyrlu(self.deconv1(out_encoder)))
        y2 = self.bn2(self.lkyrlu(self.deconv2(y1)))
        y3 = self.bn3(self.lkyrlu(self.deconv3(y2)))
        y4 = self.bn4(self.lkyrlu(self.deconv4(y3)))
        out_decoder = self.bn5(self.lkyrlu(self.deconv5(y4)))
#         out_decoder = self.bn5(self.relu(self.deconv5(y5)))
        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

    

class FCN_TL(nn.Module):
    
    
    def __init__(self, pretrained, n_class):
        super().__init__()
        self.pretrained = pretrained
#         num_ftrs =  self.pretrained.fc.in_features

#         self.fc = nn.Linear(num_ftrs, 512)

        self.n_class = n_class
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)


    def forward(self, x):
       
        y1 = self.bn1(self.relu(self.deconv1(self.pretrained(x))))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))
#         out_decoder = self.bn5(self.relu(self.deconv5(y5)))
        # Complete the forward function for the rest of the decoder
        
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

 
