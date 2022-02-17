import torch.nn as nn
from torchvision import models

   
class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2   = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv4   = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv5   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv6   = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv7   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv8   = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv9   = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv10   = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        
        self.relu    = nn.ReLU(inplace=True)
        self.maxP1 = nn.MaxPool2d(2, stride=2)
        
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv11   = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv12   = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv13   = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv14   = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv13   = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv14   = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv13   = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv14   = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1)
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
