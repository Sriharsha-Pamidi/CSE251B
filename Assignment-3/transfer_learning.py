from basic_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
from matplotlib import pyplot as plt
import time
import torch.nn as nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn.functional as F
from torch.autograd import Variable

   
class FCN2(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1    = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2    = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2,2, return_indices=True)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3    = nn.BatchNorm2d(128)
        self.conv4   = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4    = nn.BatchNorm2d(256)
        self.conv5   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5    = nn.BatchNorm2d(512)
        self.conv6   = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd6    = nn.BatchNorm2d(1024)
        self.relu    = nn.Tanh()
        self.deconv = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn     = nn.BatchNorm2d(1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.maxunpool2 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)


    def forward(self, x):
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        x5 = self.bnd5(self.relu(self.conv5(x4)))
        out_encoder = self.bnd6(self.relu(self.conv6(x5)))
        
        out_encoder, ind = self.maxpool1(out_encoder)
        out_encoder = self.maxunpool2(out_encoder, ind)
        
        y = self.bn(self.relu(self.deconv(out_encoder)))
        y1 = self.bn1(self.relu(self.deconv1(y)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        #y4 = self.maxunpool2(y4, ind)
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))
#       out_decoder = self.bn5(self.relu(self.deconv5(y5)))
        # Complete the forward function for the rest of the decoder
        #print(out_decoder.size())
        score = self.classifier(out_decoder)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)

    

# TODO: Some missing values are represented by '__'. You need to fill these up.

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   


def train():
    best_iou_score = 0.0
    train_metric = {'epochs': [], 'train_loss': [],'valid_loss': [],'Accuracy': [],'IOU_score': []}
    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()
            

            # # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the labels to the same device as the model's

            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            outputs = fcn_model(inputs) 
            #we will not need to transfer the output, it will be automatically in the same device as the model's!
            #labels = Variable(torch.FloatTensor(8).uniform_(0, 120).long())
            #labels = np.argmax(labels,axis=1)
            #print(labels.size())
            loss = criterion(outputs, labels.long())#calculate loss

            # backpropagate
            loss.backward()

            # update the weights
            optimizer.step()

            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        train_metric['epochs'].append(epoch + 1)
        train_metric['train_loss'].append(loss.item())

        val_loss , current_miou_score, current_accuracy = val(epoch)
        train_metric['valid_loss'].append(val_loss)
        train_metric['Accuracy'].append(current_accuracy)
        train_metric['IOU_score'].append(current_miou_score)
        print(f"\n\n")
    
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            best_accuracy = current_accuracy
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= 10:
            #save the best model
            torch.save(fcn_model, "Models/model_adam_0p0005.pth")
            break
            

    print("best IOU === ", best_iou_score)
    print("best accuracy ===",best_accuracy)
    return train_metric
    

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the labels to the same device as the model's
            
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = fcn_model(inputs)
        
            loss = criterion(outputs, labels.long())#calculate loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work
            outputs = outputs.data.cpu().numpy()
            N, _, h, w = outputs.shape
            pred = outputs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
            #valu,pred = torch.max(outputs) # Make sure to include an argmax to get the prediction from the outputs of your model
            
            labels = labels.data.cpu().numpy().reshape(N, h, w)
            mean_iou_scores.append(np.nanmean(iou(pred, labels, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, labels)) # Complete this function in the util
    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!   
    
    
    print(" IOU === ", np.mean(mean_iou_scores))
    print(" accuracy ===",np.mean(accuracy))
    
    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)

def test():
    # TODO: load the best model and complete the rest of the function for testing
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    temp = [(inputs,labels) for iter,(inputs, labels) in enumerate(val_loader)]
    mask_1 = temp[0][1][0]
    image_1 = temp[0][0][0]
    print(mask_1.size())
    plt.imshow(mask_1)
    plt.savefig('Images/Mask1 '+str(round(time.time()%10000))+".jpg")
    plt.imshow(image_1.permute(1, 2, 0)  )
    plt.savefig('Images/Image1 '+str(round(time.time()%10000))+".jpg")

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the labels to the same device as the model's

            outputs = fcn_model(inputs)
        
            loss = criterion(outputs, labels.long())#calculate loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work
            outputs = outputs.data.cpu().numpy()
            N, _, h, w = outputs.shape
            pred = outputs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
            #valu,pred = torch.max(outputs) # Make sure to include an argmax to get the prediction from the outputs of your model
            
            labels = labels.data.cpu().numpy().reshape(N, h, w)
            mean_iou_scores.append(np.nanmean(iou(pred, labels, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, labels)) # Complete this function in the util

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!    
    return np.mean(losses), np.mean(mean_iou_scores), np.mean(accuracy)




device = torch.device('cuda') # determine which device to use (gpu or cpu)
use_gpu = torch.cuda.is_available()
print("gpu availability ----------------->" , use_gpu)

train_dataset = TASDataset('tas500v1.1') 
val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')

batchsize = 2
train_loader = DataLoader(dataset=train_dataset, batch_size= batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batchsize, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= batchsize, shuffle=False)



if __name__ == "__main__":
    
    epochs = 100
    criterion = nn.CrossEntropyLoss() 
    # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
   
    n_class = 10
    
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, 3)
    
    #fcn_model = FCN2(n_class=n_class)
    #fcn_model = fcn_model.cuda()
    #fcn_model.apply(init_weights)
    fcn_model = model
    
    #optimizer = optim.AdamW(fcn_model.parameters(), lr=0.0001, )
    #optimizer = optim.SGD(fcn_model.parameters(), lr=0.005, momentum=0.9)  # choose an optimizer
    optimizer = optim.AdamW(fcn_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    
    fcn_model = fcn_model.to(device) #transfer the model to the device
    
    #val(0)  # show the accuracy before training
    train_metrics = train()
    _ ,test_miou_score, test_accuracy = test()
    print(f"Test Accuracy = ", test_accuracy)
    print(f"Test IOU score = ", test_miou_score)
    
    plt.rcParams.update({'font.size': 30})
    plt.figure(1)
    fig = plt.figure(figsize=(15,15))
    plt.plot(train_metrics['epochs'], train_metrics['train_loss'], label='train')
    plt.plot(train_metrics['epochs'], train_metrics['valid_loss'], label='validation')
    plt.xlabel('NUM OF EPOCHS')
    plt.ylabel('LOSS')
    plt.title("Loss Vs num of Epochs")
    plt.legend()
    plt.show()
    fig.savefig('Images/Loss_plot '+str(round(time.time()%10000))+".jpg", bbox_inches='tight', dpi=150)
    
 
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()