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
import numpy

device = torch.device('cuda')  # determine which device to use (gpu or cpu)
use_gpu = torch.cuda.is_available()
print("gpu availability ----------------->", use_gpu)

train_dataset = TASDataset('../tas500v1.1', augment=False)
val_dataset = TASDataset('../tas500v1.1', eval=True, mode='val', augment=False)
test_dataset = TASDataset('../tas500v1.1', eval=True, mode='test', augment=False)

batchsize = 1

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)

if __name__ == "__main__":

    criterion = nn.CrossEntropyLoss()  # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    n_class = 10
    counts = np.zeros(10)
    temp = np.zeros(10)
    counter = np.zeros(10)
#     for iter, (image, mask) in enumerate(train_loader) :
#         temp_mask = mask[0].reshape(1,-1)[0].numpy()
#         for i in range(10):
#             temp[i] = np.sum(temp_mask == i)
#         counts = counts + temp
# #         numpy.concatenate(counter,temp)
#     print(counts)

    c = [2.2829796e+07, 7.8633099e+07, 5.2830670e+06, 1.1524840e+06, 1.3038040e+07, 8.8161500e+05, 1.8582400e+05, 2.0445000e+04, 6.1251140e+06, 1.3723600e+05]
    weights = [sum(c)/ct for ct in c ]
    print(weights)
    ####
    weight = [5.619267031558232, 1.6314595460621488, 24.282622196538487, 111.31323298197633, 9.839417581170176, 145.51331363463643, 690.3667986912347, 6274.723404255319, 20.944380790300393, 934.7891223877117]
    counts = numpy.log(c)
    plt.bar(range(10),counts)
    plt.xlabel("Class Label")
    plt.ylabel("Total Counts in log scale")
    plt.title("Counts in log scale of Different Classes")   
    plt.show()
    plt.savefig('Images/data.jpg')

    # plt.rcParams.update({'font.size': 30})
    # plt.figure(1)
    # fig = plt.figure(figsize=(15, 15))
    # plt.plot(train_metrics['epochs'], train_metrics['train_loss'], label='train')
    # plt.plot(train_metrics['epochs'], train_metrics['valid_loss'], label='validation')
    # plt.xlabel('NUM OF EPOCHS')
    # plt.ylabel('LOSS')
    # plt.title("Loss Vs num of Epochs")
    # plt.legend()
    # plt.show()
    # fig.savefig('Images/Loss_plot ' + str(round(time.time() % 10000)) + ".jpg", bbox_inches='tight', dpi=150)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()