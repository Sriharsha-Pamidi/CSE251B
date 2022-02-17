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
	for iter, (image, mask) in enumerate(train_loader) :
		print(mask)
		break
	

	
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