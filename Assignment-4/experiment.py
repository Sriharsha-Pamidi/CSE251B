################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import warnings
from torch.nn.utils.rnn import pack_padded_sequence

warnings.filterwarnings("ignore")

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.config_data = config_data
        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader, self.__coco_train = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
#         self.__criterion = nn.NLLLoss()
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data["experiment"]["learning_rate"])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)
            

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0
        bleu1_list = []
        bleu4_list = []
        bleu1_value = 0
        bleu4_value = 0

        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        print("gpu availability ----------------->" , use_gpu)


        for j, (images, captions, lengths, img_ids) in enumerate(self.__train_loader):
#             reset optimizer gradients
            self.__optimizer.zero_grad()
         
            
       
            # # both inputs and labels have to reside in the same device as the model's
            images   = images.to(device) #transfer the input to the same device as the model's
            captions = captions.to(device) #transfer the labels to the same device as the model's
            self.__model.to(device)
            output_captions, output_captions_idx = self.__model(images,captions,train=True)
            
            packed_output_captions = pack_padded_sequence(output_captions, lengths,batch_first = True)
            packed_captions=pack_padded_sequence(captions, lengths,batch_first = True)
            

            loss = self.__criterion(packed_output_captions.data, packed_captions.data)#calculate loss
            loss.backward()
                            
            for i in range(output_captions_idx.shape[0]):
                pred_captions  = []
                label_captions = []
                for word in output_captions_idx[i]:
                    a = word.item()
                    word_value = self.__vocab.idx2word[a]
                    if (word_value != "<start>") and (word_value != "<end>" ) and (word != ".") and (word != ",") and (word != "!"):
                        pred_captions.append(word_value)
                

    
                count = 0
                while count < 5:
                    temp_sent = []                        
                    caption_generated = self.__coco_train.imgToAnns[img_ids[i]][count]["caption"]
                    for word in caption_generated.split():
                        if (word != ".") and (word != ",") and (word != "!"):
                            temp_sent.append(word)
                    label_captions.append(temp_sent)
                    count +=1

                
                bleu1_value = bleu1(label_captions, pred_captions)
                bleu4_value = bleu4(label_captions, pred_captions)

                bleu1_list.append(bleu1_value)
                bleu4_list.append(bleu4_value)

            if j%200 == 0 :            
                    print("Pred Captions----", pred_captions)
                    print("Label Captions ---",label_captions)
            if j%200 == 0 :            
                    print("trainLoss: {},Bleu1: {}, Bleu4: {}".format(loss.item(), np.mean(bleu1_list),np.mean(bleu4_list)))
                    print("\n")
          
            # update the weights
            self.__optimizer.step()
#             break
            #raise NotImplementedError()
        return loss.item()

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        self.__model.to(device)
        
        val_loss_list = []
   
        with torch.no_grad():
            for j, (images1, captions1, lengths, img_ids) in enumerate(self.__val_loader):
#                 print(captions1.shape)
                images1   = images1.to(device)
                captions1 = captions1.to(device)
                
                output_captions, output_captions_idx = self.__model(images1,captions1,train=False)     
                output_captions_for_loss, output_captions_idx_for_loss = self.__model(images1,captions1,train=True)
                
                packed_output_captions = pack_padded_sequence(output_captions_for_loss, lengths,batch_first = True)
                packed_captions=pack_padded_sequence(captions1, lengths,batch_first = True)

                val_loss       = self.__criterion(packed_output_captions.data, packed_captions.data) #calculate loss         
                val_loss_list.append(val_loss.item())
                

                result_str = "Val Performance: Loss: {}".format(np.mean(val_loss_list))
        self.__log(result_str)
            
        return np.mean(val_loss_list)
    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        device = torch.device('cuda') # determine which device to use (gpu or cpu)
        use_gpu = torch.cuda.is_available()
        self.__model.to(device)
        
        test_loss_list = []
        bleu1_list = []
        bleu4_list = []
        bleu1_value = 0
        bleu4_value = 0

        with torch.no_grad():
            for j, (images, captions, lengths, img_ids) in enumerate(self.__test_loader):
#                 print(captions.shape)

                images   = images.to(device)
                captions = captions.to(device)
                
                output_captions, output_captions_idx = self.__model(images,captions,train=False)     
                output_captions_for_loss, output_captions_idx_for_loss = self.__model(images,captions,train=True)
                
                packed_output_captions = pack_padded_sequence(output_captions_for_loss, lengths,batch_first = True)
                packed_captions=pack_padded_sequence(captions, lengths,batch_first = True)

                test_loss       = self.__criterion(packed_output_captions.data, packed_captions.data) #calculate loss   
                test_loss_list.append(test_loss.item())

                for i in range(output_captions_idx.shape[0]):
                    pred_captions  = []
                    label_captions = []
                    for word in output_captions_idx[i]:
                        a = word.item()
                        word_value = self.__vocab.idx2word[a]
                        if (word_value != "<start>") and (word_value != "<end>" ):
                            pred_captions.append(word_value)
#                     if j%500 == 0:
#                         print("Pred Captions----------------------",pred_captions)
                            
                    count = 0
                    while count < 5:
                        temp_sent = []                        
                        caption_generated = self.__coco_test.imgToAnns[img_ids[i]][count]["caption"]
                        for word in caption_generated.split():
                            temp_sent.append(word)
                        label_captions.append(temp_sent)
                        count +=1
#                     if j%500 == 0:
#                         print("Label Captions----------------------",label_captions)
                    bleu1_value = bleu1(label_captions, pred_captions)
                    bleu4_value = bleu4(label_captions, pred_captions)

                    bleu1_list.append(bleu1_value)
                    bleu4_list.append(bleu4_value)
#                 break
                if j%100 == 0 :            
                    print("TestLoss: {},Bleu1: {}, Bleu4: {}".format(np.mean(test_loss_list), np.mean(bleu1_list),np.mean(bleu4_list)))
                    print("\n")
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(np.mean(test_loss_list), np.mean(bleu1_list),np.mean(bleu4_list))
        self.__log(result_str)

        return np.mean(test_loss_list)
    
    
    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        training_losses = torch.tensor(self.__training_losses, device="cpu")
        val_losses = torch.tensor(self.__val_losses, device="cpu")
        plt.plot(x_axis, training_losses, label="Training Loss")
        plt.plot(x_axis, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
