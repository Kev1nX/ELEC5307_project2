'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim

from network import Network # the network you used

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()
generator = torch.Generator().manual_seed(5307)





# basic training process used in last project
def train_net(net, trainloader, valloader,learningrate,nepoch):
########## ToDo: Your codes goes below #######
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningrate, momentum=0.9)
    # take sample for faster training, can also use samplesize = 1 to train with full dataset
    
    trainloss = []
    valloss = []
    epoch_time = []
    for epoch in range(nepoch):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 20 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 2000))
                trainloss.append(running_loss/100)
                running_loss = 0.0
    
        val_running_loss = 0.0

        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            # Forward Pass
            outputs = net(inputs)
            # Find the Loss
            loss = criterion(outputs,labels)
            # print statistics
            val_running_loss += loss.item()
            if i % 100 == 99:    # print every 200 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, val_running_loss / 200))
                valloss.append(val_running_loss/100)
                val_running_loss = 0.0
        epoch_time.append(time.time() - start)
    return trainloss,valloss,nepoch,epoch_time

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

train_image_path = '../train/' 
validation_image_path = '../validation/' 

trainset = ImageFolder(train_image_path, train_transform)
valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.

network = Network()
if args.cuda:
    network = network.cuda()

# train and eval your trained network
# you have to define your own 
val_acc = train_net(network, trainloader, valloader)

print("final validation accuracy:", val_acc)

# ==================================


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# functions to show the loss
def loss_curve(train_loss,val_loss,epoch,title):
    iter = range(1, len(train_loss) + 1)
    # Plot the loss curve
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(iter, train_loss)
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    plt.plot(iter, train_loss, marker='o', linestyle='-', color='b')
    plt.plot(iter, intercept + slope * iter, color="red", label="Line of Best Fit")
    plt.annotate(f"Slope : {round(slope, 3)}, R : {round(r_value,3)}", (round(len(train_loss)/2), min(train_loss)), color='r',
    fontsize=10, ha='center', va='bottom', backgroundcolor='w')
    plt.title(f'Training loss Curve for {title}')
    plt.xlabel('No. of 1000 batch iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    for i in iter:
        if i % (len(train_loss)/epoch) == 0:
            plt.axvline(x=i, color='r', linestyle='--', linewidth=1)
            plt.annotate(f'Epoch {int(i/(len(train_loss)/epoch))}', (i, max(train_loss)), color='r',
                     fontsize=10, ha='center', va='bottom', backgroundcolor='w')
            
    iter = range(1, len(val_loss) + 1)
    plt.subplot(122)
    plt.plot(iter, val_loss, marker='o', linestyle='-', color='b')
    plt.title(f'Validation loss Curve for {title}')
    plt.xlabel('No. of 100 batch iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    for i in iter:
        if i % (len(val_loss)/epoch) == 0:
            plt.axvline(x=i, color='r', linestyle='--', linewidth=1)
            plt.annotate(f'Epoch {int(i/(len(val_loss)/epoch))}', (i, max(val_loss)), color='r',
                     fontsize=10, ha='center', va='bottom', backgroundcolor='w')
            
    plt.tight_layout()

    plt.show()