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

from network import GoogLeNet # the network you used
from sklearn.model_selection import StratifiedKFold,cross_val_score
from skorch import NeuralNetClassifier

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()
generator = torch.Generator().manual_seed(5307)

# def create_logger(final_output_path):
#     log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
#     head = '%(asctime)-15s %(message)s'
#     logging.basicConfig(filename=os.path.join(final_output_path, log_file),
#                         format=head)
#     clogger = logging.getLogger()
#     clogger.setLevel(logging.INFO)
#     # add handler
#     # print to stdout and log file
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
#     clogger.addHandler(ch)
#     return clogger

# # basic training process used in last project
def train_net(net, trainloader, valloader,learningrate,nepoch):
########## ToDo: Your codes goes below #######
    val_accuracy = 0
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learningrate)
    # modified section
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.9295372635802442)
    ###
    trainloss = []
    valloss = []
    epoch_time = []
    net = net.train()
    for epoch in range(nepoch):  # loop over the dataset multiple times
        start = time.time()
        
        # modified section
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()
        #####
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if args.cuda:
                loss = loss.cpu()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 2000))
                trainloss.append(running_loss/20)
                running_loss = 0.0
    
        val_running_loss = 0.0

        for i, data in enumerate(valloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # Forward Pass
            outputs = net(inputs)
            outputs = outputs.logits
            # Find the Loss
            loss = criterion(outputs,labels)
            if args.cuda:
                loss = loss.cuda()
            # print statistics
            val_running_loss += loss.item()
            if i % 1 == 0:    # print every 10 batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, val_running_loss / 200))
                valloss.append(val_running_loss/1)
                val_running_loss = 0.0
        epoch_time.append(time.time() - start)
        print(f"epoch{epoch} : {time.time()-start}")
    return trainloss,valloss,nepoch,epoch_time

##############################################

# Analysis tool

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
    plt.xlabel('No. of 100 batch iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    count = 0
    for i in iter:
        if i % (len(train_loss)/epoch) == 0:
            count+=1
            if count == 2:
                plt.axvline(x=i, color='r', linestyle='--', linewidth=1)
                plt.annotate(f'E{int(i/(len(train_loss)/epoch))}', (i, max(train_loss)), color='r',
                        fontsize=10, ha='center', va='bottom', backgroundcolor='w')
                count = 0
            
    iter = range(1, len(val_loss) + 1)
    plt.subplot(122)
    plt.plot(iter, val_loss, marker='o', linestyle='-', color='b')
    plt.title(f'Validation loss Curve for {title}')
    plt.xlabel('No. of 10 batch iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    for i in iter:
        if i % (len(val_loss)/epoch) == 0:
            plt.axvline(x=i, color='r', linestyle='--', linewidth=1)
            plt.annotate(f'E{int(i/(len(val_loss)/epoch))}', (i, max(val_loss)), color='r',
                     fontsize=10, ha='center', va='bottom', backgroundcolor='w')
            
    plt.tight_layout()

    plt.show()

def eval_net(net, loader, logging, mode="baseline"):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    # if args.pretrained:
    #     if args.cuda:
    #         net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cuda'))
    #     else:
    #         net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cuda()
            labels = labels.cuda()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    print('=' * 20)
    print('SUMMARY of '+ mode)
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print('=' * 520)
    # logging.info('=' * 55)
    # logging.info('SUMMARY of '+ mode)
    # logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))
    # logging.info('=' * 55)
    torch.save(net.state_dict(),f"{100*correct/total}.pth")


############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.

train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.564108, 0.50346, 0.427237), (0.20597, 0.206595, 0.21542))
])

####################################

####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.
from torchvision.datasets import ImageFolder
# train_transform = transforms.Compose([
#         transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#         transforms.ToTensor(),
# ])


TrainSet = ImageFolder('../2023_ELEC5307_P2Train/train', transform=train_transform)
ValSet = ImageFolder('../2023_ELEC5307_P2Train/test', transform=train_transform)
# train_image_path = '../2023_ELEC5307_P2Train/train'
# validation_image_path = '../2023_ELEC5307_P2Train/test'

# trainset = ImageFolder(train_image_path, train_transform)
# valset = ImageFolder(validation_image_path, train_transform)

trainloader = torch.utils.data.DataLoader(TrainSet, batch_size=32,
                                         shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(ValSet, batch_size=32,
                                         shuffle=True, num_workers=2)
####################################

# ==================================
# use cuda if called with '--cuda'.




# train and eval your trained network
# you have to define your own 
# val_acc = train_net(network, trainloader, valloader)
# logger = create_logger("")

# print("final validation accuracy:", val_acc)


# ==================================

if __name__ == '__main__':     # this is used for running in Windows
    # train modified network
    network = GoogLeNet()
    if args.cuda:
        network = network.cuda()
    trainloss,valloss,nepoch,epoch_time = train_net(network, trainloader, valloader,0.00014464051511481836,40)
    print(trainloss)
    eval_net(network,valloader,"base")
    loss_curve(trainloss,valloss,nepoch,"GoogLeNet")


