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

import optuna
from optuna.trial import TrialState
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


def create_logger(final_output_path):
    log_file = '{}.log'.format(final_output_path)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger


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
BATCHSIZE = 32
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
####################################
# Define the training dataset and dataloader.
# You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.
from torchvision.datasets import ImageFolder
def get_dataloader():
    # Load FashionMNIST dataset.
    TrainSet = ImageFolder('../2023_ELEC5307_P2Train/train', transform=train_transform)
    ValSet = ImageFolder('../2023_ELEC5307_P2Train/test', transform=train_transform)

    trainloader = torch.utils.data.DataLoader(TrainSet, batch_size=BATCHSIZE,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(ValSet, batch_size=BATCHSIZE,
                                            shuffle=True, num_workers=2)

    return trainloader, valloader

####################################
log = create_logger(f"optuna_trials")
# # basic training process used in last project
def train_net(trial):
########## ToDo: Your codes goes below #######
    net = GoogLeNet(num_classes=25)
    if args.cuda:
        net = net.cuda()
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.
    criterion = nn.CrossEntropyLoss()
    # hyperparameters that is being trialed and optimised
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    scheduler_name = "StepLR"
    # step_size = trial.suggest_int("stepsize",1,25,log=True)
    step_size = 1
    gamma = trial.suggest_float("gamma",0.9,1,log=True)
    learningrate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)    
    nepoch = trial.suggest_int("epoch",50,75,log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=learningrate)
    scheduler = getattr(optim.lr_scheduler,scheduler_name)(optimizer, step_size=step_size, gamma=gamma)
    # take sample for faster training, can also use samplesize = 1 to train with full dataset
    trainloader, valloader = get_dataloader()
    trainloss = []
    valloss = []
    epoch_time = []
    
    for epoch in range(nepoch):  # loop over the dataset multiple times
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()
        start = time.time()
        running_loss = 0.0
        net = net.train()
        for i, data in enumerate(trainloader, 0):
            
            if i * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break
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
                loss = loss.cuda()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 20 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 2000))
                trainloss.append(running_loss/100)
                running_loss = 0.0
                

        
        # Validation of the model.
        net.eval()
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                if i * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                # get the inputs
                inputs, labels = data
                if args.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                # Get the index of the max log-probability.
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / min(len(valloader.dataset), N_VALID_EXAMPLES)
        trial.report(accuracy, epoch)
        epoch_time.append(time.time() - start)
        print(f"epoch{epoch} : {time.time()-start}")
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    return accuracy

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
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    print('=' * 20)
    print('SUMMARY of '+ mode)
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print('=' * 520)
    torch.save(net.state_dict(),f"{100*correct/total}.pth")






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
    study = optuna.create_study(direction="maximize")
    study.optimize(train_net, n_trials=25, timeout=10000)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    for completed in complete_trials:
        log.info(f"Trial {completed.number} finished with value : {completed.value} and parameters : {completed.params}")
    trial = study.best_trial
    value = trial.value
    log.info("Best trial:")
    log.info(f"Value: {value}")
    log.info("  Params: ")
    for key, value in trial.params.items():
        log.info("    {}: {}".format(key, value))



