

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as D
from torchvision.transforms import ToTensor, ToPILImage, Compose, CenterCrop, Normalize
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import timeit
#import pickle
from argparse import ArgumentParser

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2 
from erfnet import Net

from PIL import Image
import numpy as np
from carladataset import carla
import torch.optim as optim

from pathlib import Path
import json

use_crop = True

def train(train_loader, val_loader, optimizer, scheduler, criterion, net, args, device):
    start = timeit.default_timer()
    device = args.device

    max_Car_F = 0.
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, target, filename = data
            inputs, target = inputs.to(device), target.to(device) 
            if(use_crop):
                target = target[:,:,100:540,:]
                inputs = inputs[:,:,100:540,:]
            target = target[:,0,:,:]
            #target = torch.squeeze(target, 0)

            inputs, target = Variable(inputs), Variable(target)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            outputs = net(inputs)

            loss = criterion(outputs.float(),target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        scheduler.step(running_loss)

        Car_F = validate(val_loader, net, device)   
        if(Car_F > max_Car_F):
            # save best result
            max_Car_F = Car_F
            torch.save(net.state_dict(), args.savefile)         
            print("Saving - Car_F: ", max_Car_F)

    stop = timeit.default_timer()
    print("Runtime: ", stop - start)
    print('Finished Training')



def validate(val_loader, net, device):
    frames_processed = 0

    Car_TP = 0 # True Positives
    Car_FP = 0 # Flase Positives
    Car_TN = 0 # True Negatives
    Car_FN = 0 # True Negatives

    Road_TP = 0 # True Positives
    Road_FP = 0 # Flase Positives
    Road_TN = 0 # True Negatives
    Road_FN = 0 # True Negatives

    start = timeit.default_timer()

    net.eval()
    for step, (images, target, filenames) in enumerate(val_loader):

            img = Variable(images).to(device)
            #convert image back to Height,Width,Channels
            ##img = np.transpose(img, (1, 2, 0))
            target = target.cpu().numpy()[0][0]
            #convert image back to Height,Width,Channels
            target = np.transpose(target, (0, 1))

            result = net(img)     
            result = result.cpu().detach().numpy()[0]
            result = np.argmax(result, axis=0)

            
            truth_data_car =  (target == 2).astype(float)
            truth_data_road =  (target == 1).astype(float)
            student_data_car = (result == 2).astype(float) #2
            student_data_road = (result == 1).astype(float) #1

            Car_TP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 1))
            Car_FP += np.sum(np.logical_and(student_data_car == 1, truth_data_car == 0))
            Car_TN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 0))
            Car_FN += np.sum(np.logical_and(student_data_car == 0, truth_data_car == 1))

            Road_TP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 1))
            Road_FP += np.sum(np.logical_and(student_data_road == 1, truth_data_road == 0))
            Road_TN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 0))
            Road_FN += np.sum(np.logical_and(student_data_road == 0, truth_data_road == 1))

            
            frames_processed+=1

    stop = timeit.default_timer()

    Car_precision = Car_TP/(Car_TP+Car_FP)/1.0
    Car_recall = Car_TP/(Car_TP+Car_FN)/1.0
    Car_beta = 2
    Car_F = (1+Car_beta**2) * ((Car_precision*Car_recall)/(Car_beta**2 * Car_precision + Car_recall))
    Road_precision = Road_TP/(Road_TP+Road_FP)/1.0
    Road_recall = Road_TP/(Road_TP+Road_FN)/1.0
    Road_beta = 0.5
    Road_F = (1+Road_beta**2) * ((Road_precision*Road_recall)/(Road_beta**2 * Road_precision + Road_recall))

    print("Processed Frames: ", frames_processed)
    print("Runtime: ", stop - start)
    #print("FPS: ", frames_processed/ (stop-start))
    print("Car F score: %05.3f  | Car Precision: %05.3f  | Car Recall: %05.3f  |\n\
    Road F score: %05.3f | Road Precision: %05.3f | Road Recall: %05.3f | \n\
    Averaged F score: %05.3f" %(Car_F,Car_precision,Car_recall,Road_F,Road_precision,Road_recall,((Car_F+Road_F)/2.0)))        
    return Car_F

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    train_batch_size = args.batch_size

    input_transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
        ToTensor(),
        #Normalize([0.35676643, 0.33378336, 0.31191254], [0.24681774, 0.23830362, 0.2326341 ]),
    ])

    target_transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])


    datadir = args.datadir
    dataset = carla(datadir, input_transform=input_transform, target_transform=target_transform)
    val_dataset = carla(datadir, input_transform=ToTensor(), target_transform=None)

    dataset_len = len(dataset)
    dataset_idx = list(range(dataset_len))

    #split into training & validation set
    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    split = int(np.floor(train_ratio * dataset_len))
    train_idx = np.random.choice(dataset_idx, size=split, replace=False)
    val_idx = list(set(dataset_idx) - set(train_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(dataset, 
                    batch_size=train_batch_size, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                    batch_size=1, sampler=val_sampler)


    print('Total images = ', dataset_len)
    print('Number of images in train set = ', train_batch_size * len(train_loader))
    print('Number of images in validation set = ', len(val_loader))

    net = Net(num_classes=3)
    net = net.to(device)

    weights = [0.1, 0.5, 2.0]
    weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(net.parameters(), lr=args.lr) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    if(args.loadfile != None):
        net.load_state_dict(torch.load(args.loadfile, map_location={'cuda:1':'cuda:0'}))
        print("Loaded saved model: ", args.loadfile)

    train(train_loader, val_loader, optimizer, scheduler, criterion, net, args, device)
    #validate(val_loader, net, device)
    #torch.save(net.state_dict(), args.savefile)




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', action='store', default='cuda', help='choices={cpu, cuda, cuda:0, cuda:1, ...}')
    parser.add_argument('--epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savefile', default='./results', help='file to save the model')
    parser.add_argument('--loadfile', help='load an existing model to continue training')
    parser.add_argument('--datadir', default='data/new', help='location of dataset')
    parser.add_argument('--resume', type=bool, default=False, help='Use this flag to load last checkpoint for training')  #
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier')

    main(parser.parse_args())
