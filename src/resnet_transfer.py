from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tensorboardX as tbx
import adabound
import optuna

class ResNet50Classifier(object):
    def __init__(self):
        #define network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = models.resnet50(pretrained=True)

        #reset fc layer
        self._num_ftrs = self.net.fc.in_features
        self._num_prediction = 2
        self.net.fc = nn.Linear(self._num_ftrs, self._num_prediction)

        #use gpu if possible
        self.net = self.net.to(self.device)

        #define loss function
        self.criterion = nn.CrossEntropyLoss()

        #define optimizer
        self.optimizer = optim.Optimizer(self.net.parameters(),{})

        #make tbx writer
        self.writer = tbx.SummaryWriter()

    def load_data(self):
        #define preprocessing
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        #set data
        self.data_dir = 'data/hymenoptera_data'
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  self.data_transforms[x])
                          for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

    def set_optimizer(self):
        self.optimizer = adabound.AdaBound(self.net.parameters(),\
         lr=1e-5, final_lr=0.001, betas=(0.9,0.999), gamma=0.001, weight_decay=5e-4)

    def single_propagation(self,inputs,labels,mode):
        #initialize
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        #forward
        with torch.set_grad_enabled(mode == 'train'):
            outputs = self.net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # backward + optimize only if in training phase
            if mode == 'train':
                loss.backward()
                self.optimizer.step()

        #report batch results
        batch_loss = loss.item() * inputs.size(0) #default: size_average=True
        batch_corrects = torch.sum(preds == labels.data)

        return batch_loss, batch_corrects

    def single_eval_process(self,inputs,labels):
        #initialize
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        #forward
        with torch.set_grad_enabled(False):
            outputs = self.net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

        #report batch results
        batch_loss = loss.item() * inputs.size(0) #default: size_average=True
        batch_corrects = torch.sum(preds == labels.data)

        return batch_loss, batch_corrects

    def train(self, epoch=20):


        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0

        #loop epoch times
        for step in range(epoch):
            since = time.time()
            train_epoch_loss = 0
            train_epoch_corrects = 0
            train_epoch_acc = 0
            eval_epoch_loss = 0
            eval_epoch_corrects = 0
            eval_epoch_acc = 0

            ##############training#########################
            for inputs, labels in self.dataloaders['train']:
                train_loss, train_corrects = self.single_propagation(inputs,\
                                                  labels,mode='train')
                train_epoch_loss += train_loss
                train_epoch_corrects += train_corrects
            train_epoch_acc = train_epoch_corrects.double()/self.dataset_sizes['train']
            train_epoch_loss = train_epoch_loss / self.dataset_sizes['train']

            #make report
            self.writer.add_scalar("train loss", train_epoch_loss, step)
            self.writer.add_scalar("train acc", train_epoch_acc, step)
            print('train Loss: {:.4f} Acc: {:.4f}'.format(
                 train_epoch_loss, train_epoch_acc))


            ###############evaluating#######################
            for inputs, labels in self.dataloaders['val']:
                eval_loss, eval_corrects = self.single_propagation(inputs,\
                                                  labels,mode='eval')
                eval_epoch_loss += eval_loss
                eval_epoch_corrects += eval_corrects
            eval_epoch_acc = eval_epoch_corrects.double()/self.dataset_sizes['val']
            eval_epoch_loss = eval_epoch_loss / self.dataset_sizes['val']

            #make report
            self.writer.add_scalar("val loss", eval_epoch_loss, step)
            self.writer.add_scalar("val acc", eval_epoch_acc, step)
            print('val Loss: {:.4f} Acc: {:.4f}'.format(
                 train_epoch_loss, train_epoch_acc))

            time_elapsed = time.time() - since

            print('{:3d}/{:3d}th epoch'.format(step,epoch))
            print('single epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            #save best parameters
            if  eval_epoch_acc > best_acc:
                best_acc = eval_epoch_acc
                best_model_wts = copy.deepcopy(self.net.state_dict())

        print('Best val Acc: {:4f}'.format(best_acc))

        #set training results
        self.net.load_state_dict(best_model_wts)
            #set best_param
            #set training acc and loss

        #return best_param, best_acc, min_loss, time_comsumed

    def save_model(self,path):
        torch.save(self.net.state_dict(), path)


def main():
    classifier = ResNet50Classifier()
    classifier.load_data()
    classifier.set_optimizer()
    classifier.train(epoch=50)
    classifier.save_model(path='model/resnet_transfer.pth')

if __name__=='__main__':
    main()
