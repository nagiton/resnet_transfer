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

plt.ion()   # interactive mode


data_transforms = {
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

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer,device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    iter = 0

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            iter +=1

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #    phase, epoch_loss, epoch_acc))
        #writer.add_scalar("loss", epoch_loss, iter)
        #writer.add_scalar("acc", epoch_acc, iter)

        #print()

    return model, (1-epoch_acc)


def get_optimizer(trial, model_ft):
    lr = trial.suggest_loguniform('learning_rate',1e-7,1e-3)
    final_lr = trial.suggest_loguniform('final_lr',1e-5,1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay',1e-6,1e-2)
    optimizer_ft = adabound.AdaBound(model_ft.parameters(), lr=lr, \
                     final_lr=final_lr, betas=(0.9,0.999), gamma=0.001, weight_decay=weight_decay)
    return optimizer_ft

# Observe that all parameters are being optimized
#optimizer_ft = adabound.AdaBound(model_ft.parameters(), lr=1e-5, final_lr=0.001, betas=(0.9,0.999), gamma=0.001, weight_decay=5e-4)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #tbx
    #comment = 'resnet50+adabound'
    #writer = tbx.SummaryWriter("runs/resnet50_adabound_small_lr",comment = comment)

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = get_optimizer(trial,model_ft)

    EPOCH = 10
    for step in range(EPOCH):
        model_ft, error_rate = train_model(model_ft, criterion, optimizer_ft,device)
    return error_rate

study = optuna.create_study()
study.optimize(objective, n_trials=100)
