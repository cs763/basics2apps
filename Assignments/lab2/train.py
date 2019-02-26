import torch
import torch.nn as nn
import torch.utils.data as data
from dataloader import VOC
from model import SegNet
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def compute_acc(pred, gt):
    '''
        Assuming pred and gt are both tensors of same size
    '''
    pred = pred.argmax(1).float()
    gt = gt.argmax(1).float()
    acc = pred.eq(gt).float().sum() / torch.numel(gt)
    return acc

def train(epoch, dataloader, model, criterion, optimizer, image_set = 'train'):
    loss_meter = 0
    acc_meter = 0

    for i, (input, target) in enumerate(dataloader):
        if image_set == 'train':
            input = input.requires_grad_(True).float().cuda()
        else:
            input = input.float().cuda()
        target = target.float().cuda()

        # Get the model output
        output = 

        # Introducing the loss here. Compute the loss value
        loss = 
        loss_meter += loss.item()

        # Compute acc here
        acc = compute_acc(output, target)
        acc_meter += acc.item()

        if image_set == 'train':
            # In the next three lines:
            # Zero the existing gadients
            # Do a backward pass
            # Update the weights

        if i % 3 == 0:
            print(image_set, ' loss at epoch ', str(epoch), ' iteration ', str(i), ' is: ', loss_meter / (i+1),
                       ' and acc is: ', acc_meter / (i+1))


if __name__ == "__main__":
    train_dataset = VOC('./VOCdevkit/', 'train')
    val_dataset = VOC('./VOCdevkit/', 'val')
    train_dataloader = data.DataLoader(
                        train_dataset,
                        batch_size = 6,
                        shuffle = True,
                        num_workers = 4)

    val_dataloader = data.DataLoader(
                        val_dataset,
                        batch_size = 1,
                        shuffle = False,
                        num_workers = 1)

    model = SegNet()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    # Comment if not using a GPU
    model = model.cuda()
    criterion = criterion.cuda()

    # Inititialize the optimizer.
    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr)
    n_epochs = 10
    for i in range(n_epochs):
        train(i, train_dataloader, model, criterion, optimizer, 'train')
        if i % 2 == 0:
            train(i, val_dataloader, model, criterion, optimizer, 'val')


