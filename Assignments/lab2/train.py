import torch
import torch.nn as nn
import torch.utils.data as data
from dataloader import VOC
from model import SegNet
from torch.autograd import Variable

def compute_iou(pred, gt):
    '''
        Your Code here
    '''
    return 0

def train(epoch, dataloader, model, criterion, optimizer, image_set = 'train'):
    loss_meter = 0
    # iou_meter = 0

    # We need to switch some functions off while evaluating, eg. dropout
    if image_set == 'valid' or image_set == 'test':
        model = model.eval()

    for i, (input, target) in enumerate(dataloader):
        input = Variable(input.float()).cuda()
        target = Variable(target).float().cuda()

        output = model(input)

        # Introducing the loss here
        loss = criterion(output, target)
        loss_meter += loss

        # Compute iou here
        # iou = compute_iou(output, target)
        # iou_meter += iou

        if image_set == 'train':
            optimizer.zero_grad() # Why did we do this?
            loss.backward()
            optimizer.step()

        print('loss at epoch ', str(epoch), ' iteration ', str(i), ' is: ', loss.data.cpu().numpy())


if __name__ == "__main__":
    train_dataset = VOC('./', 'train')
    test_dataset = VOC('./', 'val')
    train_dataloader = data.DataLoader(
                        train_dataset,
                        batch_size = 8,
                        shuffle = False,
                        num_workers = 4)

    model = SegNet()
    criterion = nn.MSELoss()

    # Comment if not using a GPU
    model = model.cuda()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum = 0.9)
    n_epochs = 10
    for i in range(n_epochs):
        train(i, train_dataloader, model, criterion, optimizer, 'train')


