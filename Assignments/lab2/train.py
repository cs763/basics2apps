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
    
    # gt_bin = gt.int()
    # intersection = (pred_bin & gt_bin).float().sum()
    # union = (pred_bin | gt_bin).float().sum()
    # iou = intersection / (union + 0.000001)
    return acc

def segments_to_masks(seg):
	seg_height = seg.shape[0]
	seg_width = seg.shape[1]
	masks = np.zeros((21, seg_height, seg_width))
	for i in range(seg_height):
		for j in range(seg_width):
			pixel_class = seg[i,j]

			# Since white lines are just for the sake of boundary, we will
			# predict them
			if pixel_class != 255:
				masks[pixel_class, i, j] = 1
	return masks

def plot_seg(image, seg):
    seg = seg[0].argmax(-3).squeeze()
    image = image[0].squeeze()
    print(seg.shape)
    masks = segments_to_masks(seg)
    plt.figure
    plt.imshow(image)
    for i in range(masks.shape[0]):
        plt.imshow(masks[i], alpha=0.1)
    plt.show()


def train(epoch, dataloader, model, criterion, optimizer, image_set = 'train'):
    loss_meter = 0
    acc_meter = 0

    for i, (input, target) in enumerate(dataloader):
        if image_set == 'train':
            input = input.requires_grad_(True).float().cuda()
        else:
            input = input.float().cuda()
        target = target.float().cuda()

        output = model(input)

        # Introducing the loss here
        loss = criterion(output, target)
        loss_meter += loss.item()

        # Compute acc here
        acc = compute_acc(output, target)
        acc_meter += acc.item()

        # Visualizing the outputs
        # if epoch > 0 and image_set == 'val':
        plot_seg(input.detach().cpu().numpy(), output.detach().cpu().numpy())
            

        if image_set == 'train':
            # routine lines from last lab
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

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
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    n_epochs = 10
    for i in range(n_epochs):
        train(i, train_dataloader, model, criterion, optimizer, 'train')
        if i % 2 == 0:
            train(i, val_dataloader, model, criterion, optimizer, 'val')


