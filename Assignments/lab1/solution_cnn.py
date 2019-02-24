# dirname is hardcoded : '/mnt/d1/data/norb'
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable
from model import Lenet as Lenet
# import torchfile

###########################################################################
#             This is where we preprocess the data                        #
###########################################################################

train_data = np.load('norb/train.npy')
train_labels = np.load('norb/train_cat.npy')

test_data = np.load('norb/test.npy')
test_labels = np.load('norb/test_cat.npy')

D_in = 108*108
D_out = 6
gpu = 1

test_data = test_data[:1000]
test_labels = test_labels[:1000]

# The network expects an input of batch_size x (height * width)
# n_channels in our case is 1. For RGB images, it is 3.

# Preprocess your images if you want
# train_data = preprocess_data()

# Lenet is written in model.py
model = Lenet()

# Converting inputs and labels into cuda (gpu) enabled torch 'Variables'.
train_data = torch.from_numpy(train_data).float().requires_grad_(True)
test_data = torch.from_numpy(test_data).float().requires_grad_(True)
train_labels = torch.from_numpy(train_labels).long()
test_labels = torch.from_numpy(test_labels).long()

train_data = train_data.cuda()
test_data = test_data.cuda()

train_labels = train_labels.cuda()
test_labels = test_labels.cuda()

# Converting the entire data into cuda variable is NOT a good practice.
# We're still able to do it here because our data is small and can fit in
# the gpu memory. When working with larger datasets (will see tomorrow) and,
# bigger networks, it is advisable to convert the minibatches into cuda just
# before they're fed to the network.


###########################################################################
#             This is where we make the network learn                     #
###########################################################################

# Converting the model into a cuda model and setting up the loss function
model = model.cuda()
loss_fn = nn.CrossEntropyLoss().cuda()

# Hyper-parameters for training
learning_rate = 0.0001
batch_size = 324
n_batch = train_data.shape[0] // batch_size
accuracy = 0

# Initializing the optimizer with hyperparameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(10):
    for m in range(n_batch):
        inp = train_data[m * batch_size: (m+1) * batch_size]
        tar = train_labels[m * batch_size: (m+1) * batch_size ]

        # Add random perturbations in this functions. Define
        # this function if you wish to use it.
        # inp = add_noise(inp)

        # Compute the network's output: Forward Prop
        pred = model(inp)

        # Compute the network's loss
        loss = loss_fn(pred, tar)

        # Zero the gradients of all the network's parameters
        optimizer.zero_grad()

        # Computer the network's gradients: Backward Prop
        loss.backward()

        # Update the network's parameters based on the computed
        # gradients
        # optimizer.step()
        for f in model.parameters():
            f.data -= 0.01 * f.grad # 0.01 is the learning rate

        print(t, m, loss.item(), accuracy)

    # Validation after every 2nd epoch
    if t % 1 == 0:
        # Forward pass
        output = model(test_data)

        # get the index of the max log-probability
        pred = output.data.max(1)[1]

        correct = pred.eq(test_labels).sum()
        accuracy = correct.item() / 1000
        print("\n*****************************************\n")
        print(accuracy)
        print("\n*****************************************\n")


# dt=torchfile.load('Used/Test/test.bin')
# dt = dt.reshape(dt.shape[0],dt.shape[1]*dt.shape[2])
# dtpy = Variable(torch.from_numpy(dt)).cuda().float()
# y_dtpy = model(dtpy)
# label=y_dtpy.data.max(1)[1]
# lnpy = label.cpu().numpy()
