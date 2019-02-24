# dirname is hardcoded : '/mnt/d1/data/norb'
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import create_fcn as create_fcn
# import torchfile

###########################################################################
#             This is where we preprocess the data                        #
###########################################################################

train_data = np.load('norb/train.npy')
train_labels = np.load('norb/train_cat.npy')

test_data = np.load('norb/test.npy')
test_labels = np.load('norb/test_cat.npy')

# Let's print the shape of the training and test data:
# print(train_data.shape, train_labels.shape)
# print(test_data.shape, test_labels.shape)

D_in = 108*108
D_out = 6
gpu = 1

# We'll only take a small subset of the test data for quick evaluation
test_data = test_data[:1000]
test_labels = test_labels[:1000]

# The network expects an input of batch_size x (height * width)
# n_channels in our case is 1. For RGB images, it is 3.
train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

# Preprocess your images if you want
# train_data = preprocess_data()

# create_fcn function is written in model.py.
model = create_fcn(D_in, D_out)
# Initialise a loss function.
# eg. if we wanted an MSE Loss: loss_fn = nn.MSELoss()
# Please search the PyTorch doc for cross-entropy loss function
loss_fn =

# Activate gradients for the input data
# x = torch.from_numpy(x)
# x = x.requires_grad_(True)
train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()

train_data =
test_data =

# If we're planning to use cross entropy loss, the data type of the
# targets needs to be 'long'.
train_labels = torch.from_numpy(train_labels).long()
test_labels = torch.from_numpy(test_labels).long()

# Convert the data and labels into cuda Variables now: x = x.cuda()
train_data =
test_data =

train_labels =
test_labels =

# Converting the entire data into cuda variable is NOT a good practice.
# We're still able to do it here because our data is small and can fit in
# the gpu memory. When working with larger datasets (will see tomorrow) and,
# bigger networks, it is advisable to convert the minibatches into cuda just
# before they're fed to the network.


###########################################################################
#             This is where we make the network learn                     #
###########################################################################

# Converting the model and loss function into a cuda variables just like we
# did with the data variables
model =
loss_fn =

# Hyper-parameters for training. Please play with these values to find the
# optimal training hyperparameters
learning_rate = 0.0001
batch_size = 324
n_batch = train_data.shape[0] // batch_size
accuracy = 0.0
n_epoch = 10

# Initializing the optimizer with hyperparameters.
# Please play with SGD, RMSProp, Adagrad, etc.
# Note that different optimizers may require differen hyperparameter values
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer =

for t in range(n_epoch):
    for m in range(n_batch):
        # Slicing the input batch
        inp = train_data[m*batch_size : (m+1)*batch_size]
        tar = train_labels[m*batch_size : (m+1)*batch_size]

        # Add random perturbations in this functions. Define
        # this function if you wish to use it.
        # inp = add_noise(inp)

        # Compute the network's output: Forward Prop the minibatch input
        # using the earlier defined model
        # pred = model(inp)
        pred =

        # Compute the network's loss. Use the earlier defined loss_fn.
        # loss = loss_fn(pred, tar)
        loss =

        # Zero the gradients of all the network's parameters
        # For this, checkout zero_grad() function of optim

        # Computer the network's gradients: Backward Prop
        loss.backward()

        # Update the network's parameters based on the computed
        # gradients
        optimizer.step()

        print(t, m, loss.item(), accuracy)

    # Validation after every 2nd epoch
    if t % 2 == 0:
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
