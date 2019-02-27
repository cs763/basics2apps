import torch
import torch.nn as nn
import torch.utils.data as data
from names_loader import NameData
from model import RNN
from torch.autograd import Variable

def accuracy(pred, gt):
    pred = pred.argmax(1)
    correct, count = 0, 0
    for i in range(pred.size(0)):
        if pred[i] == gt[i]:
            correct += 1
        count += 1
    accuracy = correct / count
    return accuracy


def train(epoch, dataloader, model, criterion, optimizer, categories, split = 'train'):
    loss_meter, acc_meter, count = 0, 0, 0

    if split == 'valid' or split == 'test':
        model = model.eval()

    for i, (input, target) in enumerate(dataloader):
        input = Variable(input.float()).cuda()
        target = Variable(target.reshape(-1,)).long().cuda()

        # Initializing the hidden state
        batch_size = input.size(0)
        hidden = Variable(model.init_hidden(batch_size)).cuda()

        # seq_len = input.size(1)
        model.zero_grad()
        for f in range(input.size(1)):
            output, hidden = model(input[:,f,:], hidden)

        loss = criterion(output, target)
        acc = accuracy(output, target)
        loss_meter += loss.data.cpu().numpy()
        acc_meter += acc

        if split == 'train':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        count += 1
        # print('loss at epoch ', str(epoch), ' iteration ', str(i), ' is: ', loss.data.cpu().numpy())
        if i % 100 == 0:
            print('epoch ', epoch, ' iteration ', i, ' loss is : ', loss_meter / count, ' accuracy is ', acc_meter / count)

    print('loss at epoch ', str(epoch), ' is: ', loss_meter / count)


if __name__ == "__main__":
    dataset = NameData('./data', 'train')
    dataset_val = NameData('./data', 'val')
    dataloader_train = data.DataLoader(
                 dataset, batch_size = 8, shuffle = True, num_workers = 4)

    dataloader_val = data.DataLoader(
                 dataset_val, batch_size = 1, shuffle = False, num_workers = 4)

    print(dataset.n_categories)
    categories = dataset.all_categories

    # Initialize the network. Hidden size: 1024.
    # 57 is the length of the one-hot-encoded input at each timestep
    model = RNN(57, 1024, dataset.n_categories)
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # comment if not using a gpu
    model = model.cuda()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.005)#$, momentum = 0.9)
    n_epochs = 10
    for i in range(n_epochs):
        train(i, dataloader_train, model, criterion, optimizer, categories, 'train')
        if i % 2 == 1:
            train(i, dataloader_val, model, criterion, optimizer, categories, 'val')





