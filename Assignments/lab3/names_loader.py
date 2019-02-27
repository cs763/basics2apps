from __future__ import unicode_literals, print_function, division
import os.path
import torch
import random
# import numpy as np
import torch.utils.data as data
import unicodedata
import string
import glob

class NameData(data.Dataset):
    def __init__(self, root, split):
        '''
            root: The root directory of the dataset
            split: 'train'|'val'|'test'

        '''
        self.root = root
        self.split = split

        # Build the category_lines dictionary, a list of names per language
        self.category_lines = {}
        self.all_categories = []

        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)


        for filename in self.findFiles(root + '/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)
        self.inputs = []
        self.labels = []
        for cat  in self.category_lines:
            values = self.category_lines[cat]
            for j in range(len(values)):
                self.labels.append(cat)
                self.inputs.append(values[j])

        # Shuffling the list
        combined = list(zip(self.labels, self.inputs))
        random.shuffle(combined)
        self.labels, self.inputs = zip(*combined)

        if split == 'train':
            self.inputs = self.inputs[:15000]
            self.labels = self.labels[:15000]
        else:
            self.inputs = self.inputs[15000:]
            self.labels = self.labels[15000:]

        self.max_length = 18


    def findFiles(self, path): return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor
    
    def nameToTensor(self, name):
        tensor = self.lineToTensor(name)
        input = torch.zeros((self.max_length, 1, self.n_letters))

        if tensor.size(0) < self.max_length:
            n_zeros = self.max_length - tensor.size(0)
            input[n_zeros:,:,:] = tensor
        
        return input.squeeze()
    

    def __getitem__(self, index):
        category = self.labels[index]
        name = self.inputs[index]

        # Fetching the tensors representing the category and the input name
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(name)
        input = torch.zeros((self.max_length, 1, self.n_letters))

        if line_tensor.size(0) < self.max_length:
            n_zeros = self.max_length - line_tensor.size(0)
            input[n_zeros:,:,:] = line_tensor

        return input.squeeze(), category_tensor

    def __len__(self):
        return len(self.inputs)


# if __name__ == "__main__":
#    dataset = NameData('./data', 'train')
#    name = 'krizhevsky'
#    print(dataset.nameToTensor(name).shape)
#    dataloader = data.DataLoader(
#             dataset, batch_size = 2, shuffle = False)

    # for i, (f, t) in enumerate(dataloader):
