import torch.nn as nn
import torch
from torch.autograd import Variable
import pdb

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # A Linear layer of size (input_size + hidden_size, hidden_size)
        self.i2h = 
        # A Linear layer of size (input_size + hidden_size, output) 
        self.i2o = 

    def forward(self, input, hidden):
        # Concatenates input and hidden vectors
        combined = torch.cat((input, hidden), 1)
        # Connect the two layers in the RNN fashion, i.e., combined to i2h,
        # and combined to i2o
        hidden = 
        output = 
        # Return the output at each time step along with the hidden vector
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


# if __name__ == "__main__":
#     rnn = RNN(1, 128, 1)
#     hidden = rnn.init_hidden(6)
#     rnn.zero_grad()
#     inp = torch.rand(6, 48)
#     for i in range(48):
#         output, hidden = rnn(Variable(inp[:,i].view(-1,1)), hidden)

#     print(output)

