import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time, math

import torch
import torch.nn as nn
from torch.autograd import Variable

import string
import random
import codecs
import re
import argparse

# Hyperparameters
parser = argparse.ArgumentParser(description='PyTorch Char RNN')
parser.add_argument('--chunk_len', type=int, default=100, metavar='N',
                    help='split characters into chunks for training (default: 100)')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden_size', type=int, default=100, metavar='N',
                    help='hidden size of RNN (default:100)')
parser.add_argument('--n_layers', type=int, default=1, metavar='N',
                    help='layer number of RNN (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#############  DATASET   ################
# Dictionary
all_characters = string.printable
n_characters = len(all_characters)

# load training data
file = codecs.open("input.txt", "r", "utf-8").read()
print('Found %d chracters.'%(len(file)))

# turn a string into a tensor
def char_tensor(s):
    l = len(s)
    t = torch.LongTensor(l)
    for i in range(l):
        t[i] = all_characters.index(s[i])

    return Variable(t)

# generate a random chunk
def random_chunk(file,chunk_len):
    start_idx = random.randint(0,len(file) - chunk_len)
    end_idx = start_idx + chunk_len + 1
    return file[start_idx:end_idx]

def random_training_set(file, chunk_len):
    chunk = random_chunk(file, chunk_len)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

##############   Helper Functions   ##############
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

##############   RNN   ###################
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self,x,hidden):
        embedding = self.encoder(x.view(1,-1))
        output, hidden = self.rnn(embedding.view(1,1,-1),hidden)
        output = self.decoder(output.view(1,-1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.Tensor(n_layers, 1, hidden_size).fill_(0))

###############   training/evaluating   ###############
def evaluate(prime_str='A', predict_len=100, temperature=0.5):
    hidden = decoder.init_hidden()
    predicted = prime_str
    prime_tensor = char_tensor(prime_str)

    # feed prime_str to RNN
    for i in range(len(prime_str)-1):
        _,hidden = decoder(prime_tensor[i], hidden)

    # generate a string of length predict_len
    inp = prime_tensor[-1]
    for i in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # sample a multi_nominal Gaussian to get different character each time
        output_dist = output.data.view(-1).div(temperature).exp()
        idx = torch.multinomial(output_dist, 1)[0]
        character = all_characters[idx]
        predicted += character

        inp = char_tensor(character)

    return predicted

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    # forward
    for i in range(len(inp)):
        output, hidden = decoder(inp[i], hidden)
        loss += criterion(output, target[i])

    # backward
    loss.backward()
    decoder_optimizer.step()

    return loss.data[0]/len(inp)

###############   Main Script   #################
n_epochs = args.epochs
print_every = args.log_interval
plot_every = 10
hidden_size = args.hidden_size
n_layers = args.n_layers
lr = args.lr
chunk_len = args.chunk_len

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set(file, chunk_len))
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

# plot losses
plt.figure()
plt.plot(all_losses)
plt.show()

# evaluate
print(evaluate('Th', 200, temperature=0.5))
