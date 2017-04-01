import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchnet.meter as meter
import time
import argparse
from model.CNN import CNN
from model.NIN import NIN
from model.ResNet import ResNet, ResidualBlock

############  Hyper Parameters   ############
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--network', type=str, default='CNN', metavar='N',
                    help='which model to use, CNN|NIN|ResNet')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#############  DATASET   ################
train_dataset = datasets.MNIST(root = '../data/',
                               train = True,
                               transform = transforms.ToTensor(),
                               download = True)

test_dataset = datasets.MNIST(root = '../data/',
                              train = False,
                              transform = transforms.ToTensor(),
                              download = True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = args.batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = args.batch_size,
                                           shuffle = False)

###############   Model   ##################
if(args.network == 'CNN'):
    cnn = CNN()
elif(args.network == 'NIN'):
    cnn = NIN()
elif(args.network == 'ResNet'):
    cnn = ResNet(ResidualBlock, [2, 2, 2, 2])
if not args.no_cuda:
    cnn.cuda()
print(cnn)

################   Loss   #################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=args.lr)
mtr = meter.ConfusionMeter(k=10)

################   Training   #############
def train(epoch):
    cnn.train()

    for i , (images,labels) in enumerate(train_loader):
        if not args.no_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)

        # forward
        optimizer.zero_grad()
        outputs = cnn(images)

        loss = criterion(outputs,labels)

        # backward
        loss.backward()
        optimizer.step()
        if (i+1) % args.log_interval == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   %(epoch+1, 5, i+1, len(train_dataset)//args.batch_size, loss.data[0]))

def test():
    cnn.eval()

    # training data test
    for images,labels in train_loader:
        if not args.no_cuda:
            images = images.cuda()
        images = Variable(images)

        # forward
        outputs = cnn(images)
        mtr.add(outputs.data, labels)

    trainacc = mtr.value().diagonal().sum()*1.0/len(train_dataset)
    mtr.reset()

    # testing data test
    for images,labels in test_loader:
        if not args.no_cuda:
            images = images.cuda()
        images = Variable(images)

        # forward
        outputs = cnn(images)
        mtr.add(outputs.data, labels)

    testacc = mtr.value().diagonal().sum()*1.0/len(test_dataset)
    mtr.reset()

    # logging
    print('Accuracy on training data is: %f . Accuracy on testing data is: %f. '%(trainacc, testacc) )

##################   Main   ##################
for epoch in range(args.epochs):
    train(epoch)
    test()
torch.save(cnn.state_dict(), 'cnn.pkl')
