from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torchdata.datapipes as dp
import glob
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  DataLoader, Dataset, SubsetRandomSampler
import math

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,478,1)
        self.fc1 = nn.Linear(478,367)
        self.fc2 = nn.Linear(367,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(478)
        self.bn4 = nn.BatchNorm1d(367)
        self.bn5 = nn.BatchNorm1d(256)
       
    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        # add identity to the output
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix
    
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,478,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(478)
        
    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64
    
class PointNet(nn.Module):
    def __init__(self, classes=8):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(478, 367)
        self.fc2 = nn.Linear(367, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(367)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        output = self.logsoftmax(output)
        #print(output, output.shape)
        return output, matrix3x3, matrix64x64

class ExpressionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        return data, label
    
def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.CrossEntropyLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)

def train(pointnet, optimizer, train_loader, device, batch_size, epoch, test_loader=None):
    pointnet.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
        loss = pointnetloss(outputs, labels, m3x3, m64x64)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                running_loss = 0.0

    pointnet.eval()

    correct = total = 0

    # validation
    if test_loader:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.reshape(batch_size, 478, 3)
                labels = labels.reshape(batch_size, 1, 8)
                labels = (labels == 1).nonzero(as_tuple=False)
                labels = labels[:,-1].clone()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = 100. * correct / total
        print('Valid accuracy: %d %%' % test_acc)

def get_data():
    #directory
    directory = os.getcwd()

    #folder path and extension
    data_path = os.path.join(directory, r'ExpressionDetection\NormData\AggregatedSpatialData.npy')
    label_path = os.path.join(directory, r'ExpressionDetection\NormData\AggregatedLabelData.npy')
    
    data=np.load(data_path)
    labels=np.load(label_path)

    data = data.reshape(len(labels), 478, 3)
    
    return data, labels

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    data, data_labels = get_data()   

    dataset = ExpressionDataset(data, data_labels)

    test_split = .2
    batch_size = 32
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    indices = [*range(len(data.data))]
    split = int(np.floor(test_split * (len(data.data))))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size, drop_last=True)

    pointnet = PointNet().to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.002)

    for epoch in range(1, args.epochs + 1):
        train(pointnet, optimizer, train_loader, device, batch_size, epoch, test_loader=test_loader)

    if args.save_model:
        torch.save(pointnet.state_dict(), 
                   os.path.join(os.getcwd(),
                                'ExpressionDetection\Models\pointnet{}.pt'.format(1+len(os.listdir(os.path.join(os.getcwd(),'ExpressionDetection\Models'))))))

if __name__ == '__main__':
    main()
