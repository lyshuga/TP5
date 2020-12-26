import torch
import torch.nn as nn
import torch.optim as optim
import torch
torch.cuda.empty_cache()
import math
import numpy as np
import pickle
from torchvision import transforms
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from PIL import Image
from os.path import isfile

def requires_grad(p):
    return p.requires_grad


class SimpleConvModel(nn.Module):
    
    def __init__(self, block, layers, num_classes=13):
        self.inplanes = 64
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class BasicCNN(BaseEstimator):
    def __init__(self, learning_rate=1e-3, nb_epoch = 6, batch_size = 48, verbose=False, use_cuda=False):
        super(BasicCNN, self).__init__()
        if learning_rate is None:
            learning_rate = 1e-3
        if nb_epoch is None:
            nb_epoch = 10
        if batch_size is None:
            batch_size = 32
        if verbose is None:
            verbose = False
        if use_cuda is None:
            use_cuda = False
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_cuda = use_cuda
        self.model_conv = SimpleConvModel(Bottleneck, [2,2,2,2])
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        if self.use_cuda:
            self.model_conv.cuda()
            self.criterion.cuda()
        # Optimizer
        self.optim = optim.Adagrad(self.model_conv.parameters(), lr=1e-3, weight_decay=0.05)

    def fit(self, X, Y):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 128
            param Y: numpy.ndarray
                shape = (num_sample, 1)
        '''
        X = self.process_data(X)
        Y = self.process_label(Y)
        self.model_conv.train()
        nb_batch = int(X.shape[0] / self.batch_size)
        for e in range(self.nb_epoch):
            sum_loss = 0
            for i in range(nb_batch):
                print(i, 'out of', nb_batch)
                self.optim.zero_grad()
                beg = i * self.batch_size
                end = min(X.shape[0], (i + 1) * self.batch_size)
                x = X[beg:end]
                y = Y[beg:end]
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()
#                     print(x.isnan().any(), y.isnan().any(), np.isnan(x.cpu().numpy()).any())
                out = self.model_conv(x)
                loss = self.criterion(out, y)
                del x
                del y
                loss.backward()
                self.optim.step()
#                 print(loss.item())
                sum_loss += loss.item()
            sum_loss /= nb_batch
            if self.verbose:
                print("Epoch %d : loss = %f" % (e, sum_loss))

    def process_data(self, X):
        n_sample = X.shape[0]
        mean = np.mean(X, axis=1)[:, np.newaxis]
        std = np.std(X, axis=1)[:, np.newaxis]
        X = (X - mean) / (std+1e-8)
        X = X.reshape(n_sample, 3, 128, 128)
        X = X.astype(np.float)# / 255.
        #print(X[0])
        isnan = np.isnan(X).any()
        if isnan:
            raise Exception()

        return torch.Tensor(X)

    def process_label(self, y):
        res = torch.zeros(1)
        for i in range(y.shape[0]):
            l = torch.Tensor([y[i,0]])
            res = torch.cat((res, l))
        return res[1:].type(torch.long)
    
    #def predict(self, X):
      #  self.model_conv.eval()
      #  X = self.process_data(X)
      #  if self.use_cuda:
      #      X = X.cuda()
       # pred = self.model_conv(X).argmax(dim=1).cpu().numpy()
       # return pred
    
    def predict(self, X):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 128
            return: numpy.ndarray
                of int with shape (num_sample) ?
                of float with shape (num_sample, num_class) ?
                of string with shape (num_sample) ?
        '''
        # inverted_dico = {v:k for k,v in self.label_dico.items()}
        self.model_conv.eval()
        X = self.process_data(X)

        nb_batch = int(X.shape[0] / self.batch_size)
        pred = []
        for i in range(nb_batch):
            beg = i * self.batch_size
            end = min(X.shape[0], (i + 1) * self.batch_size)
            x = X[beg:end]
            
            if self.use_cuda:
                x = x.cuda()
            preds = self.model_conv(x).argmax(dim=1).cpu().numpy()
            pred.append(preds)
            
        x = X[end:]
            
        if self.use_cuda:
            x = x.cuda()
        preds = self.model_conv(x).argmax(dim=1).cpu().numpy()
        pred.append(preds)
        pred = np.array([item for sublist in pred for item in sublist]).reshape((-1,1))
        return pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
