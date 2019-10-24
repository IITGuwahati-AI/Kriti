from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, maxabs_scale, MaxAbsScaler
import time

class Encoder(nn.Module):
    def __init__(self, inputt, hidden):
        super(Encoder, self).__init__()
        self.inputt = inputt
        self.hidden = hidden
        self.fca = nn.Linear(inputt, inputt)
        self.fc1 = nn.Linear(inputt, hidden)

    def forward(self, x, thresh):
        x = x.view(-1, self.inputt)
        x_a = F.sigmoid(self.fca(x))
        x_a = (x_a > thresh).float()
        x = torch.mul(x, x_a)
        return F.relu(self.fc1(x))

class Decoder(nn.Module):
    def __init__(self, inputt, hidden):
        super(Decoder, self).__init__()
        self.inputt = inputt
        self.hidden = hidden
        self.fc1 = nn.Linear(inputt, hidden)

    def forward(self, x):
        return F.relu(self.fc1(x))

class AutoEncoder(nn.Module):
    def __init__(self, inputt=54, hidden=200):
        super(AutoEncoder, self).__init__()
        self.inputt = inputt
        self.hidden = hidden
        self.fc1 = Encoder(inputt, hidden)
        self.fc2 = Decoder(hidden, inputt)

    def forward(self, x, thresh):
        return self.fc2(self.fc1(x, thresh))

class Classifier(nn.Module):
    def __init__(self, inputt=200,out=7):
        super(Classifier, self).__init__()
        self.inputt = inputt
        self.out = out
        self.fc1 = nn.Linear(inputt, out)

    def forward(self, x):
        return F.softmax(self.fc1(x))

def kl_divergence(p, q):

    p = F.softmax(p)
    q = F.softmax(q)

    return torch.sum(p * torch.log(p / q)) + torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))

def train(model, model2, device, train_loader, optimizer, optimizer2, epoch, log_interval, sparsity_param, thresh, inp_size, batch_size):
    model.train()
    eff_number_of_sensors = []
    train_loss = 0
    correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):  #Itererate over the training data in batches

        data, label = data.to(device), label.to(device) # copy train data to either GPU or CPU
        optimizer.zero_grad() # Set grad to zero
        optimizer2.zero_grad() # Set grad to zero
        output = model(data, thresh)  # forward propagation
        target = data.view(-1, inp_size)
        attention = F.sigmoid(model.fc1.fca(data.view(-1, inp_size)))
        encoded = model.fc1(torch.mul(data.view(-1, inp_size),attention), thresh)
        ########## classify the encodings #######
        output2 = model2(encoded)  # forward propagation
        pred = output2.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss2 = F.nll_loss(output2, label) # Negative Log likelihood loss
        loss2.backward(retain_graph=True) # Error Backpropagation
        optimizer2.step() # update weights
        ###############################################
        rho_hat = torch.sum(encoded, dim=0, keepdim=True)/len(target)
        loss = F.mse_loss(output, target) + 5*kl_divergence(sparsity_param,  rho_hat.cpu()).to(device) + 0.001*torch.sum(attention.to(device))/(batch_size *inp_size)
        train_loss += F.mse_loss(output, target, reduction='sum').item()

        eff_number_of_sensors.append((torch.sum((attention>thresh).float())/(len(target))).cpu().detach().numpy())
        loss.backward() # Error Backpropagation
        optimizer.step() # update weights
        if batch_idx % log_interval == 0:  # for printing loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain accuracy:', acc)
    train_loss /= (len(train_loader)*inp_size)
    print('\nTrain set: Average loss:', train_loss)
    return train_loss, np.mean(eff_number_of_sensors), acc

def test(model, model2, device, test_loader, thresh, inp_sz):
    model.eval()
    test_loss = 0
    test_loss2 = 0
    correct = 0
    eff_number_of_sensors_test = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)  # copy test data to either GPU or CPU
            output = model(data, thresh)  # forward propagation
            target = data.view(-1, inp_sz)
            start = time.time()
            attention = F.sigmoid(model.fc1.fca(data.view(-1, inp_sz)))
            encoded = model.fc1(torch.mul(data.view(-1, inp_sz), attention), thresh)
            #############################################################
            ## Classify the encodings #################################
            output2 = model2(encoded)  # forward propagation
            end = time.time()
            diff = end - start
            test_loss2 += F.nll_loss(output2, label, reduction='sum').item()  # sum up batch loss
            pred = output2.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
            ############################################################
            eff_number_of_sensors_test.append(
                (torch.sum((attention > thresh).float()) / (len(target))).cpu().detach().numpy())
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
    #             print(test_loss)
    test_loss /= (len(test_loader) * inp_sz)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss:', test_loss)
    print('\nTest accuracy:', acc)

    return test_loss, np.mean(eff_number_of_sensors_test), acc

#####################################
def sample(x, n):
    return x.iloc[random.sample(x.index, n)]

def balance_data(df):
    for i in range(1, 8):
        print(i)
        #         print(df.index)
        cond = df.Cover_Type == i
        subset = df[cond].dropna()
        #         print(subset)
        if i == 4:
            subset = subset.sample(n=2747)
        else:
            subset = subset.sample(n=2747)
        if i == 1:
            balanced_data = subset
        else:
            balanced_data = balanced_data.append(subset, ignore_index=True)

    return balanced_data


def getData(filename, balance=False):
    chunksize = 1200000
    flag =1
    for data in pd.read_csv(filename, sep=",", chunksize=chunksize, error_bad_lines=False):
        while flag<2 and chunksize < 1200000:
            print(data)
        flag+=1
#     print(data.head())
    print(list(data.columns.values))

    if balance:
        data = balance_data(data)
    return data


def convert2Tensor(trainData, trainLabel, testData, testLabel, batch_size, kwargs):
    # training data
    tensor_x = torch.stack([torch.Tensor(i) for i in trainData])  # transform to torch tensors
    tensor_y = torch.Tensor(trainLabel).long()
    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, **kwargs)  # create your dataloader

    # test data
    tensor_x = torch.stack([torch.Tensor(i) for i in testData])  # transform to torch tensors
    tensor_y = torch.Tensor(testLabel).long()
    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your dataset
    test_loader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=False, **kwargs)  #

    return train_loader, test_loader


def normalize_data(x_train, x_test):
    # training
    norm_tcolumns = x_train[x_train.columns[:10]]  # only the first ten columns need normalization, the rest is binary
    #     scaler = MinMaxScaler(copy=True, feature_range=(0, 1)).fit(norm_tcolumns.values)
    scaler = MaxAbsScaler(copy=True).fit(norm_tcolumns.values)
    scaledf = scaler.transform(norm_tcolumns.values)
    training_examples = pd.DataFrame(scaledf, index=norm_tcolumns.index,
                                     columns=norm_tcolumns.columns)  # scaledf is converted from array to dataframe
    x_train.update(training_examples)

    # validation
    norm_vcolumns = x_test[x_test.columns[:10]]
    vscaled = scaler.transform(norm_vcolumns.values)  # this scaler uses std and mean of training dataset
    validation_examples = pd.DataFrame(vscaled, index=norm_vcolumns.index, columns=norm_vcolumns.columns)
    x_test.update(validation_examples)
    return x_train, x_test


def split_data(data):
    # splitting the data
    msk = np.random.rand(len(data)) < 0.8
    yy = ['Cover_Type']

    train = data[msk]
    test = data[~msk]
    #     print(train.head())
    #     print(test.head())
    x_train = train[train.columns[:train.shape[1] - 1]]  # all columns except the last are x variables
    y_train = train[yy[0]].tolist()  # the last column as y variable
    x_test = test[test.columns[:test.shape[1] - 1]]
    y_test = test[yy[0]].tolist()  # the last column as y variable

    y_train = [0 if i == 7 else i for i in y_train]  # replace label 7 with 0
    y_test = [0 if i == 7 else i for i in y_test]  # replace label 7 with 0

    return x_train, y_train, x_test, y_test

def main():
    #################################################
    ### Training settings ###########################

    inp_size = 54
    hidden_size = 50
    numClasses = 7
    batch_size = 100
    test_batch_size = 50
    epochs = 1500
    lr = 0.95
    seed = 1
    log_interval = 10
    use_cuda = True
    rho = 0.5
    thresh = 0.1

    param = {'inp_size': inp_size,'hidden_size': hidden_size, 'batch_size': batch_size, 'numClasses': numClasses, 'epochs': epochs, 'lr': lr, 'seed': seed, 'rho': rho, 'thresh': thresh}

    sparsity_param = torch.FloatTensor([rho for _ in range(hidden_size)]).unsqueeze(0)

    torch.manual_seed(seed)
    ########## Choosing GPU or CPU ######################
    device = torch.device("cuda" if use_cuda else "cpu")
    #############  Data Loader ##############
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    ####Training
    ###########################################################
    filename = '/data/covtype.csv'
    dataset = getData(filename, balance=True)
    X_train, y_train, X_test, y_test = split_data(dataset)
    X_train, X_test = normalize_data(X_train, X_test)

    print('==============Training================')
    print('number of class 1: ', y_train.count(1))
    print('number of class 2: ', y_train.count(2))
    print('number of class 3: ', y_train.count(3))
    print('number of class 4: ', y_train.count(4))
    print('number of class 5: ', y_train.count(5))
    print('number of class 6: ', y_train.count(6))
    print('number of class 7: ', y_train.count(0))

    print('==============Testing================')
    print('number of class 1: ', y_test.count(1))
    print('number of class 2: ', y_test.count(2))
    print('number of class 3: ', y_test.count(3))
    print('number of class 4: ', y_test.count(4))
    print('number of class 5: ', y_test.count(5))
    print('number of class 6: ', y_test.count(6))
    print('number of class 7: ', y_test.count(0))

    X_train = X_train.iloc[:, :].values
    X_test = X_test.iloc[:, :].values
    train_loader, test_loader = convert2Tensor(X_train, y_train, X_test, y_test, batch_size, kwargs)
    #############################################################
    ############# Instantiate Model #############################
    model = AutoEncoder(inp_size, hidden_size).to(device)
    print(model)
    classifier = Classifier(hidden_size, numClasses).to(device)
    print(classifier)
    ######################### Define optimization #####################
    optimizer  = optim.SGD(model.parameters(), lr=lr)
    optimizer_classifier = optim.SGD(classifier.parameters(), lr=0.01)
    ###################################################################
    ### Epoch training
    loss_train_arr = []
    loss_test_arr = []
    sensors_train = []
    sensors_test = []
    test_acc = []
    train_acc = []
    prob_test = []
    acc_best = 0
    for epoch in range(1, epochs + 1):
        loss_train, no_sensors_train, acc_train = train(model, classifier, device, train_loader, optimizer,
                                                              optimizer_classifier, epoch, log_interval, sparsity_param,
                                                              thresh, inp_size, batch_size)
        loss_test, no_sensors_test, acc = test(model, classifier, device, test_loader, thresh,
                                                                     inp_size)

        ## Store Metrics
        loss_train_arr.append(loss_train)
        loss_test_arr.append(loss_test)
        sensors_train.append(no_sensors_train)
        sensors_test.append(no_sensors_test)
        test_acc.append(acc)
        train_acc.append(acc_train)

    return loss_train_arr, loss_test_arr, sensors_train, sensors_test, train_acc, test_acc


if __name__ == '__main__':
        loss_train_arr, loss_test_arr, sensors_train, sensors_test, train_acc, test_acc = main()
        print('Train Accuracy', np.max(train_acc))
        print('Test Accuracy', np.max(test_acc))

        idx = np.argmax(test_acc)
        print('index is:', idx)
        print('Number of train sensors', sensors_train[idx])
        print('Number of test sensors', sensors_test[idx])
        print('Train recon loss', loss_train_arr[idx])
        print('Test recon loss', loss_test_arr[idx])
