import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch.nn import Conv1d,MaxPool1d
import random


labels = {'nochange': 0, 'added': 1, 'removed': 2, 'change': 3, 'color_change': 4}


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


class Incep1(torch.nn.Module):
    def __init__(self,features):
        super().__init__()
        self.c1 = Conv1d(features,int(features/2), kernel_size=1)
        self.c2 = Conv1d(int(features/2),int(features / 2), kernel_size=1)
        self.c3 = Conv1d(int(features/2),features, kernel_size=1)
        self.bn1 = BN(int(features))
        self.bn2 = BN(int(features/2))
    def forward(self,x):
        x1 = self.c1(x)
        x1 = F.relu(x1)
        x1 = self.bn2(x1)
        x2 = self.c2(x1)
        x2 = F.relu(x2)
        x2 = self.bn2(x2)
        x3 = self.c2(x1)
        x3 = F.relu(x3)
        x3 = self.bn2(x3)
        x4 = MaxPool1d(kernel_size=1)(x1)
        x5 = self.c3(x4)
        x5 = F.relu(x5)
        x5 = self.bn1(x5)
        merged = torch.cat([x1, x2, x3, x5], dim=1)
        return merged


class Incep2(torch.nn.Module):
    def __init__(self,features):
        super().__init__()
        self.c1 = Conv1d(features,features,kernel_size=1)
        self.c2 = Conv1d(features,int(features/2), kernel_size=1)
        self.c3 = Conv1d(features,int(features/4), kernel_size=1)
        self.bn1 = BN(features)
        self.bn2= BN(int(features/2))
        self.bn3 = BN(int(features / 4))

    def forward(self,x):
        s1 = self.c1(x)
        s1 = F.relu(s1)
        s1 = self.bn1(s1)
        s1 = F.relu(s1)

        s2 = self.c2(x)
        s2 = F.relu(s2)
        s2 = self.bn2(s2)
        s2 = F.relu(s2)

        s3 = self.c3(x)
        s3 = F.relu(s3)
        s3 = self.bn3(s3)
        s3 = F.relu(s3)

        merged = torch.cat([s1,s2,s3],dim = 1)
        merged = F.relu(merged)
        return merged


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(33, 128)
        self.conv2 = GCNConv(128,64)
        self.conv3 = GCNConv(64,32)
        self.lin1 = MLP([280, 128])
        self.mlp = Seq(
            MLP([128, 64]), Dropout(0.5), MLP([64, 32]), Dropout(0.5),
            Lin(32, 6))

        self.lin2 = MLP([140, 128])
        self.mlp2 = Seq(
            MLP([128, 64]), Dropout(0.5), MLP([64, 32]), Dropout(0.5),
            Lin(32, 6))

        self.colconv = GCNConv(39, 128)
        self.finallin = Lin(12,5)
        self.i1 = Incep1(32)
        self.i2 = Incep2(80)


    def forward(self, data16, data20,datacol):
        x16, edge_index16, batch16 = data16.features, data16.edge_index, data16.batch
        x20, edge_index20, batch20 = data20.features, data20.edge_index, data20.batch
        xcol,edge_indexcol,batchcol = datacol.features, datacol.edge_index , datacol.batch
        x16 = self.conv1(x16, edge_index16)
        x16 = F.relu(x16)
        x16 = self.conv2(x16, edge_index16)
        x16 = F.relu(x16)
        x16 = self.conv3(x16, edge_index16)
        x16 = torch.transpose(x16,0,1)
        x16 = self.i1(x16.unsqueeze(0))
        x16 = self.i2(x16)
        x16 = torch.transpose(x16.squeeze(0),0,1)
        x16 = global_mean_pool(x16,batch16)

        x20 = self.conv1(x20, edge_index20)
        x20 = F.relu(x20)
        x20 = self.conv2(x20, edge_index20)
        x20 = F.relu(x20)
        x20 = self.conv3(x20, edge_index20)
        x20 = torch.transpose(x20, 0, 1)
        x20 = self.i1(x20.unsqueeze(0))
        x20 = self.i2(x20)
        x20 = torch.transpose(x20.squeeze(0), 0, 1)

        x20 = global_mean_pool(x20, batch20)

        xcol = self.colconv(xcol, edge_indexcol)
        xcol = F.relu(xcol)
        xcol = self.conv2(xcol, edge_indexcol)
        xcol = F.relu(xcol)
        xcol = self.conv3(xcol, edge_indexcol)
        xcol = torch.transpose(xcol, 0, 1)
        xcol = self.i1(xcol.unsqueeze(0))
        xcol = self.i2(xcol)
        xcol = torch.transpose(xcol.squeeze(0), 0, 1)

        xcol = global_mean_pool(xcol, batchcol)
        xcol = self.lin2(xcol)

        xcol = self.mlp2(xcol)

        x = torch.cat([x16, x20], dim=1)
        x = self.lin1(x)
        x = self.mlp(x)
        out =torch.cat([x,xcol],dim = 1)
        out = self.finallin(out)
        return F.log_softmax(out,dim =-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(loader):
    model.train()
    correct = 0
    confmatrix = np.zeros((5, 5))
    predictions = []
    total_loss = 0
    for i in loader:
        if i[0].flag == 1 and i[1].flag == 1:
            if (labels[i[0].truelab[0]] == 0):
                correct += 1
                if (labels[i[0].truelab[0]] == 0):
                    confmatrix[0][0] += 1
                    correct += 1
                else:
                    confmatrix[0][labels[i[0].truelab[0]]] += 1
            predictions.append(0)
        else:
            data16 = i[0].to(device)
            data20 = i[1].to(device)
            dc = i[2].to(device)
            optimizer.zero_grad()
            model.eval()
            out = model(data16,data20,dc)
            model.train()
            loss = F.nll_loss(out,data16.y)
            yval = labels[data16.truelab[0]]
            pred = out.max(dim=1)[1].cpu().detach().numpy()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct +=out.max(dim=1)[1].eq(data16.y).sum().item()
            predictions.append(out.max(dim=1)[1].cpu().detach().numpy())
            confmatrix[pred[0]][yval] += 1
    return correct/len(loader),predictions,confmatrix


def test(loader):
    model.eval()
    confmatrix = np.zeros((5,5))
    correct = 0
    predictions = []
    for i in loader:
        if i[0].flag == 1 and i[1].flag == 1:
            if labels[i[0].truelab[0]] == 0:
                confmatrix[0][0] += 1
                correct += 1
            else:
                confmatrix[0][labels[i[0].truelab[0]]] += 1
            predictions.append(0)
        else:
            data16 = i[0].to(device)
            data20 = i[1].to(device)
            dc = i[2].to(device)
            with torch.no_grad():
                out = model(data16,data20,dc)
            yval = labels[data16.truelab[0]]
            pred = out.max(dim=1)[1].cpu().detach().numpy()
            correct += out.max(dim=1)[1].eq(data16.y).sum().item()
            predictions.append(pred)
            confmatrix[pred[0]][yval] +=1
    return correct/len(loader),predictions,confmatrix


def save_model(model):
    torch.save(model.state_dict(), "/home/rada/sdp/data/model1.pth")


def load_model(model):
    model.load_state_dict(torch.load("/home/rada/sdp/data/model1.pth"))
    model.eval()


if __name__ == '__main__':
    maxacc = 0
    with open("data/train.dat", "rb") as fp:
        loader_train = pickle.load(fp)
    with open("data/validation.dat", "rb") as fp2:
        loader_test = pickle.load(fp2)

    for epoch in range(0, 120):
        print(epoch)
        print("train:",end =' ')
        trainacc, trainpreds,train_confusion_matrix = train(loader_train)
        random.shuffle(loader_train)
        print(trainacc)
        print(train_confusion_matrix)
        print("test",end=' ')
        testacc, testpreds,test_confusion_matrix = test(loader_train)
        print(testacc)
        print(test_confusion_matrix)
        random.shuffle(loader_test)

        #if testacc > maxacc:
            #save_model(model)




