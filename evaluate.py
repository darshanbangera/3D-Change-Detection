import os
import torch
from train import Net
import pickle
import numpy as np
labels = {'nochange': 0, 'added': 1, 'removed': 2, 'change': 3, 'color_change': 4}
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--Mode', type=str, default="train_val", help='train_val or test ')
opt = parser.parse_args()
if opt.Mode == "train_val":
    op = "/data/validation.dat"
if opt.Mode =="test":
    op ="/data/test.dat"


def inference(loader, path='data/Model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    confmatrix = np.zeros((5, 5))
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
            confmatrix[pred[0]][yval] += 1
    return correct/len(loader), predictions,confmatrix


if __name__ == '__main__':
    with open("datapath.txt", "r") as myfile:
        path = myfile.readlines()[0]
    path = path[:-1] + op
    with open(path, "rb") as fp2:
        loader_test = pickle.load(fp2)

    test_acc,predictions,confmatrix = inference(loader_test)

    print("Test accuracy: {:.5f}".format(test_acc))
