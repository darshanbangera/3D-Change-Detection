import data_utils as du
from torch_geometric.data import DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--Mode', type=str, default="train_val", help='train_val or test ')
opt = parser.parse_args()

dir_1 = "Shrec_change_detection_dataset_public/2016/"
dir_2 = "Shrec_change_detection_dataset_public/2020/"
classified_dir = "Shrec_change_detection_dataset_public/labeled_point_lists_train/2016-2020/"



d16,d20,data_col= du.generate_dataset(dir_1, dir_2, classified_dir)



ids = np.arange(len(d16))
tr = []
tes= []
if opt.Mode =="train_val":
    tr, tes = train_test_split(ids, test_size=0.20)
if opt.Mode =="val":
    tr = ids
    tes=[]
train16 = []
train20 = []
traincol = []

test16 = []
test20 = []
test_col = []



for i in range(len(d16)):
    if i in tr:
        train16.append(d16[i])
        train20.append(d20[i])
        traincol.append(data_col[i])
    if i in tes:
        test16.append(d16[i])
        test20.append(d20[i])
        test_col.append(data_col[i])


# Creating train set loader
loader16 = DataLoader(train16, batch_size=1, shuffle=False)
loader20 = DataLoader(train20, batch_size=1, shuffle=False)
loader_col = DataLoader(traincol, batch_size=1, shuffle=False)

loader_train = []
for i,j,k in zip(loader16,loader20,loader_col):
    temp = []
    temp.append(i)
    temp.append(j)
    temp.append(k)
    loader_train.append(temp)


# Creating Test set loader
loader16_tes = DataLoader(test16, batch_size=1, shuffle=False)
loader20_tes = DataLoader(test20, batch_size=1, shuffle=False)
loadercol_tes= DataLoader(test_col, batch_size=1, shuffle=False)

loader_test = []
for i,j,k in zip(loader16_tes,loader20_tes,loadercol_tes):
    temp = []
    temp.append(i)
    temp.append(j)
    temp.append(k)
    loader_test.append(temp)

if opt.Mode =="train_val":
    filehandler1 = open("data/train.dat","wb")
    pickle.dump(loader_train, filehandler1)
    filehandler1.close()

    filehandler2 = open("data/validation.dat","wb")
    pickle.dump(loader_test, filehandler2)
    filehandler2.close()
if opt.Mode == "test":
    filehandler1 = open("data/test.dat", "wb")
    pickle.dump(loader_train, filehandler1)
    filehandler1.close()
