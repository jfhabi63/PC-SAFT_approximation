from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

input_length = 6
num_targets = 2


def load_raw_data(rand_seed):
    dataset_kwargs = {"sep": ';',
                      "header": 0,
                      "float_precision": 'round_trip',
                      "na_values": 'none'}
    raw_data = pd.read_csv("./input/paper1_sphere_gesamt.csv",sep=";",header=0,index_col=0)
    raw_data = raw_data.astype('float32')
    X = raw_data.iloc[:,0:6]
    y = raw_data.iloc[:,7:10:2]
    comp = raw_data.index
    raw_data_test = pd.read_csv("./input/paper1_random_gesamt.csv", sep=";", header=0, index_col=0)
    raw_data_test = raw_data_test.astype('float32')
    X_test = raw_data_test.iloc[:,0:6]
    y_test = raw_data_test.iloc[:,7:10:2]
    print(y)
    comp_test = raw_data_test.index

    X_train = X
    y_train = y
    comp_train = comp

    #X_train, X_test, y_train, y_test, comp_train, comp_test = train_test_split(X,y,comp,test_size=0.15,random_state=rand_seed)

    scx = StandardScaler()
    scy = StandardScaler()
    X_train = scx.fit_transform(X_train)
    X_test = scx.transform(X_test)

    y_train = scy.fit_transform(y_train)
    y_test = scy.transform(y_test)

    scaler_parameters = pd.DataFrame(data=[list(scx.mean_)+list(scy.mean_),list(scx.var_)+list(scy.var_)])
    scaler_parameters.to_csv("./scaler_parameters.csv",sep=";")


    joblib.dump(scx, './scaler/std_scaler_X.bin', compress=True)
    joblib.dump(scy, './scaler/std_scaler_y.bin', compress=True)


    return X_train,X_test,y_train,y_test,comp_train,comp_test


class Dataset_input(Dataset):
    def __init__(self,X,y, comp, transform=None,train=True):
        # data loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform

        self.x = torch.from_numpy(X).to(device)
        if train == True:
            self.y = torch.from_numpy(y).to(device)  #reshape(-1, 1)   .to_numpy(dtype=np.float32)
        self.n_samples = X.shape[0]
        self.comp = comp
        self.input_length = input_length

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x=self.transform(x)
        try:
            return x, y
        except AttributeError:
            return x

    def __len__(self):
        self.size = len(self.x)
        return self.size


def retransform(y):
    scy = joblib.load("./scaler/std_scaler_y.bin")
    y = pd.DataFrame(y)
    y = np.round(scy.inverse_transform(y), 6)
    return y