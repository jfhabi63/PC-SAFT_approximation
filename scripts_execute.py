import pandas as pd
from torch.utils.data import DataLoader
import joblib
import torch
from torch.utils.data import Dataset
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_raw_data():
    dataset_kwargs = {"sep": ';',
                      "header": 0,
                      "float_precision": 'round_trip',
                      "na_values": 'none'}
    raw_data = pd.read_csv("./active_model/input.csv",sep=";",header=0,index_col=0)
    raw_data = raw_data.astype('float32')
    X = raw_data.iloc[:,0:6]
    print(X)
    comp = raw_data.index


    sc_X = joblib.load('./active_model/std_scaler_X.bin')
    X = sc_X.transform(X)
    print(X)
    return X,comp


class Dataset_input(Dataset):
    def __init__(self,X,comp, transform=None):
        # data loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform
        self.x = torch.from_numpy(X).to(device)
        self.n_samples = X.shape[0]
        self.comp = comp
    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x=self.transform(x)
        return x
    def __len__(self):
        self.size = len(self.x)
        return self.size


def calc_pred(data_loader, model):
    torch.manual_seed(1)
    model.eval()
    y_pred = []
    y_test = []
    for batch, Xi in enumerate(data_loader):
        Xi = Xi.to(device)
        pred = model(Xi)
        pred_cpu = pred.cpu().detach().numpy()

        y_pred.extend(pred_cpu.tolist())
    y_pred = pd.DataFrame(data=retransform(y_pred))
    return y_pred

def retransform(y):
    scy = joblib.load("./active_model/std_scaler_y.bin")
    y = pd.DataFrame(y)
    y = np.round(scy.inverse_transform(y), 6)
    return y
def retransform_X(X):
    scx = joblib.load("./active_model/std_scaler_X.bin")
    print(scx.mean_,scx.var_)
    X = pd.DataFrame(X)
    X = np.round(scx.inverse_transform(X),6)
    return X