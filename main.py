import pickle

import pandas as pd
import torch
import torchvision
from copy import copy
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import *
from model import *
import random

def main(rand_seed,lr,wd):
    torch.manual_seed(2)
    X_train, X_test, y_train, y_test, comp_train, comp_test = load_raw_data(rand_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = Dataset_input(X_train,y_train,comp_train)
    test_dataset = Dataset_input(X_test,y_test,comp_test)



    loader_kwargs = {
        "batch_size": 512,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    train_loader = DataLoader(dataset=train_dataset,
                              **loader_kwargs, shuffle=True
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             **loader_kwargs, shuffle=False
                             )

    build_dir = Path("build")
    if not build_dir.exists():
        build_dir.mkdir()


    model_path = build_dir / "nn.torch"
    model = Model().to(device)
    model.apply(weights_init_uniform_rule)
    model.to(device)

    epochs = 2000
    loss_history = {"train": [], "validation": []}
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=wd,
                                 lr=lr
                                 )
    loss_min = 0.05
    epoch_min = 0

    for t in range(epochs):
        print(
            f"\nEpoch {t + 1}/{epochs}"
            "\n-------------------------------"
            f"\ntrain (seed:{rand_seed}):"
        )

        train(train_loader, model, optimizer, loss_history["train"])
        print("validate:")
        val_loss = validate(test_loader, model, loss_history["validation"])
        if val_loss < loss_min:
            epoch_min = t+1
            loss_min = val_loss
        print(f"loss: {val_loss:.5f}\n"
              f"minimal loss: {loss_min:.5f} ; epoch: {epoch_min}")
        if val_loss < 0.00563:
            break

    torch.save(model,
               './model/PC-SAFT_approx')
    torch.save(model.state_dict(),"./model/state_dict.pt")
    visualize_training(
        loss_history["train"],
        loss_history["validation"],
        build_dir / "loss.png",
    )
    limits=[5,500]
    y_pred_test, y_test = calc_pred(test_loader,model)
    MAE_vector_test = calc_MAE(y_pred_test, y_test,limits)
    while len(MAE_vector_test)< y_pred_test.shape[0]:
        MAE_vector_test.append("")
    output = pd.concat([y_pred_test,y_test],axis = 1) #columns=["AARD_rho_pred","AARD_pLV_pred"])  columns=["AARD_rho_pred"])
    output["stats"] = MAE_vector_test
    output.to_csv("./output_test.csv",sep=";")

    y_pred_train, y_train = calc_pred(train_loader,model)
    MAE_vector_train = calc_MAE(y_pred_train, y_train,limits)
    while len(MAE_vector_train)< y_pred_train.shape[0]:
        MAE_vector_train.append("")

    output = pd.concat([y_pred_train,y_train],axis = 1) #columns=["AARD_rho_pred","AARD_pLV_pred"])  columns=["AARD_rho_pred"])
    output["stats"] = MAE_vector_train
    output.to_csv("./output_train.csv",sep=";")
    return [MAE_vector_test[0:4],MAE_vector_train[0:4],[loss_min],[epoch_min],[lr],[wd]]

def calc_pred(data_loader, model):
    torch.manual_seed(1)
    model.eval()
    y_pred = []
    y_test = []
    for batch, (Xi,yi) in enumerate(data_loader):
        Xi = Xi.to(device)
        yi= yi.to(device)

        pred = model(Xi)
        pred_cpu = pred.cpu().detach().numpy()

        y_pred.extend(pred_cpu.tolist())
        y_test.extend(yi.cpu().detach().numpy().tolist())
    y_pred = pd.DataFrame(data=retransform(y_pred))
    y_test = pd.DataFrame(data=retransform(y_test))

    return y_pred,y_test

def calc_MAE(y_pred,y_true,limit):
    MAE=list(np.mean(abs(y_pred-y_true)))
    for i in range(y_pred.shape[1]):
        MAE.append(np.mean([abs(y_pred.iloc[k,i]-y_true.iloc[k,i]) for k in range(y_pred.shape[0]) if y_pred.iloc[k,i] < limit[i]]))
    return MAE


results_save = []
MAE_test_min = []

steps = 1
lr_base = [0.00056]#0.0004
wd_base = [0]#2e-7

lr_used = []
rndm = True

for i in range(steps):
    for k in range(len(wd_base)):
        lr_temp = lr_base[k]
        wd_temp = wd_base[k]
        #lr_temp = round(lr_temp + random.uniform(-4e-4,10e-4),5)
        #wd_temp = round(wd_temp + random.uniform(-1e-7,8e-7),7)
        if lr_temp not in lr_used and rndm==True:
            results = main(11, lr_temp, wd_temp)
            results_flatten = [item for sublist in results for item in sublist]
            MAE_test_min.append(results_flatten[0])
            results_save.append(results_flatten)
        lr_used.append(lr_temp)
        print(F"MAE_min: {min(MAE_test_min):.4f}\n"
              F"iteration: {i + 1}/{steps}\n"
              F"calculation: {k+1} / {steps*len(wd_base)}")


output_df = pd.DataFrame(data=results_save, columns=["MAE_AARD_test","MAE_RMSE_test","MAE_AARD_test_limit","MAE_RMSE_test_limit","MAE_AARD_train","MAE_RMSE_train","MAE_AARD_train_limit","MAE_RMSE_train_limit","loss_min","epoch_min","lr","wd"])
if steps !=1 or len(lr_base)!=1:
    output_df.to_csv("./optim.csv",sep=";")