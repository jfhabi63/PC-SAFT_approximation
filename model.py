import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self):
        from Dataset import num_targets, input_length
        super(Model, self).__init__()
        act_fct=nn.LeakyReLU()
        self.net1 = nn.Sequential(nn.Linear(input_length, 512),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(512, 1024),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(1024, 2048),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(2048, 1024),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(1024, num_targets),
                                  )
    def forward(self, x):
        forward = self.net1(x)
        return forward

def train(dataloader, model, optimizer, loss_history=None):
    model.train()
    from Dataset import input_length
    tmp_loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X).to(device)
        criterion= torch.nn.L1Loss()  #torch.nn.MSELoss()   #torch.nn.HuberLoss()  ##pc_param torch.nn.L1Loss()
        loss =criterion(pred,y).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"loss: {loss.item():>7f}")
        tmp_loss_history.append(loss.item())
    if loss_history is not None:
        loss_history.append(np.mean(tmp_loss_history))
        print(f"train_summary: \nloss: {np.mean(tmp_loss_history):>7f}")


def validate(dataloader,model,loss_history=None):
    model.eval()
    loss_gesamt = []
    tmp_loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)

        criterion= torch.nn.L1Loss() #criterion=torch.nn.HuberLoss() torch.nn.MSELoss()
        loss=criterion(pred,y).to(device)
        loss_gesamt.append(loss.item())
        tmp_loss_history.append(loss.item())

    if loss_history is not None:
        loss_history.append(np.mean(tmp_loss_history))
    return np.mean(loss_gesamt)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def visualize_training(train_loss, test_loss, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(train_loss)), train_loss, label="train")
    ax1.plot(range(len(test_loss)), test_loss, label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.axis([0,len(train_loss),0,0.1])
    ax1.legend()

    if filename:
        plt.savefig(filename)
    plt.close()