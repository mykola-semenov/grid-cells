import torch
from torch import optim
from torch import nn
from gridlayer import GridTissue
from dataset import Room2D
import numpy as np
from matplotlib import pyplot as plt
from autoencoder import OptModel
from torch.utils.data import DataLoader

room = Room2D(20,20)
loss_fn = nn.MSELoss()

def train(n_neurons, room, loss_fn, n_epochs=100, batch_size=20, learning_rate=0.01):

    room_dl =  DataLoader(room, shuffle=True, batch_size=batch_size)
    model = OptModel(n_neurons)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss = 0

    losses = torch.zeros((n_epochs,))
    for epoch in range(n_epochs):
        
        for train_x in room_dl:

            optimizer.zero_grad()
            pos_hat = model(train_x)

            loss = loss_fn(pos_hat, train_x)

            loss.backward(retain_graph=True)
            optimizer.step()

            losses[epoch] += loss.item()

        print(f"Epoch: {epoch}, loss: {losses[epoch]}")

    return losses


train(10, room, loss_fn)
        

