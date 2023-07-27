import torch
from gridlayer import GridTissue
from dataset import Room2D
import numpy as np
from matplotlib import pyplot as plt
from autoencoder import OptModel

model = GridTissue(n=10)

print(model(torch.tensor([5,6])))

room = Room2D(20,20)

print(room[10])

# for i in range(5):
#     print(model(room[i]))

# x = torch.linspace(0,30,200)
# y = torch.linspace(0,30,200)

# X, Y = np.meshgrid(x, y)

# def grid(X, Y):
#   result = torch.zeros((len(X), len(X[0])))
#   for i in range(len(x)):
#     for j in range(len(y)):
#       result[i,j]= model(torch.tensor([x[i], y[j]]))[0]

#   return result.detach().numpy()

# plt.imshow(grid(X, Y))
# plt.show()

model = OptModel(10)

print(model(room[20]))
