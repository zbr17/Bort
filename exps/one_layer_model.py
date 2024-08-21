import torch
from torch import Tensor

from bort import BortS, BortA

##########################################
# Example 1: BortS
##########################################

print("Experiment 1: BortS")
model = torch.nn.Linear(6, 2, bias=False)
optimizer = BortS(model.parameters(), lr=0.1, gamma=1, mode="default", amptitude=1.0)

with torch.no_grad():
    weight = model.weight.data
    sim_mat = weight @ weight.t()
    print("row", sim_mat)
    sim_mat = weight.t() @ weight
    print("col", sim_mat)

for i in range(10000):
    optimizer.zero_grad()
    data = torch.randn(10, 6)
    output = model(data)
    loss = output.sum() * 0
    loss.backward()
    optimizer.step()

with torch.no_grad():
    weight = model.weight.data
    sim_mat = weight @ weight.t()
    print("row", sim_mat)
    sim_mat = weight.t() @ weight
    print("col", sim_mat)

##########################################
# Example 1: BortA
##########################################

print("Experiment 2: BortA")
model = torch.nn.Linear(6, 2, bias=False)
optimizer = BortA(model.parameters(), lr=0.1, weight_decay=0.0, gamma=1, mode="default", amptitude=1.0)

with torch.no_grad():
    weight = model.weight.data
    sim_mat = weight @ weight.t()
    print("row", sim_mat)
    sim_mat = weight.t() @ weight
    print("col", sim_mat)

for i in range(10000):
    optimizer.zero_grad()
    data = torch.randn(10, 6)
    output = model(data)
    loss = output.sum() * 0
    loss.backward()
    optimizer.step()

with torch.no_grad():
    weight = model.weight.data
    sim_mat = weight @ weight.t()
    print("row", sim_mat)
    sim_mat = weight.t() @ weight
    print("col", sim_mat)