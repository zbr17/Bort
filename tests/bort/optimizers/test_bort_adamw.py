import unittest

import torch

from bort import BortA

class TestBortA(unittest.TestCase):
    def test_BortA(self):
        model = torch.nn.Linear(6, 2, bias=False)
        optimizer = BortA(model.parameters(), lr=0.1, weight_decay=0.0, gamma=1, mode="default", amptitude=1.0)

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
            self.assertTrue(torch.allclose(sim_mat, torch.eye(2), atol=1e-3))