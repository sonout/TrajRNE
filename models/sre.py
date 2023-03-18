import json
import os
from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model import Model
import walker


class StructuralRoadEncoder(Model):
    def __init__(
        self,
        data,  # placeholder
        device,
        network,
        emb_dim: int = 128,
        out_dim=3,
        
    ):
        """
        Initialize SRN2Vec
        Args:
            data (_type_): placeholder
            device (_type_): torch device
            network (nx.Graph): graph of city where nodes are intersections and edges are roads
            emb_dim (int, optional): embedding dimension. Defaults to 128.
        """
        self.data = data
        self.device = device
        self.emb_dim = emb_dim
        self.model = SRN2Vec(network, device=device, emb_dim=emb_dim, out_dim=out_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = nn.BCELoss()
        self.network = network

    def train(self, epochs: int = 1000, batch_size: int = 128):
        """
        Train the SRN2Vec Model (load dataset before with .load_data())
        Args:
            epochs (int, optional): epochs to train. Defaults to 1000.
            batch_size (int, optional): batch_size. Defaults to 128.
        """
        self.model.to(self.device)
        loader = DataLoader(
            SRN2VecDataset(self.data, len(self.network.line_graph.nodes)),
            batch_size=batch_size,
            shuffle=True,
        )
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for i, (X, y) in enumerate(loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.optim.zero_grad()
                yh = self.model(X)
                loss = self.loss_func(yh.squeeze(), y.squeeze())

                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                if i % 1000 == 0:
                    print(
                        f"Epoch: {e}, Iteration: {i}, sample_loss: {loss.item()}, Avg. Loss: {total_loss/(i+1)}"
                    )

            print(f"Average training loss in episode {e}: {total_loss/len(loader)}")

    

    
    def set_dataset(self, data):
        self.data = data
    
    def load_dataset(self, path: str):
        with open(path, "r") as fp:
            self.data = np.array(json.load(fp))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        ...

    def load_emb(self):
        return self.model.embedding.weight.data.cpu()

    def to(self, device):
        self.model.to(device)
        return self


class SRN2Vec(nn.Module):
    def __init__(self, network, device, emb_dim: int = 128, out_dim: int = 2):
        super(SRN2Vec, self).__init__()
        self.embedding = nn.Embedding(len(network.line_graph.nodes), emb_dim)
        self.lin_vx = nn.Linear(emb_dim, emb_dim)
        self.lin_vy = nn.Linear(emb_dim, emb_dim)

        self.lin_out = nn.Linear(emb_dim, out_dim)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        # y_emb = self.embedding(vy)

        # x = self.lin_vx(emb[:, 0])
        # y = self.lin_vy(emb[:, 1])
        x = emb[:, 0, :] * emb[:, 1, :]  # aggregate embeddings

        x = self.lin_out(x)

        yh = self.act_out(x)

        return yh


class SRN2VecDataset(Dataset):
    def __init__(self, data, num_classes: int):
        self.X = data[:, :2]
        self.y = data[:, 2:]
        self.num_cls = num_classes

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=int), torch.Tensor(
            self.y[idx]
        )  # F.one_ont(self.X[idx], self.num_cls)>
