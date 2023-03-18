import numpy as np
from .model import Model

class TrajRNE(Model):
    def __init__(self, data, device, aggregator: str, models: list, network=None):
        self.models = models
        self.device = device
        self.aggregator = aggregator

    def train(self):
        ...

    def load_emb(self):
        embs = [m.load_emb() for m in self.models]
        if self.aggregator == "add":
            emb = embs[0]
            for e in embs[1:]:
                emb = emb + e
            return emb

        elif self.aggregator == "concate":
            return np.concatenate(embs, axis=1)

    def load_model(self, path: str):
        ...


