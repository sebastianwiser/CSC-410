"""Fixed-budget CPU MLP, no hyperparameter search or test early stopping."""
import random
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, seed=42, epochs=20, hidden=32, lr=.001):
        self.seed, self.epochs, self.hidden, self.lr = seed, epochs, hidden, lr

    def fit(self, X, y):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x = torch.tensor(np.asarray(X), dtype=torch.float32)
        t = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.net_ = nn.Sequential(nn.Linear(x.shape[1], self.hidden), nn.ReLU(),
                                  nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for ix in torch.randperm(len(x)).split(256):
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self.net_(x[ix]), t[ix])
                loss.backward()
                opt.step()
        self.net_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy().ravel()
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)
