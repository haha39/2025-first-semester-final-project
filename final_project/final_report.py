"""
AI Financial Trading System - Focused on the Taiwan Stock Market (2020-2024)
Objective: Use machine learning models to predict stock trends and compare with the 0050 benchmark.

Selected Stocks:
1. 2330.TW TSMC (Large-cap/Highly correlated with the market)
2. 2317.TW Hon Hai (AI-related/Trend-following)
3. 2603.TW Evergreen Marine (Cyclical/High volatility)
Benchmark: 0050.TW Yuanta Taiwan 50 ETF
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_pipeline import download_stock_data, engineer_features, create_labels, build_dataset

warnings.filterwarnings('ignore')

# ==================== Part 1: Technical Indicator Calculation ====================

# ==================== Added: Metric Learning (Prototypical Network) Model ====================

class ProtoEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 16, hidden_dims=(64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, embedding_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProtoNet:
    """Simplified Prototypical Networks for tabular classification.
    - Use MLP to obtain embedding
    - Episodic training: In each episode, sample K supports and Q queries from each class, minimizing the cross-entropy of queries to prototype distances
    - Inference: Use the class average vectors embedded by the entire training set as prototypes, and take the nearest prototype
    """

    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 embedding_dim: int = 16,
                 hidden_dims=(64, 32),
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 epochs: int = 40,
                 episodes_per_epoch: int = 60,
                 K: int = 5,
                 Q: int = 10,
                 temperature: float = 1.0,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.K = K
        self.Q = Q
        self.temperature = temperature
        self.device = torch.device(device)

        self.encoder = ProtoEncoder(input_dim, embedding_dim, hidden_dims, dropout).to(self.device)
        self.scaler = StandardScaler()
        self.class_to_index = None
        self.index_to_class = None
        self.prototypes = None  # shape: (n_classes, embedding_dim)

    def _to_tensor(self, x_np):
        return torch.tensor(x_np, dtype=torch.float32, device=self.device)

    def _compute_prototypes(self, emb: torch.Tensor, y_idx: np.ndarray) -> torch.Tensor:
        # emb: (N, D), y_idx: (N,)
        protos = []
        for c in range(self.n_classes):
            m = (y_idx == c)
            if m.sum() == 0:
                # If there are no samples for a class, set to zero vector
                protos.append(torch.zeros(emb.shape[1], device=self.device))
            else:
                protos.append(emb[m].mean(dim=0))
        return torch.stack(protos, dim=0)  # (C, D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Build label index mapping
        classes = np.sort(np.unique(y))
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.index_to_class = {i: c for c, i in self.class_to_index.items()}
        y_idx = np.vectorize(self.class_to_index.get)(y)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = self._to_tensor(X_scaled)

        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)

        rng = np.random.default_rng(42)
        # For each class, build an index set
        idx_by_class = {c: np.where(y_idx == c)[0] for c in range(self.n_classes)}

        # Training (simplified episodic)
        self.encoder.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            episodes = 0
            for _ in range(self.episodes_per_epoch):
                support_idx = []
                query_idx = []
                valid = True
                for c in range(self.n_classes):
                    idxs = idx_by_class[c]
                    if len(idxs) == 0:
                        valid = False
                        break
                    # If the number of samples is insufficient, allow resampling
                    s = rng.choice(idxs, size=self.K, replace=(len(idxs) < self.K))
                    q = rng.choice(idxs, size=self.Q, replace=(len(idxs) < self.Q))
                    support_idx.append(s)
                    query_idx.append(q)
                if not valid:
                    continue

                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                X_sup = X_tensor[support_idx]
                X_que = X_tensor[query_idx]
                y_que = y_idx[query_idx]

                # Obtain embedding and prototype
                z_sup = self.encoder(X_sup)  # (C*K, D)
                z_que = self.encoder(X_que)  # (C*Q, D)

                # Calculate the prototype for each class (support average)
                # Split support back to class chunks
                z_chunks = torch.chunk(z_sup, self.n_classes, dim=0)
                protos = torch.stack([zc.mean(dim=0) for zc in z_chunks], dim=0)  # (C, D)

                # Distance -> logits
                # z_que: (Nq, D), protos: (C, D)
                # pairwise distances
                dists = torch.cdist(z_que, protos, p=2)  # (Nq, C)
                logits = - (dists ** 2) / self.temperature
                loss = F.cross_entropy(logits, torch.tensor(y_que, device=self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                episodes += 1

            if episodes > 0 and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / episodes
                print(f"ProtoNet Training Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_loss:.4f}")

        # Establish final prototype with all training samples
        self.encoder.eval()
        with torch.no_grad():
            z_all = self.encoder(X_tensor)
            protos = self._compute_prototypes(z_all, y_idx)
        self.prototypes = protos.detach().cpu().numpy()
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            z = self.encoder(self._to_tensor(X_scaled))
        return z.detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._embed(X)  # (N, D)
        # Distance to prototype
        dists = ((Z[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)  # (N, C)
        idx = dists.argmin(axis=1)
        # Map back to original label
        return np.vectorize(self.index_to_class.get)(idx)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self._embed(X)
        dists = ((Z[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)
        logits = - dists / self.temperature
        # softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        # Need to correspond to original class order (0,1,2)
        # self.index_to_class: idx -> class_label
        order = [self.class_to_index[c] for c in sorted(self.class_to_index.keys())]
        return p[:, order]

    # For persistence
    def get_state(self):
        return {
            'model_type': 'prototypical',
            'input_dim': self.input_dim,
            'n_classes': self.n_classes,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'lr': self.lr,
            'epochs': self.epochs,
            'episodes_per_epoch': self.episodes_per_epoch,
            'K': self.K,
            'Q': self.Q,
            'temperature': self.temperature,
            'state_dict': {k: v.cpu().numpy() for k, v in self.encoder.state_dict().items()},
            'prototypes': self.prototypes,
            'scaler_mean_': self.scaler.mean_.copy(),
            'scaler_scale_': self.scaler.scale_.copy(),
            'class_to_index': self.class_to_index,
            'index_to_class': self.index_to_class,
        }

    @staticmethod
    def from_state(state: dict):
        model = ProtoNet(
            input_dim=state['input_dim'],
            n_classes=state['n_classes'],
            embedding_dim=state['embedding_dim'],
            hidden_dims=tuple(state['hidden_dims']),
            dropout=state['dropout'],
            lr=state['lr'],
            epochs=state['epochs'],
            episodes_per_epoch=state['episodes_per_epoch'],
            K=state['K'],
            Q=state['Q'],
            temperature=state['temperature'],
            device='cpu'
        )
        # Restore encoder weights
        sd = {k: torch.tensor(v) for k, v in state['state_dict'].items()}
        model.encoder.load_state_dict(sd)
        # Restore scaler
        model.scaler.mean_ = np.array(state['scaler_mean_'])
        model.scaler.scale_ = np.array(state['scaler_scale_'])
        model.scaler.n_features_in_ = model.input_dim
        # Restore prototypes and index mapping
        model.prototypes = np.array(state['prototypes'])
        model.class_to_index = state['class_to_index']
        model.index_to_class = state['index_to_class']
        return model

# Removed data processing code such as data download, feature engineering, label creation, and train/test split.
# These functionalities are now expected to be handled externally and passed as arguments to ProtoNetModel methods.

class ProtoNetModel:
    def __init__(self, input_dim, n_classes, use_few_shot=True, config=None):
        """
        Args:
            input_dim: Number of input features.
            n_classes: Number of classes.
            use_few_shot: Whether to enable few-shot episodic training.
            config: Configuration dictionary for ProtoNet hyperparameters.
        """
        self.use_few_shot = use_few_shot
        self.config = config or {}
        self.model = ProtoNet(
            input_dim=input_dim,
            n_classes=n_classes,
            embedding_dim=self.config.get('embedding_dim', 16),
            hidden_dims=self.config.get('hidden_dims', (64, 32)),
            dropout=self.config.get('dropout', 0.1),
            lr=self.config.get('lr', 1e-3),
            epochs=self.config.get('epochs', 40),
            episodes_per_epoch=self.config.get('episodes_per_epoch', 60),
            K=self.config.get('K', 5),
            Q=self.config.get('Q', 10),
            temperature=self.config.get('temperature', 1.0),
            device=self.config.get('device', 'cpu')
        )

    def fit(self, X_train, y_train):
        """
        Train the ProtoNet model.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predict class labels for the test set.

        Args:
            X_test: Test features.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predict class probabilities for the test set.

        Args:
            X_test: Test features.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X_test)
