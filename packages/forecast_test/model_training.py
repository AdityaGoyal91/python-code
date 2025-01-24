import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Tuple

class PredictionModel:
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()
        self.nn_model = None
        self.lr_model = None
        self.rf_model = None
        self.best_model = None

    def prepare_data(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.features].values
        y = df[target].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train_models(self, X: np.ndarray, y: np.ndarray):
        self.nn_model = self._train_nn_model(X, y)
        self.lr_model = LinearRegression().fit(X, y)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

    def _train_nn_model(self, X: np.ndarray, y: np.ndarray, epochs=100, batch_size=32):
        class SimpleDataset(Dataset):
            def __init__(self, features, targets):
                self.features = torch.tensor(features, dtype=torch.float32)
                self.targets = torch.tensor(targets, dtype=torch.float32)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]

        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super(SimpleNN, self).__init__()
                self.layer1 = nn.Linear(input_dim, 64)
                self.layer2 = nn.Linear(64, 32)
                self.layer3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.relu(self.layer2(x))
                return self.layer3(x)

        dataset = SimpleDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = SimpleNN(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        for _ in range(epochs):
            for batch_features, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()

        return model

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()
        else:
            y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return rmse

    def select_best_model(self, X: np.ndarray, y: np.ndarray):
        nn_rmse = self.evaluate_model(self.nn_model, X, y)
        lr_rmse = self.evaluate_model(self.lr_model, X, y)
        rf_rmse = self.evaluate_model(self.rf_model, X, y)

        best_rmse = min(nn_rmse, lr_rmse, rf_rmse)
        if best_rmse == nn_rmse:
            self.best_model = self.nn_model
        elif best_rmse == lr_rmse:
            self.best_model = self.lr_model
        else:
            self.best_model = self.rf_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if isinstance(self.best_model, nn.Module):
            self.best_model.eval()
            with torch.no_grad():
                return self.best_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().squeeze()
        else:
            return self.best_model.predict(X_scaled)
