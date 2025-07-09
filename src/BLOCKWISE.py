##########################
#   blockwise_forecast.py
##########################

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime
import os

##########################
#   GLOBAL CONFIG        #
##########################
CONFIG = {
    "model_save_path": "saved_models",
    "selected_model": "MLP",    # Optionen: "LSTM", "MLP", "GRU", "AttentionLSTM"
    
    # Trainings-Hyperparameter
    "max_epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.002439124727737658,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.0,
    
    # Dynamische Epochenanpassung für den rekursiven Blockwise-Ansatz
    "initial_block_epochs": 50,
    "epoch_decay_step": 10,
    "min_block_epochs": 20,
    "initial_block_size": 24,
    
    # Data-Konfiguration
    "date_column": "index",
    "target_column": "price_winsorized",
    "feature_columns": [
        "GTI * gewichtung(sun)",
        "windspeed * gewichtung(wind)",
        "is_weekend"
        #"wind10_GTI"
    ],
    
    # Zeit-Parameter
    "training_weeks": 52,
    "sequence_length": 24,    # Kurzes Input-Fenster (z. B. 24 Stunden)
    "forecast_horizon": 168,
    "block_size": 24,
    
    # Scaler
    "scaler_type": "MinMax",
    
    # Gewichtungen
    "wind_weight": 1.0,
    "sun_weight": 1.0,
    
    # Loss / Metrik
    "loss_function": "MSE",
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Plot-Steuerung
    "show_plots": False,
    
    "verbose": True
}

##########################
#   DATA LOADING & PREP  #
##########################
def load_and_prepare_data(config):
    df = pd.read_pickle("ready_for_MODEL.pkl")
    df = df.copy()
    df = df.iloc[50:] # ersten 50 zeilen entfernen
    df.fillna(0, inplace=True) #NaNs durch 0 ersetzen

    df["wind10"] = df["windspeed * gewichtung(wind)"] * config["wind_weight"]
    df["wind10_GTI"] = df["wind10"] + df["GTI * gewichtung(sun)"] * config["sun_weight"]
    needed_cols = config["feature_columns"] + [config["target_column"]]
    df = df[needed_cols].copy()
    df.sort_index(inplace=True)
    
    scaler_X, scaler_y = None, None
    X = df[config["feature_columns"]].values
    y = df[config["target_column"]].values.reshape(-1,1)
    
    if config["scaler_type"] == "Standard":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
    elif config["scaler_type"] == "MinMax":
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
    
    df_scaled = df.copy()
    df_scaled[config["feature_columns"]] = X
    df_scaled[config["target_column"]] = y
    
    return df_scaled, scaler_X, scaler_y

##########################
#   DATASET & MODELS     #
##########################
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, sequence_length, forecast_horizon=1):
        super().__init__()
        self.data = data
        self.targets = targets
        self.seq_len = sequence_length
        self.horizon = forecast_horizon
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len + self.horizon - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class MLPModel(nn.Module):
    def __init__(self, flattened_input_dim, hidden_size, num_layers, dropout):
        super(MLPModel, self).__init__()
        layers = []
        in_dim = flattened_input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.net(x)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(AttentionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.attn_layer = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attn_layer(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        output = self.fc(context)
        return output

def create_model(model_type, input_dim, hidden_size, num_layers, dropout, sequence_length=None):
    if model_type == "LSTM":
        return LSTMModel(input_dim, hidden_size, num_layers, dropout)
    elif model_type == "MLP":
        if sequence_length is None:
            raise ValueError("Für MLP muss sequence_length angegeben werden.")
        return MLPModel(sequence_length * input_dim, hidden_size, num_layers, dropout)
    elif model_type == "GRU":
        return GRUModel(input_dim, hidden_size, num_layers, dropout)
    elif model_type == "AttentionLSTM":
        return AttentionLSTMModel(input_dim, hidden_size, num_layers, dropout)
    else:
        raise ValueError(f"Unbekanntes Modell: {model_type}")

def get_loss_function(loss_name):
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "MAE":
        return nn.L1Loss()
    elif loss_name == "Huber":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unbekannte Loss-Funktion: {loss_name}")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds.squeeze(), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(dataloader.dataset)

##########################
#   BLOCKWISE FUNKTIONEN
##########################
def train_and_forecast_blockwise(config, forecast_start_date):
    """
    Rekursiver blockweiser Forecast mit dynamischer Epochenanpassung.
    Erzeugt ein DataFrame (df_result) für den Zeitraum [start_date - sequence_length, start_date + forecast_horizon]
    mit unskalierten Features, realen Targets und Predictions.
    Gibt zusätzlich ein Dictionary mit den Metriken zurück.
    """
    df_scaled, scaler_X, scaler_y = load_and_prepare_data(config)
    idx_start = df_scaled.index.get_loc(forecast_start_date)
    train_hours = config["training_weeks"] * 168
    idx_train_start = max(0, idx_start - train_hours)
    sequence_length = config["sequence_length"]
    horizon = config["forecast_horizon"]
    
    # Für den ersten Block: separate Blockgröße; danach Standardblockgröße
    current_block_size = config.get("initial_block_size", config["block_size"])
    standard_block_size = config["block_size"]
    
    # Dynamische Epochenparameter
    initial_block_epochs = config.get("initial_block_epochs", config["max_epochs"])
    epoch_decay_step = config.get("epoch_decay_step", 0)
    min_block_epochs = config.get("min_block_epochs", 10)
    
    augmented_train_data = df_scaled.iloc[idx_train_start:idx_start].copy()
    
    # Initiales Training auf echten (skalierte) Daten
    train_dataset = TimeSeriesDataset(augmented_train_data[config["feature_columns"]].values,
                                      augmented_train_data[config["target_column"]].values,
                                      sequence_length, 1)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    input_dim = len(config["feature_columns"])
    model = create_model(config["selected_model"], input_dim, config["hidden_size"],
                         config["num_layers"], config["dropout"], sequence_length=sequence_length)
    model.to(config["device"])
    criterion = get_loss_function(config["loss_function"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(initial_block_epochs):
        loss_val = train_one_epoch(model, train_loader, optimizer, criterion, config["device"])
        if config["verbose"]:
            print(f"[Blockwise] Initial Train Epoch {epoch+1}/{initial_block_epochs} - Loss: {loss_val:.4f}")
    
    current_index = idx_start
    overall_predictions = []  # Speichert Tupel (i, pred) in skalierter Form
    overall_forecast_dates = []
    block_count = 0
    
    while current_index < idx_start + horizon:
        block_count += 1
        
        if block_count == 1:
            block_size_used = current_block_size
        else:
            block_size_used = standard_block_size
        
        block_end = min(current_index + block_size_used, idx_start + horizon)
        
        for t in range(current_index, block_end):
            seq_data = augmented_train_data.iloc[-sequence_length:][config["feature_columns"]].values
            x_input = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(config["device"])
            with torch.no_grad():
                pred = model(x_input).squeeze().cpu().numpy()
            overall_predictions.append((t, pred))
            overall_forecast_dates.append(df_scaled.index[t])
            new_features = df_scaled.iloc[t][config["feature_columns"]].values
            new_target = pred  # Bleibt im skalierten Raum
            new_row = pd.DataFrame([np.concatenate((new_features, [new_target]))],
                                   columns=config["feature_columns"] + [config["target_column"]],
                                   index=[df_scaled.index[t]])
            augmented_train_data = pd.concat([augmented_train_data, new_row])
        
        current_index = block_end
        
        current_block_epochs = max(min_block_epochs, initial_block_epochs - (block_count - 1) * epoch_decay_step)
        
        X_train = augmented_train_data[config["feature_columns"]].values
        y_train = augmented_train_data[config["target_column"]].values
        retrain_dataset = TimeSeriesDataset(X_train, y_train, sequence_length, 1)
        retrain_loader = DataLoader(retrain_dataset, batch_size=config["batch_size"], shuffle=True)
        model = create_model(config["selected_model"], input_dim, config["hidden_size"],
                             config["num_layers"], config["dropout"], sequence_length=sequence_length)
        model.to(config["device"])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        for epoch in range(current_block_epochs):
            loss_val = train_one_epoch(model, retrain_loader, optimizer, criterion, config["device"])
            if config["verbose"]:
                print(f"[Blockwise] Block {block_count} Retrain Epoch {epoch+1}/{current_block_epochs} - Loss: {loss_val:.4f}")
    
    start_idx_for_df = max(0, idx_start - sequence_length)
    end_idx_for_df = min(len(df_scaled)-1, idx_start + horizon - 1)
    df_result = df_scaled.iloc[start_idx_for_df: end_idx_for_df+1].copy()
    df_result["prediction"] = np.nan
    
    for (i_pos, pred_val) in overall_predictions:
        label = df_scaled.index[i_pos]
        if label in df_result.index:
            df_result.loc[label, "prediction"] = pred_val
    
    if scaler_y is not None:
        real_scaled = df_result[config["target_column"]].values.reshape(-1,1)
        real_unscaled = scaler_y.inverse_transform(real_scaled).flatten()
        df_result["real"] = real_unscaled
    else:
        df_result["real"] = df_result[config["target_column"]]
    
    # Transformiere die prediction-Spalte zurück (jetzt aus skalierten Werten)
    if scaler_y is not None:
        mask = df_result["prediction"].notna()
        preds_scaled = df_result.loc[mask, "prediction"].values.astype(float).reshape(-1,1)
        preds_unscaled = scaler_y.inverse_transform(preds_scaled).flatten()
        df_result.loc[mask, "prediction"] = preds_unscaled
    
    if scaler_X is not None:
        scaled_feats = df_result[config["feature_columns"]].values
        unscaled_feats = scaler_X.inverse_transform(scaled_feats)
        df_result[config["feature_columns"]] = unscaled_feats
    
    preds_series = df_result.loc[df_result.index[idx_start: end_idx_for_df+1], "prediction"].dropna()
    real_series = df_result.loc[preds_series.index, "real"]
    
    metrics = {}
    if len(preds_series) > 0:
        mse = mean_squared_error(real_series, preds_series)
        mae = mean_absolute_error(real_series, preds_series)
        r2 = r2_score(real_series, preds_series)
        metrics = {"mse": mse, "mae": mae, "r2": r2}
        print(f"[Blockwise] Forecast Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    else:
        print("[Blockwise] Keine Vorhersagen im angegebenen Zeitraum!")
    
    if config.get("show_plots", True):
        plt.figure(figsize=(12,6))
        plt.plot(df_result.index, df_result["real"], label="Real", marker='o')
        plt.plot(df_result.index, df_result["prediction"], label="Prediction", marker='x')
        plt.xticks(rotation=45)
        plt.title("Rekursiver Blockweiser Forecast")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if not os.path.exists(config["model_save_path"]):
        os.makedirs(config["model_save_path"])
    model_file = os.path.join(config["model_save_path"], "my_blockwise_model_recursive.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": config["selected_model"],
        "input_dim": input_dim,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"]
    }, model_file)
    print(f"[Blockwise] Modell gespeichert unter: {model_file}")
    
    return df_result, metrics

def load_and_forecast_blockwise(config, forecast_start_date):
    """
    Lädt das rekursiv trainierte blockweise Modell und erzeugt ein DataFrame,
    das den Zeitraum [start_date - sequence_length, start_date + forecast_horizon] abdeckt.
    Gibt zusätzlich ein Dictionary mit den berechneten Metriken zurück.
    """
    df_scaled, scaler_X, scaler_y = load_and_prepare_data(config)
    idx_start = df_scaled.index.get_loc(forecast_start_date)
    horizon = config["forecast_horizon"]
    sequence_length = config["sequence_length"]
    
    model_file = os.path.join(config["model_save_path"], "my_blockwise_model_recursive.pt")
    checkpoint = torch.load(model_file, map_location=config["device"])
    
    model_type = checkpoint["model_type"]
    input_dim = checkpoint["input_dim"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]
    
    model = create_model(model_type, input_dim, hidden_size, num_layers, dropout, sequence_length=sequence_length)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config["device"])
    model.eval()
    
    start_idx_for_df = max(0, idx_start - sequence_length)
    end_idx_for_df = min(len(df_scaled)-1, idx_start + horizon - 1)
    df_result = df_scaled.iloc[start_idx_for_df: end_idx_for_df+1].copy()
    df_result["prediction"] = np.nan
    
    predictions = []
    for i in range(idx_start, min(idx_start + horizon, len(df_scaled))):
        if i - sequence_length < 0:
            continue
        window_data = df_scaled.iloc[i-sequence_length:i][config["feature_columns"]].values
        x_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(config["device"])
        with torch.no_grad():
            pred = model(x_input).squeeze().cpu().numpy()
        if scaler_y is not None:
            pred = scaler_y.inverse_transform(pred.reshape(-1,1)).flatten()[0]
        predictions.append((i, pred))
    
    for (i_pos, pred_val) in predictions:
        label = df_scaled.index[i_pos]
        if label in df_result.index:
            df_result.loc[label, "prediction"] = pred_val
    
    if scaler_y is not None:
        real_scaled = df_result[config["target_column"]].values.reshape(-1,1)
        real_unscaled = scaler_y.inverse_transform(real_scaled).flatten()
        df_result["real"] = real_unscaled
    else:
        df_result["real"] = df_result[config["target_column"]]
    
    if scaler_X is not None:
        scaled_feats = df_result[config["feature_columns"]].values
        unscaled_feats = scaler_X.inverse_transform(scaled_feats)
        df_result[config["feature_columns"]] = unscaled_feats
    
    preds_series = df_result.loc[df_result.index[idx_start: end_idx_for_df+1], "prediction"].dropna()
    real_series = df_result.loc[preds_series.index, "real"]
    
    metrics = {}
    if len(preds_series) > 0:
        mse = mean_squared_error(real_series, preds_series)
        mae = mean_absolute_error(real_series, preds_series)
        r2 = r2_score(real_series, preds_series)
        metrics = {"mse": mse, "mae": mae, "r2": r2}
        print(f"[Blockwise Load] Forecast Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    else:
        print("[Blockwise Load] Keine Vorhersagen im gegebenen Zeitraum!")
    
    if config.get("show_plots", True):
        plt.figure(figsize=(12,6))
        plt.plot(df_result.index, df_result["real"], label="Real", marker='o')
        plt.plot(df_result.index, df_result["prediction"], label="Prediction", marker='x')
        plt.xticks(rotation=45)
        plt.title("Rekursiver Blockweiser Forecast (Geladenes Modell)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return df_result, metrics

if __name__ == "__main__":
    blockwise_start_date = "2025-04-02 00:00:00"
    df_result_block, metrics_block = train_and_forecast_blockwise(CONFIG, forecast_start_date=blockwise_start_date)
    df_result_block.to_pickle("df_blockwise_result.pkl")
    
    df_loaded_block, load_metrics_block = load_and_forecast_blockwise(CONFIG, forecast_start_date=blockwise_start_date)
    df_loaded_block.to_pickle("df_blockwise_loaded.pkl")
    
    print("[Blockwise] Fertig!")
