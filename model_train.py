"""Time-series forecasting utilities converted from the original notebook."""

from pathlib import Path
import math
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset, Subset

# Configure plotting style and device.
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data preparation pipeline.
DATA_PATH = Path("") # Đường dẫn input
OUTPUT_DIR = Path("") # Đường dẫn output
SEED = 42
seed_everything(SEED)
print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw dataset, sort by time, and print a quick summary."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(df.head())
    return df


def process_data_with_features_lite(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-based price/volume features, daily return, and time encodings."""
    df = df.copy()
    df["open_log"] = np.log(df["open"] + 1e-8)
    df["high_log"] = np.log(df["high"] + 1e-8)
    df["low_log"] = np.log(df["low"] + 1e-8)
    df["close_log"] = np.log(df["close"] + 1e-8)
    df["volume_log"] = np.log1p(df["volume"])
    df["daily_return"] = df["close"].pct_change().fillna(0)
    if "time" in df.columns:
        dates = pd.to_datetime(df["time"])
        df["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    return df


def plot_correlations(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Print correlation with close_log and plot the full heatmap."""
    features = ["open_log", "high_log", "low_log", "close_log", "volume_log", "daily_return", "dow_sin", "dow_cos"]
    corr_matrix = df_processed[features].corr()
    print("--- Correlation with close_log ---")
    print(corr_matrix["close_log"].sort_values(ascending=False))
    return corr_matrix


# Dataset utilities.
class MultivariateTimeSeriesDataset(Dataset):
    """Create sliding windows for multivariate time series."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        target_cols: list,
        feature_cols: list,
        normalize: bool = False,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.normalize = normalize
        self.data = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        if self.normalize:
            self.scaler = {}
            all_cols = list(set(self.feature_cols + self.target_cols))
            for col in all_cols:
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.scaler[col] = {"mean": mean, "std": std}
                if std != 0:
                    self.data[col] = (self.data[col] - mean) / std
                else:
                    self.data[col] = 0
        else:
            self.scaler = None
        self.feature_data = self.data[self.feature_cols].values.astype(np.float32)
        self.target_data = self.data[self.target_cols].values.astype(np.float32)
        self.num_features = self.feature_data.shape[1]
        self.num_targets = len(self.target_cols)

    def __len__(self) -> int:
        return len(self.feature_data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int):
        x = self.feature_data[idx : idx + self.seq_len]
        y = self.target_data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_datasets(df: pd.DataFrame, seq_lengths: list, pred_len: int, target_cols: list, feature_cols: list):
    """Initialize datasets for multiple sequence lengths."""
    datasets = {}
    for seq_len in seq_lengths:
        datasets[f"{seq_len}d"] = MultivariateTimeSeriesDataset(
            df=df,
            seq_len=seq_len,
            pred_len=pred_len,
            target_cols=target_cols,
            feature_cols=feature_cols,
            normalize=False,
        )
    return datasets


# Model definitions.
class Trend_MLP(nn.Module):
    """MLP block that models trend across time and features."""

    def __init__(self, seq_len: int, pred_len: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.time_linear = nn.Linear(seq_len, pred_len)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.feature_linear = nn.Linear(num_features, 1)
        self.feature_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.time_linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.feature_linear(x)
        x = self.feature_act(x)
        return x


class RevIN(nn.Module):
    """Reversible instance normalization for per-sample normalization."""

    def __init__(self, num_features: int, eps: float = 1e-3, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self) -> None:
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x


class SimpleTransformer_Seasonal(nn.Module):
    """Lightweight Transformer encoder-decoder for seasonal patterns."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_features: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_embedding = nn.Linear(num_features, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, seq_len + pred_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.decoder_query_token = nn.Parameter(torch.randn(pred_len, d_model))

    def forward(self, seasonal_input: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = seasonal_input.shape
        enc_input = self.input_embedding(seasonal_input)
        enc_input = enc_input + self.position_encoding[:, :length, :]
        encoder_output = self.transformer_encoder(enc_input)
        decoder_query = self.decoder_query_token.unsqueeze(0).repeat(bsz, 1, 1)
        decoder_query = decoder_query + self.position_encoding[:, length : length + self.pred_len, :]
        decoder_output = self.transformer_decoder(tgt=decoder_query, memory=encoder_output)
        return decoder_output


class DAutoformer_M_TargetFocused(nn.Module):
    """Dual-branch model combining trend MLP, seasonal transformer, and RevIN."""

    def __init__(self, seq_len: int, pred_len: int = 5, num_features: int = 5, moving_avg: int = 7, d_model: int = 64):
        super().__init__()
        self.revin = RevIN(num_features)
        safe_moving_avg = min(moving_avg, seq_len)
        if safe_moving_avg % 2 == 0:
            safe_moving_avg -= 1
        self.moving_avg = max(1, safe_moving_avg)
        self.register_buffer("avg_kernel", torch.ones(num_features, 1, self.moving_avg) / self.moving_avg)
        self.trend_model = Trend_MLP(seq_len, pred_len, num_features)
        self.seasonal_model = SimpleTransformer_Seasonal(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features, d_model=d_model, n_layers=1, n_heads=2, dropout=0.2
        )
        self.activation = nn.GELU()
        self.projection = nn.Linear(d_model, 1)

    def decompose(self, x: torch.Tensor):
        x_permuted = x.permute(0, 2, 1)
        padding = (self.moving_avg - 1) // 2
        trend = F.conv1d(x_permuted, self.avg_kernel, padding=padding, groups=self.revin.num_features)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

    def forward(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        x = self.revin(x, "norm")
        trend_input, seasonal_input = self.decompose(x)
        trend_pred = self.trend_model(trend_input)
        seasonal_base = self.seasonal_model(seasonal_input)
        seasonal_pred = self.projection(seasonal_base)
        trend_pred = self.activation(trend_pred)
        final_pred = trend_pred + seasonal_pred
        target_mean = self.revin.mean[:, :, target_idx : target_idx + 1]
        target_stdev = self.revin.stdev[:, :, target_idx : target_idx + 1]
        if self.revin.affine:
            target_bias = self.revin.affine_bias[target_idx]
            target_weight = self.revin.affine_weight[target_idx]
            final_pred = final_pred - target_bias
            final_pred = final_pred / (target_weight + self.revin.eps)
        final_pred = final_pred * target_stdev + target_mean
        return final_pred


# Training utilities and configuration.
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 150
PRED_LEN = 100
N_SPLITS = 5
PATIENCE = 15
MAX_TRAIN_SIZE = 365
SEQ_LENGTHS = [120, 240, 480]
FEATURE_COLS = ["volume_log", "close_log", "daily_return", "dow_sin", "dow_cos"]
TARGET_COLS = ["close_log"]


def train_one_epoch(model, train_loader, criterion, optimizer, device, target_idx: int) -> float:
    """Run a single training epoch."""
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x, target_idx=target_idx)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, target_idx: int) -> float:
    """Evaluate the model on a validation split."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, target_idx=target_idx)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_with_rolling_window(datasets: dict) -> tuple[dict, dict]:
    """Train models with rolling window cross-validation."""
    print(f"=== Training DAutoformer models with rolling window (max {MAX_TRAIN_SIZE} samples) ===")
    best_models = {}
    cv_results = {}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for seq_name, dataset in datasets.items():
        print(f"\n{'=' * 40}")
        print(f"Sequence Length: {seq_name} | Total Samples: {len(dataset)}")
        try:
            close_idx = dataset.feature_cols.index("close_log")
        except ValueError:
            raise ValueError("close_log must be present in feature_cols.")
        fold_metrics = []
        indices = np.arange(len(dataset))
        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
            if len(train_idx) > MAX_TRAIN_SIZE:
                train_idx = train_idx[-MAX_TRAIN_SIZE:]
            print(f"\n  --- Fold {fold + 1}/{N_SPLITS} ---")
            print(f"  Train size: {len(train_idx)} (capped) | Val size: {len(val_idx)}")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
            model = DAutoformer_M_TargetFocused(
                seq_len=dataset.seq_len,
                pred_len=PRED_LEN,
                num_features=dataset.num_features,
                d_model=64,
            ).to(DEVICE)
            criterion = nn.HuberLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=PATIENCE, verbose=False)
            best_val_loss = float("inf")
            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, target_idx=close_idx)
                val_loss = evaluate(model, val_loader, criterion, DEVICE, target_idx=close_idx)
                scheduler.step(val_loss)
                if (epoch + 1) % 20 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"    Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            fold_metrics.append(best_val_loss)
            print(f"  -> Best Val Loss for Fold {fold + 1}: {best_val_loss:.4f}")
            if fold == N_SPLITS - 1:
                best_models[seq_name] = model
        avg_val_loss = np.mean(fold_metrics)
        cv_results[seq_name] = avg_val_loss
        print(f"\n  => Rolling average Val Loss for {seq_name}: {avg_val_loss:.4f}")
    print("\n=== Training completed ===")
    print("Average validation losses:")
    for name, loss in cv_results.items():
        print(f"- {name}: {loss:.4f}")
    return best_models, cv_results


# Forecasting helpers.
def recursive_predict_100_days_v4(
    model: nn.Module, initial_input: torch.Tensor, steps: int = 100, feature_cols: list | None = None, n_simulations: int = 50
):
    """Monte Carlo recursive forecast that adds stochastic noise and clamps extremes."""
    model.eval()
    device = next(model.parameters()).device
    try:
        close_idx = feature_cols.index("close_log")
        vol_idx = feature_cols.index("volume_log") if "volume_log" in feature_cols else -1
    except ValueError:
        print("close_log missing from feature_cols.")
        return None, None, None
    current_input = initial_input.clone().to(device)
    hist_close = current_input[0, :, close_idx].cpu().numpy()
    hist_diff = np.diff(hist_close)
    std_volatility = np.std(hist_diff) if len(hist_diff) > 0 else 0.02
    mean_vol = 0
    if vol_idx != -1:
        hist_vol = current_input[0, :, vol_idx].cpu().numpy()
        mean_vol = np.mean(hist_vol)
    all_sim_paths = np.zeros((n_simulations, steps))
    for sim in range(n_simulations):
        sim_input = initial_input.clone().to(device)
        for i in range(steps):
            with torch.no_grad():
                pred_out = model(sim_input, target_idx=close_idx)
                pred_val = pred_out[0, 0, 0].item()
                noise = np.random.normal(0, std_volatility)
                last_close_val = sim_input[0, -1, close_idx].item()
                pred_val_stochastic = pred_val + noise
                max_change = 0.07
                delta = pred_val_stochastic - last_close_val
                if delta > max_change:
                    pred_val_stochastic = last_close_val + max_change
                elif delta < -max_change:
                    pred_val_stochastic = last_close_val - max_change
                all_sim_paths[sim, i] = pred_val_stochastic
                next_input_row = sim_input[0, -1, :].clone().unsqueeze(0).unsqueeze(0)
                next_input_row[0, 0, close_idx] = torch.tensor(
                    pred_val_stochastic, device=device, dtype=next_input_row.dtype
                )
                if vol_idx != -1:
                    vol_shock = np.random.normal(0, 0.5)
                    next_vol = mean_vol + vol_shock
                    next_input_row[0, 0, vol_idx] = torch.tensor(
                        next_vol, device=device, dtype=next_input_row.dtype
                    )
                if "daily_return" in feature_cols:
                    ret_idx = feature_cols.index("daily_return")
                    next_ret = pred_val_stochastic - last_close_val
                    next_input_row[0, 0, ret_idx] = torch.tensor(
                        next_ret, device=device, dtype=next_input_row.dtype
                    )
                sim_input = torch.cat([sim_input[:, 1:, :], next_input_row], dim=1)
    mean_forecast = np.mean(all_sim_paths, axis=0)
    upper_bound = np.percentile(all_sim_paths, 90, axis=0)
    lower_bound = np.percentile(all_sim_paths, 10, axis=0)
    return mean_forecast, upper_bound, lower_bound


def predict_block_recursive(model: nn.Module, initial_input: torch.Tensor, steps: int = 100, feature_cols: list | None = None):
    """Chunked recursive forecast that rolls forward by blocks of pred_len."""
    model.eval()
    device = next(model.parameters()).device
    try:
        close_idx = feature_cols.index("close_log")
        vol_idx = feature_cols.index("volume_log")
        ret_idx = feature_cols.index("daily_return")
    except ValueError:
        print("Feature list must include close_log, volume_log, and daily_return.")
        return None
    try:
        pred_len = model.seasonal_model.pred_len
    except Exception:
        pred_len = 30
    current_input = initial_input.clone().to(device)
    hist_vol = current_input[0, :, vol_idx].cpu().numpy()
    mean_vol = np.mean(hist_vol)
    std_vol = np.std(hist_vol)
    all_predictions = []
    num_blocks = math.ceil(steps / pred_len)
    for _ in range(num_blocks):
        with torch.no_grad():
            block_pred = model(current_input, target_idx=close_idx)
            block_pred_np = block_pred[0, :, 0].cpu().numpy()
            current_step = len(all_predictions)
            remaining_steps = steps - current_step
            if len(block_pred_np) > remaining_steps:
                block_pred_np = block_pred_np[:remaining_steps]
            all_predictions.extend(block_pred_np)
            if len(all_predictions) >= steps:
                break
            last_close_log = current_input[0, -1, close_idx].item()
            next_block_features = torch.zeros(1, len(block_pred_np), len(feature_cols)).to(device)
            next_block_features[0, :, close_idx] = torch.tensor(block_pred_np).to(device)
            concat_prices = np.concatenate(([last_close_log], block_pred_np))
            block_returns = np.diff(concat_prices)
            next_block_features[0, :, ret_idx] = torch.tensor(block_returns).to(device)
            vol_noise = np.random.normal(0, std_vol * 0.5, size=len(block_pred_np))
            block_vol = mean_vol + vol_noise + (np.abs(block_returns) * mean_vol * 5)
            next_block_features[0, :, vol_idx] = torch.tensor(block_vol).to(device)
            seq_len = current_input.shape[1]
            extended_input = torch.cat([current_input, next_block_features], dim=1)
            current_input = extended_input[:, -seq_len:, :]
    return np.array(all_predictions)


def run_monte_carlo_forecasts(
    best_models: dict,
    datasets: dict,
    feature_cols: list,
    steps: int = 100,
    n_simulations: int = 50,
    output_dir: Path = OUTPUT_DIR,
    single_output_path: Path | None = None,
):
    """Run Monte Carlo forecasts for each trained model and save CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Monte Carlo forecasts for {len(best_models)} models ===")
    for seq_key, model_curr in best_models.items():
        print(f"\n>> Processing model: {seq_key}")
        if seq_key not in datasets:
            print(f"   Dataset for {seq_key} missing; skipping.")
            continue
        dataset_curr = datasets[seq_key]
        last_seq_data = dataset_curr.feature_data[-dataset_curr.seq_len :]
        input_tensor = torch.FloatTensor(last_seq_data).unsqueeze(0).to(DEVICE)
        forecast_log_mean, forecast_log_upper, forecast_log_lower = recursive_predict_100_days_v4(
            model=model_curr,
            initial_input=input_tensor,
            steps=steps,
            feature_cols=feature_cols,
            n_simulations=n_simulations,
        )
        if forecast_log_mean is None:
            continue
        pred_price_mean = np.exp(forecast_log_mean)
        pred_price_upper = np.exp(forecast_log_upper)
        pred_price_lower = np.exp(forecast_log_lower)
        filename = single_output_path if single_output_path is not None else output_dir / f"submission_{seq_key}_v4.csv"
        submission = pd.DataFrame({"id": range(1, steps + 1), "close": pred_price_mean})
        submission.to_csv(filename, index=False)
        print(f"   Saved: {filename}")
    print("\n=== Monte Carlo forecasts complete ===")


def run_blockwise_forecasts(
    best_models: dict, datasets: dict, feature_cols: list, steps: int = 100, output_dir: Path = OUTPUT_DIR
):
    """Run block-wise forecasts for each trained model and save CSVs/plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting block-wise prediction loop for {len(best_models)} models ---")
    for seq_key, model_best in best_models.items():
        print(f"\n>>> Processing model: {seq_key}")
        if seq_key not in datasets:
            print(f"Skipping {seq_key}: dataset not found.")
            continue
        dataset_best = datasets[seq_key]
        last_seq_data = dataset_best.feature_data[-dataset_best.seq_len :]
        input_tensor = torch.FloatTensor(last_seq_data).unsqueeze(0).to(DEVICE)
        forecast_log_block = predict_block_recursive(
            model=model_best, initial_input=input_tensor, steps=steps, feature_cols=feature_cols
        )
        if forecast_log_block is None:
            print(f"Forecast failed for {seq_key}")
            continue
        pred_price_block = np.exp(forecast_log_block)
        history_len = 150
        close_idx = feature_cols.index("close_log")
        full_history_log = dataset_best.feature_data[:, close_idx]
        full_history_price = np.exp(full_history_log)
        last_hist_idx = len(full_history_price)
        history_index = range(last_hist_idx - history_len, last_hist_idx)
        future_index = range(last_hist_idx, last_hist_idx + steps)
        plt.figure(figsize=(14, 7))
        plt.plot(history_index, full_history_price[-history_len:], label="History (Last 150 days)", color="blue", linewidth=2)
        plt.plot(
            future_index,
            pred_price_block,
            label=f"Block-wise Forecast ({steps} days)",
            color="green",
            linewidth=2,
            linestyle="-",
        )
        plt.title(f"Stock Price Forecast (Block-wise) - Model {seq_key}")
        plt.xlabel("Time (Days)")
        plt.ylabel("Price (VND)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"forecast_chart_{seq_key}.png")
        plt.show()
        print(f"Saved forecast chart: {output_dir / ('forecast_chart_' + seq_key + '.png')}")
    print("\n=== All block-wise predictions completed ===")


# Orchestration entrypoint.
def main() -> None:
    """Load data, train models, and run both forecast strategies."""
    if not DATA_PATH.exists():
        print(f"Data file not found at {DATA_PATH}; update DATA_PATH before running.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = load_data(DATA_PATH)
    df_processed = process_data_with_features_lite(raw_df)
    datasets = create_datasets(df_processed, SEQ_LENGTHS, PRED_LEN, TARGET_COLS, FEATURE_COLS)
    best_models_all, cv_results = train_with_rolling_window(datasets)
    best_seq_key = min(cv_results, key=cv_results.get)
    print(f"\n>>> Selected best sequence length: {best_seq_key} (val loss {cv_results[best_seq_key]:.4f})")
    best_models = {best_seq_key: best_models_all[best_seq_key]}
    best_datasets = {best_seq_key: datasets[best_seq_key]}
    best_csv = OUTPUT_DIR / "best_submission.csv"
    run_monte_carlo_forecasts(
        best_models,
        best_datasets,
        FEATURE_COLS,
        steps=PRED_LEN,
        n_simulations=50,
        output_dir=OUTPUT_DIR,
        single_output_path=best_csv,
    )
    run_blockwise_forecasts(best_models, best_datasets, FEATURE_COLS, steps=PRED_LEN, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
