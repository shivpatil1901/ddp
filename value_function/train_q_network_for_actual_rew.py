# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import os
# import warnings

# warnings.filterwarnings("ignore")

# # --------------------------------------------------
# # CONFIGURATION
# # --------------------------------------------------

# EXPERT_DEMOS_PATH = "bc_gaussian_policy/expert_demos_ant_velocity_300.npz"
# SAVE_PATH = "surrogate_q_mc_regression.pth"

# GAMMA = 0.99
# BATCH_SIZE = 512
# LR = 3e-4
# NUM_EPOCHS = 30

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------------------------------
# # LOAD TRAJECTORIES
# # --------------------------------------------------

# print("Loading trajectories...")
# data = np.load(EXPERT_DEMOS_PATH, allow_pickle=True)
# trajectories = data["trajectories"]
# print(f"Loaded {len(trajectories)} trajectories.")

# # --------------------------------------------------
# # BUILD (STATE, ACTION) → MC RETURN DATASET
# # --------------------------------------------------

# print("Computing Monte Carlo returns...")

# X = []   # [state, action]
# Y = []   # MC return

# for traj in trajectories:
#     states = np.array(traj["states"])
#     actions = np.array(traj["actions"])
#     rewards = np.array(traj["rewards"])

#     G = 0.0
#     mc_returns = []

#     # Backward MC return computation
#     for r in reversed(rewards):
#         G = r + GAMMA * G
#         mc_returns.insert(0, G)

#     # Align lengths (states: T+1, actions/rewards: T)
#     T = len(actions)
#     for t in range(T):
#         sa = np.concatenate([states[t], actions[t]])
#         X.append(sa)
#         Y.append(mc_returns[t])

# X = np.array(X, dtype=np.float32)
# Y = np.array(Y, dtype=np.float32)

# print(f"Dataset size: {len(X)}")
# print(f"Return stats: mean={Y.mean():.2f}, std={Y.std():.2f}, "
#       f"min={Y.min():.2f}, max={Y.max():.2f}")

# # --------------------------------------------------
# # NORMALIZE TARGETS (CRITICAL FOR STABILITY)
# # --------------------------------------------------

# Y_mean = Y.mean()
# Y_std = Y.std() + 1e-6
# Y_norm = (Y - Y_mean) / Y_std

# # --------------------------------------------------
# # PYTORCH DATASET
# # --------------------------------------------------

# X_tensor = torch.tensor(X, dtype=torch.float32)
# Y_tensor = torch.tensor(Y_norm, dtype=torch.float32).unsqueeze(1)

# dataset = TensorDataset(X_tensor, Y_tensor)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # --------------------------------------------------
# # MODEL DEFINITION
# # --------------------------------------------------

# class QRegressor(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         return self.net(x)

# model = QRegressor(X.shape[1]).to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# loss_fn = nn.MSELoss()

# # --------------------------------------------------
# # TRAINING LOOP (PURE REGRESSION)
# # --------------------------------------------------

# print("\nTraining MC-return regressor...")

# model.train()
# for epoch in range(NUM_EPOCHS):
#     epoch_loss = 0.0

#     for xb, yb in dataloader:
#         xb = xb.to(DEVICE)
#         yb = yb.to(DEVICE)

#         pred = model(xb)
#         loss = loss_fn(pred, yb)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(dataloader)
#     print(f"Epoch {epoch+1:02d} | MSE (normalized): {avg_loss:.4f}")

# # --------------------------------------------------
# # EVALUATION (R² ON TRAINING SET)
# # --------------------------------------------------

# model.eval()
# with torch.no_grad():
#     preds_norm = model(X_tensor.to(DEVICE)).cpu().numpy().flatten()
#     preds = preds_norm * Y_std + Y_mean

# ss_res = np.sum((Y - preds) ** 2)
# ss_tot = np.sum((Y - Y.mean()) ** 2)
# r2 = 1.0 - ss_res / (ss_tot + 1e-8)

# print(f"\nTraining R²: {r2:.3f}")
# if r2 < 0.3:
#     print("⚠️ WARNING: Low R² → rules may be weak or noisy")

# # --------------------------------------------------
# # SAVE MODEL
# # --------------------------------------------------

# save_dir = os.path.dirname(SAVE_PATH)
# if save_dir:
#     os.makedirs(save_dir, exist_ok=True)
# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "input_dim": X.shape[1],
#         "return_mean": Y_mean,
#         "return_std": Y_std,
#         "gamma": GAMMA,
#     },
#     SAVE_PATH,
# )

# print(f"\n✅ MC-return Q-regressor saved to {SAVE_PATH}")


import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import pickle5 as _pickle5  # type: ignore[import-not-found]
except ImportError:
    _pickle5 = None

import pickle

try:
    import joblib
except ImportError:
    joblib = None


DEFAULT_ROLLOUT_DATA_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym_original/ppo_lagrangian_PointGoal1_s0_diverse_2000.pickle"
DEFAULT_SAVE_PATH = "q_function_models/q_network_rollout_only.pt"
DEFAULT_SCALER_PATH = "q_function_models/q_scaler_rollout_only.pkl"


def _load_pickle(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ValueError as e:
        if "unsupported pickle protocol" not in str(e):
            raise
        if _pickle5 is not None:
            with open(path, "rb") as f:
                return _pickle5.load(f)
        raise RuntimeError(
            "Dataset uses pickle protocol 5 but pickle5 is unavailable on Python %d.%d"
            % (sys.version_info.major, sys.version_info.minor)
        ) from e


def _split_trajectories_from_arrays(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    states = np.asarray(payload["states"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    rewards = np.asarray(payload["rewards"], dtype=np.float32).reshape(-1)
    costs = np.asarray(payload.get("costs", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
    dones = np.asarray(payload.get("dones", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)

    n = int(min(len(states), len(actions), len(rewards), len(costs), len(dones)))
    trajectories: List[Dict[str, np.ndarray]] = []
    start = 0
    for i in range(n):
        if bool(dones[i] >= 0.5) or i == (n - 1):
            end = i + 1
            if end > start:
                trajectories.append(
                    {
                        "states": states[start:end],
                        "actions": actions[start:end],
                        "rewards": rewards[start:end],
                        "costs": costs[start:end],
                        "dones": dones[start:end],
                    }
                )
            start = end
    return trajectories


def load_rollout_trajectories(dataset_path: str) -> List[Dict[str, np.ndarray]]:
    payload = _load_pickle(dataset_path)
    if isinstance(payload, dict) and "trajectories" in payload:
        trajectories = payload["trajectories"]
    elif isinstance(payload, list):
        trajectories = payload
    elif isinstance(payload, dict) and all(k in payload for k in ["states", "actions", "rewards"]):
        trajectories = _split_trajectories_from_arrays(payload)
    else:
        raise ValueError("Unsupported dataset format for rollout training")

    normalized: List[Dict[str, np.ndarray]] = []
    for i, traj in enumerate(trajectories):
        states = np.asarray(traj["states"], dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        rewards = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)
        n = int(min(len(states), len(actions), len(rewards)))
        if n <= 0:
            continue
        normalized.append(
            {
                "states": states[:n],
                "actions": actions[:n],
                "rewards": rewards[:n],
            }
        )
        if (i + 1) % 500 == 0:
            print("Validated %d trajectories..." % (i + 1))

    if len(normalized) == 0:
        raise ValueError("No valid trajectories found in %s" % dataset_path)
    return normalized


class RobustQNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _cuda_arch_supported() -> (bool, str):
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
    except Exception as e:
        return False, "Failed to query CUDA device capability: %s" % str(e)

    required_arch = "sm_%d%d" % (major, minor)
    arch_list = []
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            arch_list = []

    if len(arch_list) == 0:
        # If arch list is unavailable, assume compatibility and let runtime decide.
        return True, "Could not read torch CUDA arch list; proceeding with CUDA"

    if required_arch not in arch_list:
        return (
            False,
            "GPU %s requires %s, but current torch build supports: %s"
            % (device_name, required_arch, " ".join(arch_list)),
        )

    return True, "CUDA architecture is compatible"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-network from rollout trajectories only")
    parser.add_argument("--rollout_data_path", type=str, default=DEFAULT_ROLLOUT_DATA_PATH)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--scaler_path", type=str, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("LOADING ROLLOUT TRAJECTORIES (NO EXPERT DATA)")
    print("=" * 70)
    print("Rollout dataset:", args.rollout_data_path)
    all_trajectories = load_rollout_trajectories(args.rollout_data_path)
    print("Loaded trajectories:", len(all_trajectories))

    inputs = []
    returns = []

    print("\nProcessing trajectories...")
    for idx, traj in enumerate(all_trajectories):
        s = np.asarray(traj["states"], dtype=np.float32)
        a = np.asarray(traj["actions"], dtype=np.float32)
        r = np.asarray(traj["rewards"], dtype=np.float32)

        g = 0.0
        g_t = []
        for rew in reversed(r):
            g = float(rew) + float(args.gamma) * g
            g_t.insert(0, g)

        n = int(min(len(s), len(a), len(g_t)))
        for i in range(n):
            inputs.append(np.concatenate([s[i], a[i]], axis=0))
            returns.append(g_t[i])

        if (idx + 1) % 500 == 0:
            print("  Processed %d/%d trajectories..." % (idx + 1, len(all_trajectories)))

    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(returns, dtype=np.float32)

    print("Dataset size:", len(x))
    print("Return stats -> Mean: %.2f | Std: %.2f | Min: %.2f | Max: %.2f" % (float(y.mean()), float(y.std()), float(y.min()), float(y.max())))

    if len(x) > int(args.max_samples):
        print("\nDataset too large (%d). Subsampling to %d..." % (len(x), int(args.max_samples)))
        idx = np.random.choice(len(x), int(args.max_samples), replace=False)
        x = x[idx]
        y = y[idx]
        print("Subsampled dataset size:", len(x))

    print("Normalizing inputs...")
    input_scaler = StandardScaler()
    x_scaled = input_scaler.fit_transform(x)

    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-6)
    y_scaled = (y - y_mean) / y_std

    if args.device == "auto":
        if torch.cuda.is_available():
            ok, reason = _cuda_arch_supported()
            if ok:
                device = "cuda"
            else:
                print("[WARN] %s. Falling back to CPU." % reason)
                device = "cpu"
        else:
            device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available in this environment")
        ok, reason = _cuda_arch_supported()
        if not ok:
            raise RuntimeError("--device cuda requested, but %s" % reason)
        device = "cuda"
    else:
        device = "cpu"
    print("Using device:", device)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)
    pin_memory = device == "cuda"
    dataloader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=int(args.batch_size),
        shuffle=True,
        pin_memory=pin_memory,
    )

    model = RobustQNetwork(x.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)
    loss_fn = nn.MSELoss()

    print("\nStarting training...")
    for epoch in range(int(args.epochs)):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            batch_y = batch_y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0:
            print("Epoch %02d | Loss: %.4f | LR: %.1e" % (epoch + 1, avg_loss, optimizer.param_groups[0]["lr"]))

    model.eval()
    preds_chunks = []
    eval_loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=max(1, int(args.batch_size) * 4),
        shuffle=False,
        pin_memory=pin_memory,
    )
    with torch.no_grad():
        for (batch_x,) in eval_loader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            batch_pred = model(batch_x)
            preds_chunks.append(batch_pred.detach().cpu().numpy().reshape(-1))
    preds_scaled = np.concatenate(preds_chunks, axis=0)

    preds_real = (preds_scaled * y_std) + y_mean
    ss_res = float(np.sum((y - preds_real) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))

    print("\n" + "=" * 40)
    print("FINAL R^2 SCORE: %.4f" % r2)
    print("=" * 40)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    scaler_dir = os.path.dirname(args.scaler_path)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "return_mean": y_mean,
            "return_std": y_std,
            "gamma": float(args.gamma),
            "rollout_data_path": args.rollout_data_path,
        },
        args.save_path,
    )
    print("Model saved to:", args.save_path)

    if joblib is not None:
        joblib.dump(input_scaler, args.scaler_path)
        print("Scaler saved to:", args.scaler_path)
    else:
        with open(args.scaler_path, "wb") as f:
            pickle.dump(input_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Scaler saved with pickle (joblib unavailable) to:", args.scaler_path)

    y_stats_path = os.path.join(os.path.dirname(args.save_path) or ".", "y_stats.npz")
    np.savez(y_stats_path, mean=y_mean, std=y_std)
    print("Y stats saved to:", y_stats_path)


if __name__ == "__main__":
    main()
