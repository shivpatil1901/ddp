import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import pickle5 as pickle
except ImportError:
    import pickle

# ==========================================
# 1. RUDDER Architecture (Original Linear)
# ==========================================
class OriginalRUDDER(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, dropout=0.2):
        super(OriginalRUDDER, self).__init__()
        self.input_dim = state_dim + action_dim
        # Standard LSTM - PyTorch defaults (h0, c0) to zero, ensuring the "Zeroing" check.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Final layer MUST be linear (no activation) for Decomposition property.
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        """Returns predictions for every timestep: (Batch, Seq_Len, 1)"""
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out)
        drop_out = self.dropout(norm_out)
        return self.output_layer(drop_out)

    def get_redistributed_signals(self, trajectory, baseline=0.0):
        """
        Extracts step-wise signals using Identity Sequence Decomposition.
        Formula: r_t = g(t) - g(t-1)
        """
        self.eval()
        with torch.no_grad():
            if len(trajectory.shape) == 2:
                trajectory = trajectory.unsqueeze(0)
            
            # g_t is the prediction of total return at each step
            g_t = self.forward(trajectory).squeeze() 
            
            # Calculate Differences
            redistributed = torch.zeros_like(g_t)
            redistributed[0] = g_t[0]
            redistributed[1:] = g_t[1:] - g_t[:-1]
            
            # Re-add baseline share to keep the mean reward consistent
            dt = 1.0 / len(redistributed)
            final_signals = redistributed + (baseline * dt)
            
        return final_signals

# ==========================================
# 2. Training Wrapper with Baseline Subtraction
# ==========================================
class RudderTrainer:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-3,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        grad_clip_norm=1.0,
    ):
        self.model = OriginalRUDDER(state_dim, action_dim, hidden_dim, dropout=dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(scheduler_step_size),
            gamma=float(scheduler_gamma),
        )
        self.criterion = nn.MSELoss() # Regression problem
        self.baseline = 0.0
        self.grad_clip_norm = float(grad_clip_norm)
        self.best_state_dict = None
        self.best_epoch = 0
        self.best_val_acc = float("-inf")
        self.best_val_loss = float("inf")

    def _resolve_split_indices(
        self,
        labels,
        n,
        val_split=0.1,
        seed=0,
        train_idx=None,
        val_idx=None,
    ):
        labels = labels.float()
        rng = np.random.RandomState(seed)

        if train_idx is not None or val_idx is not None:
            if train_idx is None or val_idx is None:
                raise ValueError("Both train_idx and val_idx must be provided together")

            train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
            val_idx = np.asarray(val_idx, dtype=np.int64).reshape(-1)

            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError("Explicit train_idx/val_idx must both be non-empty")

            if np.intersect1d(train_idx, val_idx).size > 0:
                raise ValueError("train_idx and val_idx must be disjoint")

            all_idx = np.concatenate([train_idx, val_idx])
            if np.any(all_idx < 0) or np.any(all_idx >= n):
                raise ValueError("Found out-of-range index in train_idx/val_idx")

            train_idx = train_idx.copy()
            val_idx = val_idx.copy()
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            return train_idx, val_idx

        val_size = int(round(float(val_split) * n))
        val_size = max(1, min(val_size, n - 1))

        label_np = labels.detach().cpu().numpy().reshape(-1)
        idx_pos = np.where(label_np == 1.0)[0]
        idx_neg = np.where(label_np == 0.0)[0]

        if len(idx_pos) > 0 and len(idx_neg) > 0:
            n_val_pos = int(round(val_size * (len(idx_pos) / n)))
            n_val_neg = val_size - n_val_pos

            n_val_pos = max(1, min(n_val_pos, len(idx_pos) - 1))
            n_val_neg = max(1, min(n_val_neg, len(idx_neg) - 1))

            val_idx_pos = rng.choice(idx_pos, size=n_val_pos, replace=False)
            val_idx_neg = rng.choice(idx_neg, size=n_val_neg, replace=False)
            val_idx = np.concatenate([val_idx_pos, val_idx_neg])

            if len(val_idx) < val_size:
                remaining = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
                extra = rng.choice(remaining, size=(val_size - len(val_idx)), replace=False)
                val_idx = np.concatenate([val_idx, extra])
            elif len(val_idx) > val_size:
                val_idx = rng.choice(val_idx, size=val_size, replace=False)

            train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            return train_idx, val_idx

        perm = rng.permutation(n)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        return train_idx, val_idx

    def _evaluate_loader(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        with torch.no_grad():
            for batch_x, batch_y_centered, batch_y_raw in loader:
                all_preds = self.model(batch_x)
                final_pred = all_preds[:, -1, :]

                loss = self.criterion(final_pred, batch_y_centered)
                total_loss += float(loss.item()) * batch_x.shape[0]

                pred_label = final_pred + self.baseline
                pred_class = (pred_label >= 0.5).float()
                total_correct += int((pred_class == batch_y_raw).sum().item())
                total_count += int(batch_x.shape[0])

        avg_loss = total_loss / max(1, total_count)
        acc = total_correct / max(1, total_count)
        return avg_loss, acc

    def train(
        self,
        trajectories,
        labels,
        epochs=100,
        batch_size=32,
        val_split=0.1,
        seed=0,
        train_idx=None,
        val_idx=None,
    ):
        """
        trajectories: (N, Seq_Len, State+Action)
        labels: (N, 1) - Original binary labels (1 or 0)
        """
        # Calculate Baseline (Mean Return)
        labels = labels.float()
        self.baseline = torch.mean(labels).item()
        
        # Center the targets (Baseline Subtraction)
        centered_targets = labels - self.baseline

        n = trajectories.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 trajectories to create train/val split")

        train_idx, val_idx = self._resolve_split_indices(
            labels=labels,
            n=n,
            val_split=val_split,
            seed=seed,
            train_idx=train_idx,
            val_idx=val_idx,
        )

        train_dataset = TensorDataset(
            trajectories[train_idx],
            centered_targets[train_idx],
            labels[train_idx],
        )
        val_dataset = TensorDataset(
            trajectories[val_idx],
            centered_targets[val_idx],
            labels[val_idx],
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train/Val split: {len(train_dataset)}/{len(val_dataset)} ({(len(val_dataset)/n)*100:.1f}% val)")
        train_labels_np = labels[train_idx].detach().cpu().numpy().reshape(-1)
        val_labels_np = labels[val_idx].detach().cpu().numpy().reshape(-1)
        print(
            f"Train labels | class1={int((train_labels_np == 1.0).sum())} class0={int((train_labels_np == 0.0).sum())}"
        )
        print(
            f"Val labels   | class1={int((val_labels_np == 1.0).sum())} class0={int((val_labels_np == 0.0).sum())}"
        )

        self.best_state_dict = None
        self.best_epoch = 0
        self.best_val_acc = float("-inf")
        self.best_val_loss = float("inf")
        
        final_train_loss = float("nan")
        final_train_acc = float("nan")
        final_val_loss = float("nan")
        final_val_acc = float("nan")

        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y_centered, _batch_y_raw in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                all_preds = self.model(batch_x)
                
                # Loss is calculated ONLY against the final prediction
                final_pred = all_preds[:, -1, :]
                loss = self.criterion(final_pred, batch_y_centered)
                
                loss.backward()
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            train_loss, train_acc = self._evaluate_loader(train_loader)
            val_loss, val_acc = self._evaluate_loader(val_loader)
            final_train_loss = float(train_loss)
            final_train_acc = float(train_acc)
            final_val_loss = float(val_loss)
            final_val_acc = float(val_acc)
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6e}"
            )

            better_acc = val_acc > self.best_val_acc
            better_loss_tiebreak = (val_acc == self.best_val_acc) and (val_loss < self.best_val_loss)
            if better_acc or better_loss_tiebreak:
                self.best_val_acc = float(val_acc)
                self.best_val_loss = float(val_loss)
                self.best_epoch = int(epoch + 1)
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            self.scheduler.step()

        print(
            "Best validation checkpoint | epoch=%d val_acc=%.4f val_loss=%.6f"
            % (self.best_epoch, self.best_val_acc, self.best_val_loss)
        )
        return {
            "best_epoch": int(self.best_epoch),
            "best_val_acc": float(self.best_val_acc),
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        }

    def train_from_dataset(
        self,
        dataset,
        labels,
        epochs=100,
        batch_size=32,
        val_split=0.1,
        seed=0,
        train_idx=None,
        val_idx=None,
        num_workers=0,
        pin_memory=False,
    ):
        labels = labels.float()
        self.baseline = torch.mean(labels).item()

        centered_targets = labels - self.baseline
        n = len(dataset)
        if n < 2:
            raise ValueError("Need at least 2 trajectories to create train/val split")
        if labels.shape[0] != n:
            raise ValueError("labels length must match dataset length")

        train_idx, val_idx = self._resolve_split_indices(
            labels=labels,
            n=n,
            val_split=val_split,
            seed=seed,
            train_idx=train_idx,
            val_idx=val_idx,
        )

        train_dataset = TensorDataset(
            torch.as_tensor(train_idx, dtype=torch.int64),
            centered_targets[train_idx],
            labels[train_idx],
        )
        val_dataset = TensorDataset(
            torch.as_tensor(val_idx, dtype=torch.int64),
            centered_targets[val_idx],
            labels[val_idx],
        )

        def _collate_from_index(meta_batch):
            idx_tensor, centered_tensor, raw_tensor = zip(*meta_batch)
            sample_batch = []
            for idx_item in idx_tensor:
                sample, _ = dataset[int(idx_item.item())]
                sample_batch.append(sample)
            batch_x = torch.stack(sample_batch, dim=0)
            batch_centered = torch.stack(centered_tensor, dim=0)
            batch_raw = torch.stack(raw_tensor, dim=0)
            return batch_x, batch_centered, batch_raw

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=_collate_from_index,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=_collate_from_index,
        )

        print(f"Train/Val split: {len(train_dataset)}/{len(val_dataset)} ({(len(val_dataset)/n)*100:.1f}% val)")
        train_labels_np = labels[train_idx].detach().cpu().numpy().reshape(-1)
        val_labels_np = labels[val_idx].detach().cpu().numpy().reshape(-1)
        print(
            f"Train labels | class1={int((train_labels_np == 1.0).sum())} class0={int((train_labels_np == 0.0).sum())}"
        )
        print(
            f"Val labels   | class1={int((val_labels_np == 1.0).sum())} class0={int((val_labels_np == 0.0).sum())}"
        )

        self.best_state_dict = None
        self.best_epoch = 0
        self.best_val_acc = float("-inf")
        self.best_val_loss = float("inf")

        final_train_loss = float("nan")
        final_train_acc = float("nan")
        final_val_loss = float("nan")
        final_val_acc = float("nan")

        for epoch in range(epochs):
            self.model.train()
            for batch_x, batch_y_centered, _batch_y_raw in train_loader:
                self.optimizer.zero_grad()
                all_preds = self.model(batch_x)
                final_pred = all_preds[:, -1, :]
                loss = self.criterion(final_pred, batch_y_centered)

                loss.backward()
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            train_loss, train_acc = self._evaluate_loader(train_loader)
            val_loss, val_acc = self._evaluate_loader(val_loader)
            final_train_loss = float(train_loss)
            final_train_acc = float(train_acc)
            final_val_loss = float(val_loss)
            final_val_acc = float(val_acc)
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6e}"
            )

            better_acc = val_acc > self.best_val_acc
            better_loss_tiebreak = (val_acc == self.best_val_acc) and (val_loss < self.best_val_loss)
            if better_acc or better_loss_tiebreak:
                self.best_val_acc = float(val_acc)
                self.best_val_loss = float(val_loss)
                self.best_epoch = int(epoch + 1)
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            self.scheduler.step()

        print(
            "Best validation checkpoint | epoch=%d val_acc=%.4f val_loss=%.6f"
            % (self.best_epoch, self.best_val_acc, self.best_val_loss)
        )
        return {
            "best_epoch": int(self.best_epoch),
            "best_val_acc": float(self.best_val_acc),
            "best_val_loss": float(self.best_val_loss),
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        }


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(_repo_root(), normalized))


def _load_combined_trajectories(dataset_path: str):
    resolved = _resolve_path(dataset_path)
    with open(resolved, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "trajectories" not in payload:
        raise ValueError("Expected dataset dict with key 'trajectories'")

    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("No trajectories found in dataset")

    return trajectories, resolved


def _build_training_tensors(trajectories, cost_threshold: float, seq_len: int = 0):
    # Infer dimensions from first trajectory.
    first = trajectories[0]
    if "states" not in first or "actions" not in first or "costs" not in first:
        raise KeyError("Each trajectory must contain states, actions, and costs")

    state_dim = int(np.asarray(first["states"]).shape[-1])
    action_dim = int(np.asarray(first["actions"]).shape[-1])

    lengths = [
        min(len(t["states"]), len(t["actions"]))
        for t in trajectories
    ]
    if seq_len <= 0:
        target_len = int(max(lengths))
    else:
        target_len = int(seq_len)

    feature_dim = state_dim + action_dim
    x = np.zeros((len(trajectories), target_len, feature_dim), dtype=np.float32)
    y = np.zeros((len(trajectories), 1), dtype=np.float32)

    cumulative_costs = []
    for i, traj in enumerate(trajectories):
        states = np.asarray(traj["states"], dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        costs = np.asarray(traj["costs"], dtype=np.float32).reshape(-1)

        n = int(min(len(states), len(actions), len(costs)))
        if n <= 0:
            raise ValueError(f"Trajectory {i} is empty")

        sa = np.concatenate([states[:n], actions[:n]], axis=-1)
        use_n = min(n, target_len)
        x[i, :use_n] = sa[:use_n]

        csum = float(np.sum(costs[:n]))
        cumulative_costs.append(csum)

        # Requested labeling:
        # 1 -> cumulative cost < threshold
        # 0 -> cumulative cost >= threshold
        y[i, 0] = 1.0 if csum < cost_threshold else 0.0

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        state_dim,
        action_dim,
        np.asarray(cumulative_costs, dtype=np.float32),
        target_len,
    )


def main():
    parser = argparse.ArgumentParser(description="Train cost RUDDER on combined trajectory dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="rudder/combined_cost_balanced_1400.pkl",
        help="Path to combined trajectory dataset",
    )
    parser.add_argument(
        "--cost_threshold",
        type=float,
        default=25.0,
        help="Label threshold on cumulative cost",
    )
    parser.add_argument("--seq_len", type=int, default=0, help="Sequence length (0 means max length in data)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="rudder/models/cost_rudder_1.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trajectories, resolved_dataset = _load_combined_trajectories(args.dataset_path)
    (
        trajectories_t,
        cost_labels_t,
        state_dim,
        action_dim,
        cumulative_costs,
        target_len,
    ) = _build_training_tensors(
        trajectories=trajectories,
        cost_threshold=args.cost_threshold,
        seq_len=args.seq_len,
    )

    num_safe = int((cost_labels_t.numpy().reshape(-1) == 1.0).sum())
    num_unsafe = int((cost_labels_t.numpy().reshape(-1) == 0.0).sum())

    print("Loaded dataset:", resolved_dataset)
    print(f"Trajectories: {len(trajectories)}")
    print(f"State dim: {state_dim} | Action dim: {action_dim} | Seq len used: {target_len}")
    print(f"Cost threshold: {args.cost_threshold}")
    print("Label convention: 1 -> cost < threshold, 0 -> cost >= threshold")
    print(f"Label counts | 1 (cost < {args.cost_threshold}): {num_safe} | 0 (cost >= {args.cost_threshold}): {num_unsafe}")
    print(
        "Cumulative cost stats | min=%.3f avg=%.3f max=%.3f"
        % (float(cumulative_costs.min()), float(cumulative_costs.mean()), float(cumulative_costs.max()))
    )

    print("\n--- Training Cost RUDDER ---")
    cost_trainer = RudderTrainer(
        state_dim,
        action_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
    )
    cost_trainer.train(
        trajectories_t,
        cost_labels_t,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    save_path = _resolve_path(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": cost_trainer.model.state_dict(),
            "baseline": float(cost_trainer.baseline),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "seq_len": int(target_len),
            "cost_threshold": float(args.cost_threshold),
            "val_split": float(args.val_split),
            "dropout": float(args.dropout),
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "scheduler_step_size": int(args.scheduler_step_size),
            "scheduler_gamma": float(args.scheduler_gamma),
            "label_convention": "1 if cumulative_cost < threshold else 0",
        },
        save_path,
    )
    print("Saved trained cost RUDDER to:", save_path)


if __name__ == "__main__":
    main()