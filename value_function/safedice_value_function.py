#!/usr/bin/env python
"""
Standalone SafeDICE value-function loader.

This module lets you load a SafeDICE checkpoint and use the critic as a
callable state-value function:

    vf = load_safedice_value_function(weights_path)
    values = vf(states)  # states: [N, state_dim] or [state_dim]

Requirements to make this portable:
1) SafeDICE checkpoint pickle file containing training_state.
2) SafeDICE code directory with:
   - algorithms/safedice.py
   - algorithms/utils.py
   - optionally config/safedice_config.py
3) Python environment with TensorFlow stack compatible with SafeDICE.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


DEFAULT_CONFIG: Dict[str, Any] = {
    "hidden_size": 256,
    "critic_lr": 1e-4,
    "actor_lr": 1e-5,
    "gamma": 0.99,
    "alpha": 0.0,
    "grad_reg_coeffs": (10, 1e-6),
    "use_last_layer_bias_cost": False,
    "use_last_layer_bias_critic": False,
    "kernel_initializer": "he_normal",
}


def _import_antidice(safedice_root: Path):
    if not safedice_root.exists():
        raise FileNotFoundError(f"SafeDICE root not found: {safedice_root}")

    if str(safedice_root) not in sys.path:
        sys.path.insert(0, str(safedice_root))

    try:
        from algorithms.safedice import SafeDICE as AntiDICE  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Failed to import SafeDICE class from algorithms.safedice. "
            f"Check SafeDICE root path and dependencies. Root: {safedice_root}"
        ) from exc

    return AntiDICE


def _load_pickle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Checkpoint payload must be dict, got {type(data)}")
    if "training_state" not in data:
        raise KeyError("Checkpoint missing top-level key 'training_state'")
    return data


def _find_first_kernel_input_dim(params) -> Optional[int]:
    for name, value in params:
        if not isinstance(name, str):
            continue
        if name.endswith("mlp/dense/kernel:0") or "mlp/dense/kernel" in name:
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])
    return None


def _infer_dims_from_training_state(training_state: Dict[str, Any]) -> Tuple[int, int]:
    critic_params = training_state.get("critic_params", [])
    cost_params = training_state.get("cost_params", [])

    if not critic_params:
        raise ValueError("training_state missing non-empty 'critic_params'")
    if not cost_params:
        raise ValueError(
            "training_state missing non-empty 'cost_params'. "
            "Action dim is inferred from cost input dim - state dim."
        )

    state_dim = _find_first_kernel_input_dim(critic_params)
    cost_input_dim = _find_first_kernel_input_dim(cost_params)
    if state_dim is None:
        raise ValueError("Could not infer state_dim from critic_params")
    if cost_input_dim is None:
        raise ValueError("Could not infer cost input dim from cost_params")

    action_dim = int(cost_input_dim - state_dim)
    if action_dim <= 0:
        raise ValueError(
            f"Invalid inferred action_dim={action_dim} from cost_input_dim={cost_input_dim}, "
            f"state_dim={state_dim}"
        )

    return state_dim, action_dim


def _resolve_config(safedice_root: Path, config_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)

    config_dir = safedice_root / "config"
    if config_dir.exists() and str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))

    try:
        from safedice_config import hparams  # type: ignore

        if hparams:
            cfg.update(hparams[0])
    except Exception:
        pass

    if config_override:
        cfg.update(config_override)

    return cfg


class SafeDICEValueFunction:
    def __init__(
        self,
        weights_path: str,
        safedice_root: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.weights_path = Path(weights_path).expanduser().resolve()
        if safedice_root is None:
            self.safedice_root = (Path(__file__).resolve().parent.parent / "SafeDICE").resolve()
        else:
            self.safedice_root = Path(safedice_root).expanduser().resolve()

        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required to load SafeDICE critic. "
                "Use the SafeDICE environment or install compatible TF dependencies."
            ) from exc

        self._AntiDICE = _import_antidice(self.safedice_root)
        payload = _load_pickle(self.weights_path)
        training_state = payload["training_state"]

        self.state_dim, self.action_dim = _infer_dims_from_training_state(training_state)
        self.config = _resolve_config(self.safedice_root, config_override)

        self.model = self._AntiDICE(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mixture_actor=False,
            is_discrete_action=False,
            config=self.config,
        )
        self.model.set_training_state(training_state)

    def __call__(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return self.value(states=states, batch_size=batch_size)

    def value(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        import tensorflow as tf

        arr = np.asarray(states, dtype=np.float32)
        squeeze_back = False
        if arr.ndim == 1:
            arr = arr[None, :]
            squeeze_back = True

        if arr.ndim != 2:
            raise ValueError(f"Expected states shape [N, state_dim] or [state_dim], got {arr.shape}")
        if arr.shape[1] != self.state_dim:
            raise ValueError(f"State dim mismatch: input has {arr.shape[1]}, expected {self.state_dim}")

        outputs = []
        for start in range(0, arr.shape[0], batch_size):
            end = min(start + batch_size, arr.shape[0])
            batch = tf.convert_to_tensor(arr[start:end], dtype=tf.float32)
            v_batch, _ = self.model.critic(batch)
            outputs.append(v_batch.numpy().reshape(-1))

        values = np.concatenate(outputs, axis=0)
        if squeeze_back:
            return np.asarray(values[0], dtype=np.float32)
        return values.astype(np.float32)


def load_safedice_value_function(
    weights_path: str,
    safedice_root: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> SafeDICEValueFunction:
    return SafeDICEValueFunction(
        weights_path=weights_path,
        safedice_root=safedice_root,
        config_override=config_override,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load SafeDICE critic and evaluate value(s) for input state(s)")
    parser.add_argument("--weights", type=str, required=True, help="Path to SafeDICE weights pickle")
    parser.add_argument(
        "--safedice-root",
        type=str,
        default="",
        help="Path to SafeDICE root directory (contains algorithms/ and config/). "
        "Default: <repo>/SafeDICE",
    )
    parser.add_argument(
        "--state-npy",
        type=str,
        default="",
        help="Optional .npy file containing [state_dim] or [N, state_dim]",
    )
    parser.add_argument(
        "--random-batch",
        type=int,
        default=0,
        help="If >0 and --state-npy is not set, evaluate this many random states",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    safedice_root = args.safedice_root.strip() or None
    vf = load_safedice_value_function(weights_path=args.weights, safedice_root=safedice_root)

    print(f"Loaded SafeDICE value function from: {args.weights}")
    print(f"state_dim={vf.state_dim}, action_dim={vf.action_dim}")

    if args.state_npy:
        states = np.load(args.state_npy)
    else:
        n = int(args.random_batch)
        if n <= 0:
            n = 1
        states = np.random.randn(n, vf.state_dim).astype(np.float32)

    values = vf(states)
    print("Input states shape:", np.asarray(states).shape)
    print("Output values shape:", np.asarray(values).shape)
    print("Values sample:", np.asarray(values).reshape(-1)[:10])


if __name__ == "__main__":
    main()
