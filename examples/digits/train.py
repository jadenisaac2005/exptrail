"""Train a tiny NumPy MLP on sklearn's digits with 3 optimiser configs.

    pip install exptrail numpy scikit-learn
    python examples/digits/train.py

Each config becomes a run under examples/digits/runs/, and the results
table in examples/digits/README.md is regenerated from those runs.
"""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from exptrail import Run
from exptrail.cli import main as exptrail_cli

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs"

CONFIGS = [
    {"name": "sgd-lr0.1", "lr": 0.1, "momentum": 0.0},
    {"name": "momentum-lr0.01", "lr": 0.01, "momentum": 0.9},
    {"name": "momentum-lr0.05", "lr": 0.05, "momentum": 0.9},
]
COMMON = {"hidden": 64, "epochs": 30, "batch_size": 32, "weight_decay": 1e-4, "seed": 0}


def load_data(seed):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=seed, stratify=y_tmp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def init_params(rng, n_in, hidden, n_out):
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def forward(p, X):
    h = np.maximum(0, X @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return h, probs


def loss_and_grads(p, X, y, wd):
    h, probs = forward(p, X)
    n = len(y)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    d = probs.copy()
    d[np.arange(n), y] -= 1
    d /= n
    grads = {"W2": h.T @ d + wd * p["W2"], "b2": d.sum(0)}
    dh = (d @ p["W2"].T) * (h > 0)
    grads.update(W1=X.T @ dh + wd * p["W1"], b1=dh.sum(0))
    return loss, grads


def accuracy(p, X, y):
    return float((forward(p, X)[1].argmax(1) == y).mean())


def train(cfg):
    config = {**COMMON, **{k: v for k, v in cfg.items() if k != "name"}}
    with Run(cfg["name"], config=config, tags=["digits", "numpy-mlp"],
             seeds={"numpy": config["seed"], "split": config["seed"]}, root=ROOT) as run:
        rng = np.random.default_rng(config["seed"])
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(config["seed"])
        params = init_params(rng, X_train.shape[1], config["hidden"], 10)
        velocity = {k: np.zeros_like(v) for k, v in params.items()}

        for epoch in range(config["epochs"]):
            order = rng.permutation(len(X_train))
            losses = []
            for i in range(0, len(order), config["batch_size"]):
                idx = order[i : i + config["batch_size"]]
                loss, grads = loss_and_grads(params, X_train[idx], y_train[idx], config["weight_decay"])
                losses.append(loss)
                for k in params:
                    velocity[k] = config["momentum"] * velocity[k] - config["lr"] * grads[k]
                    params[k] += velocity[k]
            val_loss, _ = loss_and_grads(params, X_val, y_val, 0.0)
            run.log(step=epoch, train_loss=np.mean(losses), val_loss=val_loss,
                    val_acc=accuracy(params, X_val, y_val))

        run.summary(test_acc=accuracy(params, X_test, y_test),
                    val_acc=accuracy(params, X_val, y_val))
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "model.npz", **params)
            run.save_artifact(Path(tmp) / "model.npz")
    print(f"{cfg['name']:>16}: test_acc={run._summary['test_acc']:.4f}  -> {run.dir.name}")
    return run


if __name__ == "__main__":
    runs = [train(cfg) for cfg in CONFIGS]
    exptrail_cli([
        "--root", str(ROOT), "table",
        "--metric", "test_acc", "--metric", "val_acc",
        "--config", "lr", "--config", "momentum",
        "--runs", *[r.dir.name for r in runs],
        "--readme", str(HERE / "README.md"),
    ])
