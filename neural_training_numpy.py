from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class NetworkSnapshot:
    epoch: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    loss: float
    probabilities: np.ndarray
    accuracy: float


@dataclass(frozen=True)
class TrainingBundle:
    x: np.ndarray
    y: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    learning_rate: float
    snapshots: dict[int, NetworkSnapshot]
    loss_history: np.ndarray
    accuracy_history: np.ndarray
    selected_sample: int
    selected_hidden_unit: int


def make_two_moons(seed: int = 13, points_per_class: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Cria duas meias-luas entrelaçadas usando apenas NumPy."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.08, np.pi - 0.08, points_per_class)

    upper = np.column_stack((np.cos(theta), np.sin(theta)))
    lower = np.column_stack((1 - np.cos(theta), 0.50 - np.sin(theta)))

    raw = np.vstack((upper, lower))
    raw += rng.normal(0.0, 0.025, size=raw.shape)

    # Converte para as coordenadas exibidas pelo plot: x em [0, 6], y em [0, 4].
    x = np.empty_like(raw)
    x[:, 0] = 1.50 * (raw[:, 0] + 1.0) + 0.50
    x[:, 1] = 1.60 * (raw[:, 1] + 0.50) + 0.60
    y = np.concatenate(
        (
            np.zeros(points_per_class, dtype=float),
            np.ones(points_per_class, dtype=float),
        )
    )
    return x, y


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def forward(
    x_normalized: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = x_normalized @ w1 + b1
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ w2 + b2
    probabilities = sigmoid(z2).reshape(-1)
    return z1, a1, z2.reshape(-1), probabilities


def binary_cross_entropy(y: np.ndarray, probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(safe) + (1 - y) * np.log(1 - safe)))


def gradients(
    x_normalized: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    z1, hidden, _, probabilities = forward(x_normalized, w1, b1, w2, b2)
    count = len(x_normalized)
    dz2 = (probabilities - y).reshape(-1, 1) / count
    dw2 = hidden.T @ dz2
    db2 = float(np.sum(dz2))
    dz1 = (dz2 @ w2.T) * (z1 > 0)
    dw1 = x_normalized.T @ dz1
    db1 = np.sum(dz1, axis=0)
    return dw1, db1, dw2, db2


def _snapshot(
    epoch: int,
    x_normalized: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> NetworkSnapshot:
    _, _, _, probabilities = forward(x_normalized, w1, b1, w2, b2)
    predictions = (probabilities >= 0.5).astype(float)
    return NetworkSnapshot(
        epoch=epoch,
        w1=w1.copy(),
        b1=b1.copy(),
        w2=w2.copy(),
        b2=float(b2),
        loss=binary_cross_entropy(y, probabilities),
        probabilities=probabilities.copy(),
        accuracy=float(np.mean(predictions == y)),
    )


@lru_cache(maxsize=1)
def train_network() -> TrainingBundle:
    x, y = make_two_moons()
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    x_normalized = (x - mean) / scale

    rng = np.random.default_rng(13)
    hidden_units = 8
    w1 = rng.normal(0.0, 0.70, size=(2, hidden_units))
    b1 = np.zeros(hidden_units)
    w2 = rng.normal(0.0, 0.70, size=(hidden_units, 1))
    b2 = 0.0
    learning_rate = 0.60

    requested_epochs = {0, 1, 10, 11, 50, 100}
    snapshots: dict[int, NetworkSnapshot] = {
        0: _snapshot(0, x_normalized, y, w1, b1, w2, b2)
    }
    losses = [snapshots[0].loss]
    accuracies = [snapshots[0].accuracy]

    for epoch in range(1, 101):
        dw1, db1, dw2, db2 = gradients(
            x_normalized, y, w1, b1, w2, b2
        )
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

        current = _snapshot(epoch, x_normalized, y, w1, b1, w2, b2)
        losses.append(current.loss)
        accuracies.append(current.accuracy)
        if epoch in requested_epochs:
            snapshots[epoch] = current

    # Escolhe uma observação positiva cuja previsão inicial esteja próxima de
    # 0,32, mantendo o exemplo concreto sem fabricar um valor.
    positive_indices = np.flatnonzero(y == 1)
    initial_probabilities = snapshots[0].probabilities[positive_indices]
    selected_sample = int(
        positive_indices[np.argmin(np.abs(initial_probabilities - 0.32))]
    )

    # Para a demonstração da ativação, escolhe um neurônio relevante para a
    # saída e que não esteja completamente saturado na observação selecionada.
    snapshot_0 = snapshots[0]
    selected_normalized = x_normalized[selected_sample]
    hidden_z = selected_normalized @ snapshot_0.w1 + snapshot_0.b1
    relevance = np.abs(snapshot_0.w2.reshape(-1)) * (hidden_z > 0)
    selected_hidden_unit = int(np.argmax(relevance))

    return TrainingBundle(
        x=x,
        y=y,
        mean=mean,
        scale=scale,
        learning_rate=learning_rate,
        snapshots=snapshots,
        loss_history=np.asarray(losses),
        accuracy_history=np.asarray(accuracies),
        selected_sample=selected_sample,
        selected_hidden_unit=selected_hidden_unit,
    )


def predict(bundle: TrainingBundle, snapshot: NetworkSnapshot, x: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    x_normalized = (x_array - bundle.mean) / bundle.scale
    return forward(
        x_normalized,
        snapshot.w1,
        snapshot.b1,
        snapshot.w2,
        snapshot.b2,
    )[-1]


def hidden_state(
    bundle: TrainingBundle,
    snapshot: NetworkSnapshot,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    x_normalized = (x_array - bundle.mean) / bundle.scale
    z1, a1, _, _ = forward(
        x_normalized,
        snapshot.w1,
        snapshot.b1,
        snapshot.w2,
        snapshot.b2,
    )
    return z1, a1


def effective_hidden_coefficients(
    bundle: TrainingBundle,
    snapshot: NetworkSnapshot,
    hidden_unit: int,
) -> tuple[float, float, float]:
    weights = snapshot.w1[:, hidden_unit] / bundle.scale
    bias = snapshot.b1[hidden_unit] - np.sum(
        bundle.mean * snapshot.w1[:, hidden_unit] / bundle.scale
    )
    return float(weights[0]), float(weights[1]), float(bias)


def gradient_at_snapshot(
    bundle: TrainingBundle,
    snapshot: NetworkSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_normalized = (bundle.x - bundle.mean) / bundle.scale
    return gradients(
        x_normalized,
        bundle.y,
        snapshot.w1,
        snapshot.b1,
        snapshot.w2,
        snapshot.b2,
    )


def sample_binary_cross_entropy(target: float, probability: float) -> float:
    safe = float(np.clip(probability, 1e-9, 1 - 1e-9))
    return float(-(target * np.log(safe) + (1 - target) * np.log(1 - safe)))
