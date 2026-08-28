from __future__ import annotations

import numpy as np

from neural_training_numpy import (
    binary_cross_entropy,
    forward,
    gradient_at_snapshot,
    sample_binary_cross_entropy,
    train_network,
)


def best_linear_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Busca a melhor separação linear 2D em uma grade angular densa."""
    best = 0.0
    for angle in np.linspace(0.0, np.pi, 1440, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        projections = x @ direction
        ordered = np.sort(np.unique(projections))
        thresholds = np.concatenate(
            (
                [ordered[0] - 1e-6],
                (ordered[:-1] + ordered[1:]) / 2,
                [ordered[-1] + 1e-6],
            )
        )
        for threshold in thresholds:
            prediction = (projections >= threshold).astype(float)
            best = max(
                best,
                float(np.mean(prediction == y)),
                float(np.mean((1 - prediction) == y)),
            )
    return best


def finite_difference_gradients(bundle, snapshot, epsilon: float = 1e-6):
    """Calcula gradientes independentemente por diferenças finitas centrais."""
    x_normalized = (bundle.x - bundle.mean) / bundle.scale
    w1 = snapshot.w1.copy()
    b1 = snapshot.b1.copy()
    w2 = snapshot.w2.copy()
    b2 = float(snapshot.b2)

    def current_loss(current_b2: float = b2) -> float:
        probabilities = forward(x_normalized, w1, b1, w2, current_b2)[-1]
        return binary_cross_entropy(bundle.y, probabilities)

    def differentiate_array(parameter: np.ndarray) -> np.ndarray:
        result = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = float(parameter[index])
            parameter[index] = original + epsilon
            loss_plus = current_loss()
            parameter[index] = original - epsilon
            loss_minus = current_loss()
            parameter[index] = original
            result[index] = (loss_plus - loss_minus) / (2 * epsilon)
        return result

    numerical_w1 = differentiate_array(w1)
    numerical_b1 = differentiate_array(b1)
    numerical_w2 = differentiate_array(w2)
    loss_plus = current_loss(b2 + epsilon)
    loss_minus = current_loss(b2 - epsilon)
    numerical_b2 = (loss_plus - loss_minus) / (2 * epsilon)
    return numerical_w1, numerical_b1, numerical_w2, numerical_b2


def relative_l2_error(analytical, numerical) -> float:
    analytical_array = np.asarray(analytical, dtype=float)
    numerical_array = np.asarray(numerical, dtype=float)
    denominator = max(
        1e-12,
        float(np.linalg.norm(analytical_array) + np.linalg.norm(numerical_array)),
    )
    return float(np.linalg.norm(analytical_array - numerical_array) / denominator)


def main() -> None:
    bundle = train_network()
    epochs = (0, 1, 10, 50, 100)

    for epoch in epochs:
        snapshot = bundle.snapshots[epoch]
        print(
            f"epoch={epoch:3d} loss={snapshot.loss:.6f} "
            f"training_accuracy={snapshot.accuracy:.6f}"
        )

    assert np.all(np.isfinite(bundle.loss_history)), "A perda contém valor não finito."
    assert bundle.snapshots[100].accuracy == 1.0, "A acurácia de treino final não é 100%."
    assert bundle.snapshots[100].loss < bundle.snapshots[0].loss
    print(
        "strictly_decreasing_observed="
        f"{bool(np.all(np.diff(bundle.loss_history) < 0))}"
    )

    snapshot_0 = bundle.snapshots[0]
    snapshot_1 = bundle.snapshots[1]
    analytical = gradient_at_snapshot(bundle, snapshot_0)
    numerical = finite_difference_gradients(bundle, snapshot_0)
    gradient_names = ("W1", "b1", "W2", "b2")
    relative_errors = []
    for name, analytical_gradient, numerical_gradient in zip(
        gradient_names, analytical, numerical
    ):
        error = relative_l2_error(analytical_gradient, numerical_gradient)
        relative_errors.append(error)
        print(f"finite_difference_{name}_relative_error={error:.3e}")
        assert np.allclose(
            analytical_gradient,
            numerical_gradient,
            rtol=1e-6,
            atol=1e-8,
        ), f"Gradiente analítico de {name} diverge das diferenças finitas."
    print(f"finite_difference_max_relative_error={max(relative_errors):.3e}")

    dw1 = analytical[0]
    weight_index = np.unravel_index(np.argmax(np.abs(dw1)), dw1.shape)
    calculated = snapshot_0.w1[weight_index] - bundle.learning_rate * dw1[weight_index]
    observed = snapshot_1.w1[weight_index]
    assert np.isclose(calculated, observed, atol=1e-12)
    print(
        f"weight_update={snapshot_0.w1[weight_index]:.9f}"
        f"-{bundle.learning_rate:.2f}*{dw1[weight_index]:.9f}"
        f"={observed:.9f}"
    )

    sample_index = bundle.selected_sample
    probability = snapshot_0.probabilities[sample_index]
    sample_loss = sample_binary_cross_entropy(bundle.y[sample_index], probability)
    print(
        f"sample={sample_index} probability={probability:.9f} "
        f"target={bundle.y[sample_index]:.0f} bce={sample_loss:.9f}"
    )

    linear_accuracy = best_linear_accuracy(bundle.x, bundle.y)
    print(f"best_linear_accuracy={linear_accuracy:.6f}")
    assert linear_accuracy < 1.0, "Os dados ainda são linearmente separáveis."
    print("validation=OK")


if __name__ == "__main__":
    main()
