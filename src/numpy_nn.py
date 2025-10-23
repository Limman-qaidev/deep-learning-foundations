from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Tuple,
    List,
    Mapping,
    Optional
)

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


Array = NDArray[np.float64]
Activation = Callable[[Array], Array]
ActivationPrime = Callable[[Array], Array]


def sigmoid(x: Array) -> Array:
    """The sigmoid activation function.
    Element-wise $\sigma(z) = \frac{1}{1 + e^{-z}}$.
    Args:
        x: The input array.
    Returns:
        The output array with the sigmoid function applied
         element-wise.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Array) -> Array:
    """The derivative of the sigmoid activation function.
    Element-wise $\sigma'(z) = \sigma(z) * (1 - \sigma(z))$.
    Args:
        x: The input array.
    Returns:
        The output array with the derivative of the sigmoid
         function applied element-wise.
    """
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: Array) -> Array:
    """The tanh activation function.
    Element-wise $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$.
    Args:
        x: The input array.
    Returns:
        The output array with the tanh function applied
         element-wise.
    """
    return np.tanh(x)


def tanh_prime(x: Array) -> Array:
    """The derivative of the tanh activation function.
    Element-wise $\tanh'(z) = 1 - \tanh^2(z)$.
    Args:
        x: The input array.
    Returns:
        The output array with the derivative of the tanh
         function applied element-wise.
    """
    return 1 - np.tanh(x) ** 2


def relu(x: Array) -> Array:
    """The ReLU activation function.
    Element-wise $\text{ReLU}(z)=max(0, z)$.
    Args:
        x: The input array.
    Returns:
        The output array with the ReLU function applied
         element-wise.
    """
    return np.maximum(0, x)


def relu_prime(x: Array) -> Array:
    """The derivative of the ReLU activation function.
    Element-wise $\text{ReLU}'(z) = 1$ if $z > 0$ else $0$.
    Args:
        x: The input array.
    Returns:
        The output array with the derivative of the ReLU
         function applied element-wise.
    """
    return np.where(x > 0, 1, 0)


@dataclass
class LLayerParams:
    """Parameters of an L-layer fully-connected neural
      network.

    Stored as dicts keyed by layer number (1..L).
    W[l] ∈ R^{n_l × n_{l-1}}, b[l] ∈ R^{n_l × 1}
      """
    W: Dict[int, Array]
    b: Dict[int, Array]


def forward_propagation(
    X: Array,
    params: LLayerParams,
    activations: Mapping[int, Activation],
    keep_probs: Optional[Mapping[int, float]] = None,
    training: bool = False,
) -> Tuple[Array, Dict[str, Dict[int, Array]]]:
    """Forward pass with optional inverted dropout per hidden layer.

    Args:
        X: Input (n_0, m).
        params: W,b dicts.
        activations: g_l per layer l.
        keep_probs: Optional dict l -> p_l in (0,1], for dropout on A[l].
                    If None, no dropout. Typically only for hidden layers.
        training: If True, apply dropout; otherwise disabled (inference).

    Returns:
        A_L: Output activations A[L].
        cache: {"A":..., "Z":..., "D":... (dropout masks if any)}
    """
    L = len(params.W)
    A: Dict[int, Array] = {0: X}
    Z: Dict[int, Array] = {}
    D: Dict[int, Array] = {}

    use_do = training and (keep_probs is not None)

    for layer in range(1, L + 1):
        Wl, bl = params.W[layer], params.b[layer]
        Zl = Wl @ A[layer - 1] + bl
        gl = activations[layer]
        Al = gl(Zl)

        # Apply inverted dropout only if requested and p_l < 1
        if use_do and (layer in keep_probs):
            p_l = float(keep_probs[layer])
            if not (0.0 < p_l <= 1.0):
                raise ValueError(
                    f"keep_prob for layer {layer} must be in (0,1]."
                    )
            if p_l < 1.0:
                mask = (
                    np.random.random_sample(size=Al.shape) < p_l
                    ).astype(np.float64)
                Al = (Al * mask) / p_l
                D[layer] = mask  # store mask
        Z[layer], A[layer] = Zl, Al

    cache = {"A": A, "Z": Z, "D": D}
    return A[L], cache


def compute_cost(
    A_L: Array,
    Y: Array,
    params: Optional[LLayerParams] = None,
    l2_lambda: float = 0.0,
    l2_m: Optional[int] = None,  # <-- NUEVO
) -> float:
    m = Y.shape[1]
    diff = A_L - Y
    data_cost = float((0.5 / m) * np.sum(diff * diff))

    if params is None or l2_lambda <= 0.0:
        return data_cost

    l2_sum = sum(np.sum(Wl * Wl) for Wl in params.W.values())
    denom = float(l2_m if l2_m is not None else m)  # <-- usa m_total si viene
    reg_cost = (l2_lambda / (2.0 * denom)) * l2_sum
    return data_cost + float(reg_cost)


def backward_propagation(
    Y: Array,
    params: LLayerParams,
    cache: Dict[str, Dict[int, Array]],
    activations_prime: Mapping[int, ActivationPrime],
    l2_lambda: float = 0.0,
    keep_probs: Optional[Mapping[int, float]] = None,
    l2_m: Optional[int] = None,
) -> Dict[str, Dict[int, Array]]:
    """Vectorized backprop with quadratic cost, optional L2 and dropout.

    δ[L] = (A[L]-Y) ⊙ g'_L(Z[L])
    δ[l] = (W[l+1]^T δ[l+1]) ⊙ g'_l(Z[l])   for l=L-1..1
    Dropout (inverted): δ[l] ← (δ[l] ⊙ D[l]) / p_l   if used on A[l]
    L2: dW[l] += (λ/m_total) W[l]  (si l2_m=m_total)
    """
    A: Dict[int, Array] = cache["A"]
    Z: Dict[int, Array] = cache["Z"]
    D: Dict[int, Array] = cache.get("D", {})
    L = len(params.W)
    m = Y.shape[1]

    # 1) Error en la capa de salida
    delta: Dict[int, Array] = {}
    gL_prime = activations_prime[L]
    delta[L] = (A[L] - Y) * gL_prime(Z[L])

    # 2) Errores en capas ocultas (L-1 .. 1)
    for layer in range(L - 1, 0, -1):
        g_prime = activations_prime[layer]
        delta[layer] = (
            params.W[layer + 1].T @ delta[layer + 1]) * g_prime(Z[layer]
                                                                )

        # Dropout (si se aplicó en A[layer])
        if keep_probs is not None and (layer in keep_probs):
            p_l = float(keep_probs[layer])
            if p_l < 1.0:
                Dl = D.get(layer)
                if Dl is None:
                    raise RuntimeError(
                        f"Missing dropout mask D[{layer}] in cache."
                        )
                delta[layer] = (delta[layer] * Dl) / p_l

    # 3) Gradientes
    dW: Dict[int, Array] = {}
    db: Dict[int, Array] = {}
    for layer in range(1, L + 1):
        dW[layer] = (delta[layer] @ A[layer - 1].T) / m
        if l2_lambda > 0.0:
            denom = float(l2_m if l2_m is not None else m)
            dW[layer] += (l2_lambda / denom) * params.W[layer]
        db[layer] = np.mean(delta[layer], axis=1, keepdims=True)

    return {"dW": dW, "db": db}


def update_parameters(
        params: LLayerParams,
        grads: Dict[str, Dict[int, Array]],
        lr: float,
) -> LLayerParams:
    """One gradient-descent step on all layers.

    Args:
        params: Current parameters.
        grads: Dicts with dW[l], db[l] for l=1..L.
        lr: Learning rate η.

    Returns:
        Updated LLayerParams.
    """
    W_new: Dict[int, Array] = {}
    b_new: Dict[int, Array] = {}

    for layer in range(1, len(params.W) + 1):
        W_new[layer] = params.W[layer] - lr * grads["dW"][layer]
        b_new[layer] = params.b[layer] - lr * grads["db"][layer]

    return LLayerParams(W=W_new, b=b_new)


def iter_minibatches(
    X: Array,
    Y: Array,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    """Yield (X_batch, Y_batch) por columnas."""
    m = X.shape[1]
    idx = np.arange(m)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, m, batch_size):
        batch = idx[start:start + batch_size]
        yield X[:, batch], Y[:, batch]


def xavier_init(
        layer_sizes: List[int],
        rng: np.random.Generator
) -> LLayerParams:
    """Xavier/Glorot initialization (good for tanh).

    layer_sizes: [n_0, n_1, ..., n_L]
    """
    W: Dict[int, Array] = {}
    b: Dict[int, Array] = {}
    for layer in range(1, len(layer_sizes)):
        n_l, n_prev = layer_sizes[layer], layer_sizes[layer - 1]
        limit = np.sqrt(6.0 / (n_prev + n_l))
        W[layer] = rng.uniform(
            -limit, limit, size=(n_l, n_prev)
            ).astype(np.float64)
        b[layer] = np.zeros((n_l, 1), dtype=np.float64)
    return LLayerParams(W=W, b=b)


def he_init(layer_sizes: List[int], rng: np.random.Generator) -> LLayerParams:
    """He initialization (good for ReLU)."""
    W: Dict[int, Array] = {}
    b: Dict[int, Array] = {}
    for layer in range(1, len(layer_sizes)):
        n_l, n_prev = layer_sizes[layer], layer_sizes[layer - 1]
        std = np.sqrt(2.0 / n_prev)
        W[layer] = (
            rng.normal(0.0, std, size=(n_l, n_prev))
            ).astype(np.float64)
        b[layer] = np.zeros((n_l, 1), dtype=np.float64)
    return LLayerParams(W=W, b=b)


# ---- utilities ----

def predict(
        A_L: Array,
        threshold: float = 0.5
) -> Array:
    """Predict class labels based on output activations.

    Args:
        A_L: Output activations, shape (n_L, m).
        threshold: Decision boundary for classifying outputs.

    Returns:
        Predicted class labels, shape (1, m).
    """
    return (A_L > threshold).astype(np.float64)


def accuracy(
        y_pred: Array,
        y_true: Array
) -> float:
    """Compute accuracy of predictions.

    Args:
        y_pred: Predicted class labels, shape (1, m).
        y_true: True class labels, shape (1, m).

    Returns:
        Accuracy as a float.
    """
    return np.mean(y_pred == y_true)


def plot_decision_boundary(
    f_decision: Callable[[Array], Array],
    X: Array,
    Y: Array,
    padding: float = 0.5,
    h: float = 0.01,
) -> None:
    """Plot decision boundary of a classifier.

    Args:
        func: Function that takes input X and returns
         predicted class labels.
        X: Input data, shape (2, m).
        y: True class labels, shape (1, m).
        padding: Extra space around the data in the plot.
        h: Step size for mesh grid.
    """
    x_min = X[0, :].min() - padding
    x_max = X[0, :].max() + padding
    y_min = X[1, :].min() - padding
    y_max = X[1, :].max() + padding
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
    grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, N)

    Z = f_decision(grid)  # (1, N) predicciones {0,1}
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.5)  # sin especificar colores
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), edgecolor="k")
    plt.title("Decision boundary on make_moons")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


def main() -> None:
    rng = np.random.default_rng(seed=42)

    # === DATA GENERATION AND PREPROCESSING ===
    X, Y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, Y, test_size=0.3, random_state=1
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr = X_tr.T.astype(np.float64)
    X_te = X_te.T.astype(np.float64)
    y_tr = y_tr.reshape(1, -1).astype(np.float64)
    y_te = y_te.reshape(1, -1).astype(np.float64)

    # === REGULARIZATION HYPERPARAMETERS ===
    l2_lambda = 1e-3  # set to 0.0 to disable L2
    keep_probs: Dict[int, float] = {
        1: 0.9,  # dropout prob for hidden layer 1
        2: 0.9,  # dropout prob for hidden layer 2
    }
    use_dropout = True  # toggle dropout usage

    # === NETWORK ARCHITECTURE ===
    layer_sizes = [2, 16, 16, 1]
    params = xavier_init(layer_sizes, rng)
    L = len(layer_sizes) - 1
    activations: Dict[int, Activation] = {
        layer: (tanh if layer < L else sigmoid)
        for layer in range(1, L + 1)
    }
    activation_primes: Dict[int, ActivationPrime] = {
        layer: (tanh_prime if layer < L else sigmoid_prime)
        for layer in range(1, L + 1)
    }

    batch_size: Optional[int] = 64
    m_train = y_tr.shape[1]

    # === TRAINING LOOP ===
    n_epochs = 10000
    lr = 0.05

    for epoch in range(1, n_epochs + 1):
        if batch_size is None:
            # ----- FULL-BATCH (igual que antes) -----
            A_L, cache = forward_propagation(
                X_tr, params, activations,
                keep_probs=(keep_probs if use_dropout else None),
                training=True
            )
            cost = compute_cost(
                A_L, y_tr, params=params,
                l2_lambda=l2_lambda, l2_m=m_train
            )
            grads = backward_propagation(
                y_tr, params, cache, activation_primes,
                l2_lambda=l2_lambda,
                keep_probs=(keep_probs if use_dropout else None),
                l2_m=m_train
            )
            params = update_parameters(params, grads, lr)
        else:
            # ----- MINI-BATCH / SGD -----
            epoch_cost = 0.0
            n_batches = 0
            for X_b, Y_b in iter_minibatches(
                X_tr, y_tr, batch_size, rng, shuffle=True
            ):
                A_b, cache_b = forward_propagation(
                    X_b, params, activations,
                    keep_probs=(keep_probs if use_dropout else None),
                    training=True
                )
                batch_cost = compute_cost(
                    A_b, Y_b, params=params,
                    l2_lambda=l2_lambda, l2_m=m_train
                )
                grads_b = backward_propagation(
                    Y_b, params, cache_b, activation_primes,
                    l2_lambda=l2_lambda,
                    keep_probs=(keep_probs if use_dropout else None),
                    l2_m=m_train
                )
                params = update_parameters(params, grads_b, lr)
                epoch_cost += batch_cost
                n_batches += 1
            cost = epoch_cost / max(n_batches, 1)

        # --- métricas periódicas ---
        if epoch % 1000 == 0 or epoch == 1:
            # forward completo SIN dropout para medir en train
            A_tr_eval, _ = forward_propagation(
                X_tr, params, activations, keep_probs=None, training=False
            )
            acc_tr = accuracy(predict(A_tr_eval), y_tr)

            A_L_te, _ = forward_propagation(
                X_te, params, activations, keep_probs=None, training=False
            )
            acc_te = accuracy(predict(A_L_te), y_te)

            print(f"Epoch {epoch:5d}: Cost {cost:.4f}, "
                  f"Train Acc {acc_tr:.4f}, Test Acc {acc_te:.4f}")

    # === FINAL EVALUATION (NO DROPOUT) ===
    A_L_tr, _ = forward_propagation(
        X_tr, params, activations, keep_probs=None, training=False
    )
    A_L_te, _ = forward_propagation(
        X_te, params, activations, keep_probs=None, training=False
    )
    print("Final Train Acc:", accuracy(predict(A_L_tr), y_tr))
    print("Final Test Acc:", accuracy(predict(A_L_te), y_te))

    # === DECISION BOUNDARY ===
    def f_decision(X_input: Array) -> Array:
        A_out, _ = forward_propagation(
            X_input, params, activations, keep_probs=None, training=False
        )
        return predict(A_out)

    plot_decision_boundary(f_decision, X_tr, y_tr)


if __name__ == "__main__":
    main()
