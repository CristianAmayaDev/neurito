"""Experimentos con la red de Neurito (sin interfaz gráfica).

Ejecutar con:  uv run python experimentos.py
"""

import numpy as np

ACTIONS = ("COMER", "DORMIR", "JUGAR")


def synthetic_policy(data):
    # Misma política que main.py: gana la mayor prioridad entre
    # hambre, cansancio (1 - energía) y aburrimiento.
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(values):
    return np.maximum(0, values)


def softmax(values):
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def train(hidden_size=6, epochs=1200, learning_rate=0.1, seed=42):
    """Entrena la misma red que main.py y devuelve (modelo, historial de pérdida)."""
    np.random.seed(seed)
    X = np.random.rand(3000, 3)
    targets = synthetic_policy(X)
    W1 = np.random.randn(3, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 3) * 0.5
    b2 = np.zeros((1, 3))
    history = []
    count = len(X)
    for _ in range(epochs):
        z1 = X @ W1 + b1
        hidden = relu(z1)
        probabilities = softmax(hidden @ W2 + b2)
        loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
        history.append(loss)
        delta2 = probabilities.copy()
        delta2[np.arange(count), targets] -= 1
        delta2 /= count
        grad_W2 = hidden.T @ delta2
        grad_b2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ W2.T) * (z1 > 0)
        grad_W1 = X.T @ delta1
        grad_b1 = np.sum(delta1, axis=0, keepdims=True)
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2
    model = (W1, b1, W2, b2)
    return model, history


def predict(model, data):
    W1, b1, W2, b2 = model
    return softmax(relu(data @ W1 + b1) @ W2 + b2)


def accuracy(model, seed=7):
    rng = np.random.default_rng(seed)
    test = rng.random((1000, 3))
    return np.mean(np.argmax(predict(model, test), axis=1) == synthetic_policy(test))


def fmt(probs):
    return "  ".join(f"{name}={p * 100:5.1f}%" for name, p in zip(ACTIONS, probs))


def experiment_1(model):
    print("\n=== 1. Dos necesidades fijas, variar la tercera ===")
    base = {"hambre": 0.5, "energía": 0.7, "aburrimiento": 0.2}
    print(f"Base fija: {base}")
    for index, name in enumerate(base):
        print(f"\nVariando {name} (0.0 -> 1.0):")
        previous = None
        for value in np.linspace(0, 1, 21):
            needs = list(base.values())
            needs[index] = value
            probs = predict(model, np.array([needs]))[0]
            action = ACTIONS[int(np.argmax(probs))]
            expected = ACTIONS[synthetic_policy(np.array([needs]))[0]]
            mark = "  <-- cambia" if previous and action != previous else ""
            print(f"  {name}={value:.2f}  pred={action:6s} política={expected:6s}  {fmt(probs)}{mark}")
            previous = action


def experiment_2(model):
    print("\n=== 2. Prioridades casi iguales (Softmax) ===")
    cases = [
        ("Todo empatado a 0.5", [0.5, 0.5, 0.5]),
        ("Hambre apenas mayor", [0.52, 0.5, 0.5]),
        ("Cansancio apenas mayor", [0.5, 0.48, 0.5]),
        ("Aburrimiento apenas mayor", [0.5, 0.5, 0.52]),
        ("Empate alto (0.9)", [0.9, 0.1, 0.9]),
        ("Empate bajo (0.1)", [0.1, 0.9, 0.1]),
        ("Hambre claramente mayor", [0.9, 0.5, 0.3]),
    ]
    for label, needs in cases:
        probs = predict(model, np.array([needs]))[0]
        print(f"  {label:28s} {needs}  ->  {fmt(probs)}")


def experiment_3():
    print("\n=== 3. Tamaño de la capa oculta ===")
    for size in (3, 6, 10, 20):
        model, history = train(hidden_size=size)
        print(f"  {size:2d} neuronas  pérdida final={history[-1]:.4f}  precisión={accuracy(model):.2%}")


def experiment_4():
    print("\n=== 4. Épocas y tasa de aprendizaje ===")
    configs = [
        (200, 0.1), (1200, 0.1), (5000, 0.1),
        (1200, 0.01), (1200, 0.5), (1200, 1.0), (1200, 5.0), (1200, 20.0),
    ]
    for epochs, lr in configs:
        model, history = train(epochs=epochs, learning_rate=lr)
        checkpoints = [history[i] for i in (0, len(history) // 4, len(history) // 2, -1)]
        trace = " -> ".join(f"{value:.3f}" for value in checkpoints)
        print(f"  épocas={epochs:5d} lr={lr:5.2f}  pérdida: {trace}  precisión={accuracy(model):.2%}")


def experiment_5():
    print("\n=== 5. Cuarta acción: BAÑARSE (necesidad nueva: suciedad) ===")
    np.random.seed(42)
    X = np.random.rand(3000, 4)  # hambre, energía, aburrimiento, suciedad
    targets = np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2], X[:, 3])), axis=1)
    counts = np.bincount(targets, minlength=4)
    names = ("COMER", "DORMIR", "JUGAR", "BAÑARSE")
    print("  Entradas: 4 columnas; salidas: 4 neuronas (W1: 4xN, W2: Nx4, b2: 1x4)")
    print("  Ejemplos por etiqueta: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))


def experiment_6(model):
    print("\n=== Extra: por qué la precisión puede engañar ===")
    rng = np.random.default_rng(1)
    test = rng.random((1000, 3))
    targets = synthetic_policy(test)
    preds = np.argmax(predict(model, test), axis=1)
    priorities = np.column_stack((test[:, 0], 1 - test[:, 1], test[:, 2]))
    ordered = np.sort(priorities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    close = margin < 0.05
    print(f"  Precisión global: {np.mean(preds == targets):.2%}")
    print(f"  Precisión en casos casi empatados (margen < 0.05, n={close.sum()}): "
          f"{np.mean(preds[close] == targets[close]):.2%}")
    print(f"  Precisión en casos claros (n={(~close).sum()}): "
          f"{np.mean(preds[~close] == targets[~close]):.2%}")
    always_eat = np.mean(targets == 0)
    print(f"  Un modelo que SIEMPRE dice COMER acertaría: {always_eat:.2%}")


if __name__ == "__main__":
    base_model, base_history = train()
    print(f"Modelo base (6 neuronas, 1200 épocas, lr=0.1): pérdida={base_history[-1]:.4f} "
          f"precisión={accuracy(base_model):.2%}")
    experiment_1(base_model)
    experiment_2(base_model)
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6(base_model)
