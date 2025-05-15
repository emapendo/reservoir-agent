import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from random import seed

# --- 1. Signal Generators ---
def generate_signal(label: int, length: int = 30):
    t = np.linspace(0, 1, length)

    if label == 0:  # Random Walk
        return np.cumsum(np.random.randn(length)).reshape(-1, 1)
    elif label == 1:  # Sine Wave
        return np.sin(2 * np.pi * 3 * t).reshape(-1, 1)
    elif label == 2:  # Trend Line
        direction = np.random.choice([1, -1])
        return (direction * t + 0.05 * np.random.randn(length)).reshape(-1, 1)
    elif label == 3:  # Flat Noise
        return (0.1 * np.random.randn(length)).reshape(-1, 1)
    else:
        raise ValueError("Invalid label.")

def create_balanced_classification_dataset(n_samples_per_class=100, sequence_length=30):
    X = []
    y = []

    for label in range(4):
        for _ in range(n_samples_per_class):
            pattern = generate_signal(label, sequence_length)
            X.append(pattern)
            y.extend([label] * sequence_length)  # one label per time step

    X = np.vstack(X)  # shape: (num_samples * seq_len, 1)
    y = np.array(y)   # shape: (num_samples * seq_len,)
    return X, y

# --- 2. Data Prep ---

X, y = create_balanced_classification_dataset()

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_oh = encoder.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.3, shuffle=True)

# --- 3. Train ESN Classifier ---

reservoir = Reservoir(units=300, sr=0.9, lr=0.3)
readout = Ridge(ridge=1e-4)
esn = reservoir >> readout

esn = esn.fit(X_train, y_train, warmup=10, reset=True)

# --- 4. Predict ---

y_pred_oh = esn.run(X_test)
y_pred = np.argmax(y_pred_oh, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- 5. Evaluate ---

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=["Random Walk", "Sine Wave", "Trend Line", "Flat Noise"],
    zero_division=0
))

# --- 6. Plot ---

plt.figure(figsize=(10, 5))
plt.plot(y_true[:300], label="True", linestyle="--")
plt.plot(y_pred[:300], label="Predicted", linestyle=":")
plt.title("RC Classification Output")
plt.xlabel("Timestep")
plt.ylabel("Class Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rc_classification_output.png")
plt.show()