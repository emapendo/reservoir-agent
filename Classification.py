import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import set_seed, verbosity
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
import time
set_seed(int(time.time()))  # Seed based on the actual time
#set_seed(42)  # Fixed seed for reproducibility
verbosity(0)

# --- Functions ---
def generate_pattern(label, length):
    t = np.linspace(0, 2 * np.pi, length)
    if label == 0:
        return np.sin(t)          # Sinus
    elif label == 1:
        return np.sign(np.sin(t))  # Rectangle
    elif label == 2:
        return 2 * (t / (2 * np.pi)) - 1  # Triangle
    else:
        raise ValueError("Unknown Label.")

# --- 1. Generate data ---
n_patterns = 100
pattern_length = 20
n_classes = 3

X_list = []
y_list = []

for _ in range(n_patterns):
    label = np.random.randint(0, n_classes)
    pattern = generate_pattern(label, pattern_length).reshape(-1, 1)
    X_list.append(pattern)
    y_list.extend([label] * pattern_length)

X = np.vstack(X_list)
y = np.array(y_list)

# --- 2. Training-/Testdata ---
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- 3. Define model ---
reservoir = Reservoir(100, lr=0.7, sr=0.95)
readout = Ridge(ridge=1e-6)
esn_model = reservoir >> readout

# --- 4. One-Hot-Encoding for classification ---
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))

# --- 5. Training ---
esn_model = esn_model.fit(X_train, y_train_oh, warmup=10, reset=True)

# --- 6. Prediction ---
y_pred_oh = esn_model.run(X_test)
y_pred = np.argmax(y_pred_oh, axis=1)

# --- 7. Plot ---
plt.figure(figsize=(10, 4))
plt.title("Example input signals with associated class labels")
plt.plot(X_test[:200], label="Input-Signal")
plt.plot(y_test[:200], label="Ground Truth", linestyle="--")
plt.plot(y_pred[:200], label="Prediction", linestyle=":")
plt.xlabel("Time")
plt.ylabel("value / class")
plt.legend()
plt.tight_layout()
plt.show()