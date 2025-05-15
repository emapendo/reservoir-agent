import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

def create_sequence_dataset(n_samples_per_class=200, sequence_length=30):
    sequences = []
    labels = []

    for label in range(4):
        for _ in range(n_samples_per_class):
            pattern = generate_signal(label, sequence_length)
            sequences.append(pattern)
            labels.append(label)

    return np.array(sequences), np.array(labels)

# --- 2. Data Prep ---

X_seq, y = create_sequence_dataset()

X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.3, shuffle=True)

# --- 3. Build Reservoir and Extract Final States ---

reservoir = Reservoir(units=500, sr=0.95, lr=0.5)

def get_final_states(X_seq):
    states = []
    for seq in X_seq:
        state_seq = reservoir.run(seq)  # shape: (sequence_length, reservoir_units)
        final_state = state_seq[-1]     # take last timestep
        states.append(final_state)
    return np.array(states)

X_train_states = get_final_states(X_train)
X_test_states = get_final_states(X_test)

# --- 4. Train Logistic Regression Readout ---

clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
clf.fit(X_train_states, y_train)

# --- 5. Predict & Evaluate ---

y_pred = clf.predict(X_test_states)

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=["Random Walk", "Sine Wave", "Trend Line", "Flat Noise"],
    zero_division=0
))

# --- 6. Plot ---

plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label="True", linestyle="--")
plt.plot(y_pred[:50], label="Predicted", linestyle=":")
mismatch = y_test[:50] != y_pred[:50]
plt.scatter(np.where(mismatch), y_test[:50][mismatch], color="red", label="Misclassified", marker='x')
plt.title("RC + Logistic Regression Classification")
plt.xlabel("Sequence Index")
plt.ylabel("Class Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rc_logistic_classification_output.png")
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("rc_logistic_confusion_matrix.png")
plt.show()