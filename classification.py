import numpy as np
from reservoirpy.nodes import Reservoir
from sklearn.linear_model import LogisticRegression

# --- Signal Generator ---
def generate_signal(label: int, length: int = 30):
    t = np.linspace(0, 1, length)
    if label == 0: return np.cumsum(np.random.randn(length)).reshape(-1, 1)
    if label == 1: return np.sin(2 * np.pi * 3 * t).reshape(-1, 1)
    if label == 2:
        direction = np.random.choice([1, -1])
        return (direction * t + 0.05 * np.random.randn(length)).reshape(-1, 1)
    if label == 3: return (0.1 * np.random.randn(length)).reshape(-1, 1)
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

def train_classification_model(n_samples_per_class=200):
    X_seq, y = create_sequence_dataset(n_samples_per_class=n_samples_per_class)
    clf_reservoir = Reservoir(units=500, sr=0.95, lr=0.5)

    def extract_final_states(X):
        final_states = []
        for seq in X:
            state_seq = clf_reservoir.run(seq)
            final_states.append(state_seq[-1])
        return np.array(final_states)

    X_states = extract_final_states(X_seq)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_states, y)

    label_map = ["Random Walk", "Sine Wave", "Trend Line", "Flat Noise"]
    return clf, clf_reservoir, label_map