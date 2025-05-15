import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Build the ESN
reservoir = Reservoir(units=500, sr=0.9, lr=0.3)
readout = Ridge(ridge=1e-4)
esn = reservoir >> readout

# Pre-train on fake data
print("Training ESN on fake data...")
X = np.cumsum(np.random.randn(100)).reshape(-1, 1)
Y = np.roll(X, -1)

scaler = MinMaxScaler(feature_range=(-1, 1))
X, Y = scaler.fit_transform(X), scaler.transform(Y)

X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.3, shuffle=False)
esn = esn.fit(X_train, Y_train)

print("ESN training complete. Ready to accept input!")