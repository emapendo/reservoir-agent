from flask import Flask, request, jsonify, render_template
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_points = np.array(data["input"]).reshape(-1, 1)  # allow multiple inputs

    outputs = esn.run(input_points)
    output_values = outputs.flatten().tolist()

    # Agent decisions for each output
    decisions = []
    for value in output_values:
        if value > 0.05:
            decisions.append("Accelerate")
        elif value < -0.05:
            decisions.append("Decelerate")
        else:
            decisions.append("Maintain Speed")

    return jsonify({"predictions": output_values, "decisions": decisions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)