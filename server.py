from flask import Flask, request, jsonify, render_template
import numpy as np
from regression import esn
from classification import train_classification_model

# --- Flask setup ---
app = Flask(__name__)

clf, clf_reservoir, label_map = train_classification_model()
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

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        sequence = data["sequence"]
        if not sequence or len(sequence) != 30:
            return jsonify({"error": "Input must be a list of 30 values."}), 400

        seq_array = np.array(sequence).reshape(-1, 1)
        state_seq = clf_reservoir.run(seq_array)
        final_state = state_seq[-1].reshape(1, -1)
        prediction = int(clf.predict(final_state)[0])

        return jsonify({"class": prediction, "label": label_map[prediction]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)