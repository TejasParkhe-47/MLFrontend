from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from Data_Pipeline import Generate_JSON
import pandas as pd
import numpy as np
app = Flask(__name__)
CORS(app)

# Ensure necessary directories exist
os.makedirs("Output", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Keep model instance globally
model_obj = None
features_list = []

@app.route("/train", methods=["POST"])
def train_model():
    global model_obj, features_list
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = os.path.join("temp", file.filename)
    file.save(filepath)

    # Initialize model object
    model_obj = Generate_JSON(filepath, n_estimators=3, max_depth=10)
    X_train, X_test, y_train, y_test = model_obj.resample_split_data()

    # Train and generate JSON structure
    model_obj.to_json(X_train, y_train)

    # Evaluate
    y_pred = pd.DataFrame(model_obj.model.predict(X_test))
    accuracy = round((y_pred[0] == y_test.values).mean() * 100, 2)
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save features for prediction
    features_list = model_obj.feature_names

    return jsonify({
        "accuracy": accuracy,
        "classification_report": report,
        "features": features_list
    })


@app.route('/formatted-structure', methods=['GET'])
def formatted_structure():
    import json
    import pandas as pd
    import io

    try:
        with open("Output/structure.json", "r") as f:
            structure = json.load(f)
        df = pd.DataFrame(structure)

        formatted_lines = []
        lvl = -1

        for i in df.index:
            if lvl != df.loc[i, 'Level']:
                formatted_lines.append("*" * 40)
                formatted_lines.append(f"\tLevel : {df.loc[i, 'Level']}\n")
                formatted_lines.append("*" * 40)
            else:
                formatted_lines.append("\n")

            formatted_lines.append(f"Threshold : {df.loc[i, 'Threshold']}")
            formatted_lines.append(f"Feature : {df.loc[i, 'Feature']}\n")

            left = df.loc[i, 'Value Distribution in Left Node']
            right = df.loc[i, 'Value Distribution in Right Node']

            total_left = left["Class 0"] + left["Class 1"]
            total_right = right["Class 0"] + right["Class 1"]

            formatted_lines.append(f"Left (Values<={df.loc[i, 'Threshold']}) : {left}")
            formatted_lines.append(f"Right (Values>{df.loc[i, 'Threshold']}) : {right}\n")

            gini_left = df.loc[i, 'Gini Impurity in Left Node']
            gini_right = df.loc[i, 'Gini Impurity in Right Node']

            gini_head = round(
                (gini_left * (total_left / (total_left + total_right)))
                + (gini_right * (total_right / (total_left + total_right))),
                3,
            )

            formatted_lines.append(
                f"Gini Impurity in Left Node: 1 - [({left['Class 0']}/{total_left})² + ({left['Class 1']}/{total_left})²] = {gini_left}"
            )
            formatted_lines.append(
                f"Gini Impurity in Right Node: 1 - [({right['Class 0']}/{total_right})² + ({right['Class 1']}/{total_right})²] = {gini_right}\n"
            )

            formatted_lines.append(
                f"Gini Head: {gini_left}({total_left}/{total_left+total_right}) + {gini_right}({total_right}/{total_left+total_right}) = {gini_head}"
            )
            formatted_lines.append(
                f"Information Gain: {df.loc[i, 'Current Impurity']} - {gini_head} = {df.loc[i, 'Information Gain']}\n"
            )

            formatted_lines.append(f"Left Subtree: {df.loc[i, 'Left Subtree']}")
            formatted_lines.append(f"Right Subtree: {df.loc[i, 'Right Subtree']}\n")
            formatted_lines.append("-" * 40)

            lvl = df.loc[i, 'Level']

        formatted_output = "\n".join(formatted_lines)
        return jsonify({"formatted_structure": formatted_output})

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/structure', methods=['GET'])
def get_structure():
    try:
        with open('Output/structure.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    global model_obj, features_list
    if model_obj is None:
        return jsonify({"error": "Train the model first!"}), 400

    data = request.get_json()
    X = [[float(data[f]) for f in features_list]]

    # Predict
    pred = model_obj.model.predict(X)[0]

    # Track decision flow
    model_obj.x = 0
    model_obj.output_flow = {i+1: [] for i in range(model_obj.n_estimators)}
    model_obj.reset_flow() #this was changed by me in backend.py

    _ = model_obj.model.predict(X)

    # Convert output_flow safely
    flow_converted = {}
    for k, v in model_obj.output_flow.items():
        flow_converted[int(k)] = [
            [
                str(item[0]),
                str(item[1]),
                float(item[2]),
                int(item[3]) if isinstance(item[3], (int, float, np.integer, np.floating)) else str(item[3])
            ]
            for item in v
        ]

    print("Prediction flow:", flow_converted);
    return jsonify({
        "prediction": int(pred),
        "probabilities": [0.5, 0.5],
        "flow": flow_converted
    })




if __name__ == "__main__":
    app.run(debug=True)
