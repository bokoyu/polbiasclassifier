from flask import Flask, request, jsonify
from predict import predict
from train import train_model
from evaluate import evaluate_model
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_data"  # or "temp_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploadfile', methods=['POST'])
def upload_file():
    """
    Expects multipart/form-data with "data_file" as the name of the file field.
    Saves the file to a server folder and returns its saved path.
    """
    if 'data_file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['data_file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400


    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    return jsonify({"saved_path": save_path})

@app.route('/')
def hello():
    return "Backend is running. Use POST /predict, /train, /evaluate, etc."

@app.route('/predict', methods=['POST'])
def predict_route():

    data = request.get_json()
    input_text = data.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    result = predict(input_text) 
    return jsonify({"prediction": result})

@app.route('/train', methods=['POST'])
def train_route():

    data = request.get_json()
    data_path = data.get("data_path")
    epochs = data.get("epochs", 3)
    cleaning = data.get("cleaning", False)
    batch_size = data.get("batch_size", 8)
    lr_bias = data.get("lr_bias", 3e-5)
    lr_lean = data.get("lr_lean", 2e-5)

    if not data_path or not os.path.exists(data_path):
        return jsonify({"error": f"Data file does not exist: {data_path}"}), 400

    train_model(
        data_path=data_path,
        do_cleaning=cleaning,
        epochs=epochs,
        batch_size=batch_size,
        overwrite=False,
        lr_bias=lr_bias,
        lr_lean=lr_lean
    )
    return jsonify({"message": "Training started (or completed) successfully."})

@app.route('/evaluate', methods=['POST'])
def evaluate_route():
    data = request.get_json()
    data_path = data.get("data_path")
    cleaning = data.get("cleaning", False)
    
    # Run the evaluation
    results = evaluate_model(
        data_path=data_path,
        do_cleaning=cleaning,
        cleaning_func=None,  # or your custom cleaning function
        batch_size=8,
        verbose=False
    )
    # results is now a dict:
    # {
    #   "bias_metrics": { ... },
    #   "leaning_metrics": { ... }
    # }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
