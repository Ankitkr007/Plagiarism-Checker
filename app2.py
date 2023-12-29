import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from flask import Flask, render_template, jsonify
from flask_caching import Cache
app = Flask(__name__)


# Configure Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
# Load data from JSON file
with open("y_data.json", "r") as file:
    data = json.load(file)

y_test = data["y_test"]
y_pred = data["y_pred"]

def calculate_metrics(y_test, y_pred):
    accuracy = int((accuracy_score(y_test, y_pred))*100)
    precision_class_0 = int((precision_score(y_test, y_pred, pos_label=0))*100)
    recall_class_0 = int((recall_score(y_test, y_pred, pos_label=0))*100)
    precision_class_1 = int((precision_score(y_test, y_pred, pos_label=1))*100)
    recall_class_1 = int((recall_score(y_test, y_pred, pos_label=1))*100)

    return {
        "accuracy": accuracy,
        "precision_class_0": precision_class_0,
        "recall_class_0": recall_class_0,
        "precision_class_1": precision_class_1,
        "recall_class_1": recall_class_1
    }

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/get_metrics', methods=['POST'])
def get_metrics():
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    print(metrics)
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5001)