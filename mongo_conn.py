import pymongo
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from flask import Flask, render_template, jsonify
from flask_caching import Cache
app = Flask(__name__)

# Connecting to MongoDB
client=pymongo.MongoClient("mongodb://c175dcitetris.int.thomsonreuters.com:27018/")

# Selecting the Database
db = client["Dupe-Content-App"]

# Selecting the Collection
collection = db["plagiarisms"]

# Query the collection to retrieve data
#data = collection.find().skip(300).limit(22)
data = collection.find()
# Initialize lists to store the extracted data
plagiarisedSts_list = []
status_list = []

# Iterate through the last 50 documents and extract the data
for document in data:
    if "save_data" in document and "results" in document["save_data"]:
        results = document["save_data"]["results"]
        for result in results:
            if "result" in result:
                for subresult in result["result"]:
                    if "plagiarisedSts" in subresult and "status" in subresult:
                        plag_status = subresult["plagiarisedSts"]
                        status = subresult["status"]
                        
                        # Check if "Error: Unable to generate parameters." is not present
                        if plag_status != "Error: Unable to generate parameters.":
                            # Convert to numerical values
                            plag_status = 1 if plag_status == "Plagiarized" else 0
                            status = 1 if status == "Accepted" else 0

                            # Append to the respective lists
                            plagiarisedSts_list.append(plag_status)
                            status_list.append(status)

# Create the final dictionary
final_dict = {
    "plagiarisedSts": plagiarisedSts_list,
    "status": status_list
}


y_test = final_dict["status"]
y_pred = final_dict["plagiarisedSts"]


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
    app.run(debug=True, port=5003)