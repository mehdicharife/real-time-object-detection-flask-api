from flask import Flask, request, jsonify, send_file
import os
import time
from ultralytics import YOLO
from core import annotate


# Loading the YOLO model
model = YOLO("yolov8.yaml")
model = YOLO('yolov8n.pt')

app = Flask(__name__)

# Define a list to store the objects
objects = []

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file is present in the request
        if 'file' not in request.files:
            return "No file uploaded"

        file = request.files['file']
        objects = request.form.getlist('objects')  # Get the list of objects from the request
        print("Received objects:", objects)

        # Check if the file is empty
        if file.filename == '':
            return "Empty file uploaded"

        # Save the file to a temporary location
        file_path = os.path.join('incoming', file.filename)
        file.save(file_path)

        ret_file_path = annotate(model, file_path, file.filename)


        # Delay for 5 seconds before returning the file
        time.sleep(5)

        # Return the processed file to the user
        return send_file(ret_file_path, as_attachment=True)
        

    return "Hello World!"

@app.route("/getallobjects", methods=['GET'])
def get_all_objects():
    global objects

    objects = [
        {'id': 1, 'name': 'Car'},
        {'id': 2, 'name': 'Bus'},
        {'id': 3, 'name': 'Truck'},
        {'id': 4, 'name': 'House'},
        {'id': 5, 'name': 'Person'},
    ]

    # Return the list of objects as JSON response
    return jsonify(objects)

if __name__ == "__main__":
    app.run()