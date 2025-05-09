from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import torch
import io
import numpy as np
import os

app = Flask(__name__)

model = YOLO(
    # load most up-to-date model
    r'C:\Users\22eva\Documents\ecen4493_final_project\runs\classify\train\weights\last.pt')

# for user uploads of images :)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    # Load home screen html template
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']
    if not file:
        return "No file uploaded", 400

     # Save the file to static/uploads/{filename}
    upload_path = f"static/uploads/{file.filename}"
    file.save(upload_path)

    # Open the image using PIL
    try:
        img = Image.open(upload_path)
    except Exception as e:
        return f"Error opening image: {e}", 400

    # put the image through the trained model!
    results = model(img)

    if results:

        # only 1 image, so results[0]
        probs = results[0].probs

        # get tensor form of probability data so torch.argmax() can be used
        probs_tensor = probs.data

        # Get class index of the highest probability (predicted) class
        max_prob_idx = torch.argmax(probs_tensor).item()

        # Get the predicted class name from the results
        predicted_class = results[0].names[max_prob_idx]

        # Get the confidence score for the highest probability
        confidence = probs_tensor[max_prob_idx].item()

        # Return the predicted class and confidence
        return render_template('result.html', class_name=predicted_class, confidence=confidence, image_path=file.filename)
    else:
        return render_template('result.html', class_name="No results from model", confidence=None, image_path=None)


if __name__ == '__main__':
    app.run(debug=True)
