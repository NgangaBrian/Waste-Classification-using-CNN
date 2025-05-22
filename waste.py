from flask import Flask, request, render_template_string, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the Keras model
model = tf.keras.models.load_model("waste_classification.keras")

# HTML Template with image display
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Waste Classification</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        background: #f4f4f4;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
      }
      .container {
        background: white;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        text-align: center;
        max-width: 400px;
        width: 100%;
      }
      h1 {
        color: #333;
        margin-bottom: 20px;
      }
      input[type=file] {
        margin: 20px 0;
      }
      input[type=submit] {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
      }
      input[type=submit]:hover {
        background-color: #45a049;
      }
      .result {
        margin-top: 20px;
        font-size: 20px;
        color: #444;
        font-weight: bold;
      }
      .uploaded-image {
        margin-top: 20px;
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        border: 1px solid #ccc;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Waste Classification</h1>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br>
        <input type="submit" value="Classify">
      </form>
      {% if image_url %}
        <img src="{{ image_url }}" class="uploaded-image" alt="Uploaded Image">
      {% endif %}
      {% if result %}
        <div class="result">Prediction: {{ result }}</div>
        <div class="result">{{ recomend }}</div>
      {% endif %}
    </div>
  </body>
</html>
"""

def preprocess_image(image, target_size=(150, 150)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def classify():
    result = None
    recomend = None
    image_url = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            # Save image
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_url = f"/{file_path.replace(os.sep, '/')}"

            # Preprocess and predict
            image = Image.open(file_path)
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)

            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            if predicted_class == 0:
                result = "Non-Recyclable"
                recomend = "Dispose the item in the non-recyclable bin"
            else:
                result = "Recyclable"
                recomend = "Dispose the item in a recyclables bin"

    return render_template_string(HTML_TEMPLATE, result=result, recomend=recomend, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
