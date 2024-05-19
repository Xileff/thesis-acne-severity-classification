from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from keras.models import load_model
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Validate file extension
def validate_file(filename: str):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to classify image
def classify_image(image_path: str):
  clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
  image = cv2.resize(cv2.imread(image_path), (224, 224), interpolation=cv2.INTER_AREA)
  r_channel, g_channel, b_channel = cv2.split(image)
  enhanced_r = clahe.apply(r_channel)
  enhanced_g = clahe.apply(g_channel)
  enhanced_b = clahe.apply(b_channel)
  image = cv2.merge((enhanced_r, enhanced_g, enhanced_b))
  image = np.expand_dims(image, axis=0)
  try:
    model_path = os.path.join("ml-model", "InceptionV3-Dermnet-SMOTE-Minority-LR-0.001-Batch Size 16.h5")
    model = load_model(model_path)
    result = model.predict(image)
    print(result)
    return str(result)
  except Exception as e:
    print(e)
    print("ERROR")

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    if 'image' not in request.files:
      return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
      return redirect(request.url)
    if file and validate_file(file.filename):
      filename = secure_filename(file.filename)
      file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(file_path)
      result = classify_image(file_path)
      return render_template('index.html', result=result, filename=filename)
  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)