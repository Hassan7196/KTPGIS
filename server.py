import numpy as np
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import tempfile
import zipfile
import geopandas as gpd
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image, decode_one_hot_to_rgb, allowed_file, save_prediction

app = Flask(__name__)


# np.random.seed(42)
# tf.random.set_seed(42)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'kml', 'kmz', 'geojson', 'zip', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('final_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        preprocessed_image = preprocess_image(file_path)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        prediction = model.predict(preprocessed_image)
        predicted_mask = np.argmax(prediction[0], axis=-1)

        print(f"Prediction shape: {prediction.shape}")
        print(f"Unique values in predicted mask: {np.unique(predicted_mask)}")




        original_image = Image.open(file_path)


        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')


        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='viridis')
        plt.colorbar()
        plt.title('Predicted Mask Visualization')
        plt.axis('off')

        plt.show()

        return 'Prediction and visualization completed in the terminal.'

    else:
        return 'File type not allowed'



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        geojson_filename = convert_to_geojson(file_path, filename)
        if geojson_filename:
            return redirect(url_for('show_map', geojson_file=geojson_filename))


def convert_to_geojson(file_path, filename):
    gdf = None
    if filename.lower().endswith('.kmz'):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            kml_files = [f for f in os.listdir(temp_dir) if f.endswith('.kml')]
            if kml_files:
                gdf = gpd.read_file(os.path.join(temp_dir, kml_files[0]))
    elif filename.lower().endswith('.kml'):
        gdf = gpd.read_file(file_path)
    elif filename.endswith('.geojson'):
        return filename  # Already in GeoJSON format
    elif filename.endswith('.zip'):  # Assuming this is a Shapefile
        gdf = gpd.read_file("zip://" + file_path)
    else:
        return None

    geojson_filename = os.path.splitext(filename)[0] + '.geojson'
    geojson_path = os.path.join(app.config['UPLOAD_FOLDER'], geojson_filename)
    gdf.to_file(geojson_path, driver='GeoJSON')
    return geojson_filename


@app.route('/map')
def show_map():
    geojson_file = request.args.get('geojson_file')
    return render_template('map.html', geojson_file=geojson_file)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(debug=True)

