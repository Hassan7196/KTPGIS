import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 5


class_colors = {
    0: (0, 255, 255),  # Urban - Label 0
    1: (0, 0, 255),    # Water - Label 1
    2: (0, 255, 0),    # Forest - Label 2
    3: (255, 255, 0),  # Agriculture - Label 3
    4: (255, 0, 255)   # Road - Label 4
}

def preprocess_image(image_path):

    image = Image.open(image_path).convert('RGB').resize((256, 256))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

def preprocess_mask(mask_path):

    mask_image = Image.open(mask_path).convert('RGB').resize((256, 256))
    mask_image = np.array(mask_image)


    mask_out = np.zeros((256, 256), dtype=np.uint8)

    class_colors = {
        (0, 255, 255): 0,   # Urban - Label 0
        (0, 0, 255): 1,     # Water - Label 1
        (0, 255, 0): 2,     # Forest - Label 2
        (255, 255, 0): 3,   # Agriculture - Label 3
        (255, 0, 255): 4    # Road - Label 4
    }


    for rgb, idx in class_colors.items():
        mask_out[(mask_image == rgb).all(axis=-1)] = idx

    mask_out = to_categorical(mask_out, num_classes=5)
    return mask_out


def decode_one_hot_to_rgb(one_hot_mask):

    height, width = one_hot_mask.shape[:2]
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_index, color in class_colors.items():
        rgb_mask[one_hot_mask.argmax(axis=-1) == class_index] = color

    return rgb_mask



def save_prediction(prediction, filename, output_dir='static/predictions/'):
    predicted_rgb_mask = decode_one_hot_to_rgb(prediction)

    os.makedirs(output_dir, exist_ok=True)

    prediction_path = os.path.join(output_dir, 'pred_' + filename)
    Image.fromarray(predicted_rgb_mask).save(prediction_path)

    return prediction_path

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
