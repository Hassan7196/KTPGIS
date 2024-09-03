import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Load model
model = load_model('final_model.keras')

# Preprocess image
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

# Load and preprocess test image
image_path = "PHOTO-2023-04-08-11-50-00.jpg"
preprocessed_image = preprocess_image(image_path)
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# Predict
prediction = model.predict(preprocessed_image)
print(prediction)  # Debugging output

# Decode and save prediction
def decode_one_hot_to_rgb(one_hot_mask):
    height, width = one_hot_mask.shape[:2]
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    class_colors = {
        0: (0, 255, 255),  # Urban
        1: (0, 0, 255),    # Water
        2: (0, 255, 0),    # Forest
        3: (255, 255, 0),  # Agriculture
        4: (255, 0, 255)   # Road
    }
    for class_index, color in class_colors.items():
        rgb_mask[one_hot_mask.argmax(axis=-1) == class_index] = color
    return rgb_mask

predicted_rgb = decode_one_hot_to_rgb(prediction[0])
Image.fromarray(predicted_rgb).save("test_pred.png")
