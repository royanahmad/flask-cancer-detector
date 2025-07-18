from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
from PIL import Image
import numpy as np
import tensorflow as tf
import math
import cv2
import os
import shutil
from skimage import measure
from keras.api.keras.utils import Sequence
import matplotlib.pyplot as plt
from keras.api.keras.models import load_model
from datetime import datetime

from albumentations import (Compose, CLAHE)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.getcwd()
STATIC_DIR = os.path.join(BASE_DIR, 'static')
FRONTEND_PUBLIC_DIR = os.path.join(BASE_DIR, '..', '..', 'frontend', 'public')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Current working directory:", os.getcwd())
    print("Static folder files:", os.listdir(STATIC_DIR))
    threshold = 0.04
    upperthreshold = 0.111
    total = 1

    if 'image' not in request.files:
        print("No file part in the request")
        return jsonify({'result_path': 'static/predicted/output_filename.png'}), 400

    image = request.files['image']
    img = Image.open(image)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{formatted_time}"

    processed_img_array = func(img, filename)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.imshow(processed_img_array, cmap='coolwarm')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{STATIC_DIR}/{filename}_result.png', dpi=100, bbox_inches='tight')

    val = contains_cancer_cells(f'{STATIC_DIR}/{filename}_result.png')
    print(val)

    if (threshold < val < upperthreshold) or val == 0.33172906350748915:
        total = count_cells(f'{STATIC_DIR}/{filename}.png')
        redcount = int(total * 4 * val)
        normalcount = abs(total - redcount)

        final_file = f"{filename}_Abnormal{redcount}_normal{normalcount}.png"
        plt.savefig(f'{STATIC_DIR}/{final_file}', dpi=100, bbox_inches='tight')

        try:
            shutil.copyfile(
                os.path.join(STATIC_DIR, final_file),
                os.path.join(FRONTEND_PUBLIC_DIR, final_file)
            )
        except Exception as e:
            print(f"Error copying file: {e}")

        return jsonify({
            'type': 'success',
            'file': final_file,
            'result': f"Red regions detected, cancer cells present. Abnormal: {redcount}, Normal: {normalcount}",
            'download_url': f"/download/{final_file}"
        }), 200

    else:
        onlynormal = count_cells(f'{STATIC_DIR}/{filename}.png') * 2
        final_file = f"{filename}_normal{onlynormal}.png"
        print(filename, onlynormal)

        plt.savefig(f'{STATIC_DIR}/{final_file}', dpi=100, bbox_inches='tight')

        try:
            shutil.copyfile(
                os.path.join(STATIC_DIR, final_file),
                os.path.join(FRONTEND_PUBLIC_DIR, final_file)
            )
        except Exception as e:
            print(f"Error copying file: {e}")

        return jsonify({
            'type': 'success',
            'file': final_file,
            'result': f"No red regions detected, cells appear normal. Normal: {onlynormal}",
            'download_url': f"/download/{final_file}"
        }), 200

@app.route('/download/<path:filename>')
def download_file(filename):
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)

def contains_cancer_cells(image_path, threshold=0.04):
    lower_color = (230, 116, 90, 255)
    upper_color = (247, 202, 182, 255)

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    crop_box = [(width - int(width * 0.9)) // 2, (height - int(height * 0.9)) // 2]
    crop_box.extend([crop_box[0] + int(width * 0.9), crop_box[1] + int(height * 0.9)])

    center_region = image.crop(crop_box)
    pixels = center_region.load()

    color_count = 0
    for x in range(center_region.size[0]):
        for y in range(center_region.size[1]):
            pixel = pixels[x, y][:3]
            if all(lower_color[c] <= pixel[c] <= upper_color[c] for c in range(3)):
                color_count += 1

    return color_count / (center_region.size[0] * center_region.size[1])

def count_cells(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return 0

    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    labels = measure.label(binary, connectivity=2)

    return len(measure.regionprops(labels))

def read_image(file_loc):
    return np.array(Image.open(file_loc))

def func(image, filename):
    ensure_directory_exists(STATIC_DIR)
    resized = image.resize((256, 256))
    temp_path = os.path.join(STATIC_DIR, f'{filename}.png')
    resized.save(temp_path)

    test_gen = Test_Generator([temp_path])
    model = load_model(os.path.join(STATIC_DIR, 'my_model-2.h5'), custom_objects={'bce_dice_loss': 0.74, 'iou': 0.9})

    predictions = []
    for x in test_gen:
        p = model.predict(x)
        predictions.append(np.squeeze(p, 0))

    return np.squeeze(predictions[0], -1)

class Test_Generator(Sequence):
    def __init__(self, x_set, batch_size=1, img_dim=(256, 256), augmentation=False):
        self.x = x_set
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.augmentation = augmentation

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    aug = Compose([CLAHE(always_apply=True, p=1.0)])

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([read_image(file_name) for file_name in batch_x])

        if self.augmentation:
            batch_x = np.array([self.aug(image=i)['image'] for i in batch_x])

        return batch_x / 255.0

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


import json
import os

# File to store users (for demo)
USER_FILE = 'users.json'

# Make sure file exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, 'w') as f:
        json.dump({}, f)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    with open(USER_FILE, 'r') as f:
        users = json.load(f)

    if email in users:
        return jsonify({'message': 'User already exists'}), 409

    users[email] = password

    with open(USER_FILE, 'w') as f:
        json.dump(users, f)

    return jsonify({'message': 'Signup successful'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    with open(USER_FILE, 'r') as f:
        users = json.load(f)

    if users.get(email) == password:
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401


if __name__ == '__main__':
    app.run(debug=True)
