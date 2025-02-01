from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
import logging
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import glob
import joblib

app = Flask(__name__)
CORS(app)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming the model and feature vectors are saved in the same directory as the script
current_directory = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_directory, 'model_80x80.pkl')  # Update with your .pkl file
logger.info(f"Model Path: {model_file_path}")

# Load the pre-trained model using joblib
loaded_model = joblib.load(model_file_path)

# Assuming the feature vectors file is in the same directory as the script
feature_vectors_file = os.path.join(current_directory, 'raw_features_MainDataset_80x80.npy')

# Load the pre-computed image feature vectors
normalized_features = np.load(feature_vectors_file)
logger.info(f"Normalized features: {normalized_features}")

target_size = (80, 80)

def extract_image_features(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        x = tf.keras.applications.mobilenet.preprocess_input(img[np.newaxis, ...])
        features = loaded_model.predict(x)
        return features
    except Exception as e:
        logger.error(f"Error processing the uploaded image: {e}")
        return None


def get_similar_images(image_index, cosine_similarities, top_n=5):
    similarity_scores = list(enumerate(cosine_similarities[image_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_images = similarity_scores[1:top_n+1]
    similar_image_indices = [index for index, _ in top_similar_images]
    return similar_image_indices

@app.route('/')
def index():
    return "Welcome to the Fashion Recommendation API!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/image/<path:image_filename>')
def serve_image(image_filename):
    try:
        return send_from_directory(os.path.join(current_directory, 'images'), image_filename)
    except Exception as e:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        uploaded_image = request.files['image']
        temp_image_path = 'temp.jpg'
        uploaded_image.save(temp_image_path)

        uploaded_features = extract_image_features(temp_image_path)

        if uploaded_features is None:
            return jsonify({'error': 'Error processing the uploaded image. Please try again.'}), 500

        # Append the new image features to the existing normalized features
        updated_features = np.vstack((normalized_features, uploaded_features))
        logger.info(f"updated_features: {updated_features}")

        # Normalize the entire feature array
        # data_normalized = updated_features / np.linalg.norm(updated_features, axis=1, keepdims=True)
        # logger.info(f"data_normalized: {data_normalized}")

        # Compute cosine similarity with the input image
        # logger.info("Before computing cosine similarities.")
        # cosine_similarities = np.dot(data_normalized, data_normalized.T)
        cosine_similarities = cosine_similarity(updated_features)
        logger.info("After computing cosine similarities: %s", cosine_similarities[-1])

        image_index = -1  # Index of the uploaded image
        similar_image_indices = get_similar_images(image_index, cosine_similarities)

        # Get the file paths of recommended images
        image_dir = 'images'
        image_paths = glob.glob(os.path.join(current_directory, image_dir, '*.jpg'))

        # Get the filenames with extensions of recommended images
        image_filenames = [os.path.basename(image_paths[idx]) for idx in similar_image_indices]
        logger.info(f"Recommended Image Filenames: {image_filenames}")

        return jsonify({'recommended_images': image_filenames}), 200

    except Exception as e:
        logger.exception('Error occurred during recommendation')
        return jsonify({'error': 'An error occurred during recommendation. Please try again later.'}), 500



if __name__ == '__main__':
    app.run(debug=True)
