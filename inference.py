import json
import numpy as np
import tensorflow as tf

# The 'model_fn' is required to load the model when the SageMaker endpoint starts
def model_fn(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

# The 'input_fn' is used to deserialize the JSON input data into a NumPy array
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# The 'predict_fn' is used to perform inference using the loaded model
def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions

# The 'output_fn' is used to serialize the predictions into JSON format
def output_fn(predictions, content_type='application/json'):
    if content_type == 'application/json':
        return json.dumps(predictions.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
