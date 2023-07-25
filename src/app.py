from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path 
import os
from sklearn.utils.class_weight import compute_class_weight

DATADIR = "C:/Users/acer/Downloads/w1867202/SDGP/Dataset"
CATEGORIES = ['Chickenpox', 'Mild', 'Monkeypox', 'Normal', 'Severe']
IMG_SIZE = 224
img_size = (224, 224)  # Specify the desired image size
batch_size = 32

# Create and convert mapping from class labels to integers
class_labels = CATEGORIES
label_to_integer = {label: idx for idx, label in enumerate(class_labels)}
y_train_int = [label_to_integer[label] for label in class_labels]

# Calculate the class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels)
class_weights_dict = dict(zip(np.unique(y_train_int), class_weights))


# Define custom loss function with class weights
def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        class_weights_float32 = tf.cast(class_weights, dtype=tf.float32)

        # Calculate the cross-entropy loss for each sample
        ind_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weights = tf.gather(class_weights_float32, tf.argmax(y_true, axis=1))
        weighted_loss = ind_loss * weights

        return tf.reduce_mean(weighted_loss)

    return loss


# Load the pre-trained model
with tf.keras.utils.custom_object_scope({"loss": weighted_categorical_crossentropy}):
    model = load_model('saved_model.h5')

def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0

    # Get the prediction
    predicted_img = model.predict(img)
    index_predicted_img = np.argmax(predicted_img)
    predict = CATEGORIES[index_predicted_img]
    print(predict)
    return predict



app = Flask(__name__)

@app.route("/")
def index():
    # Render the index.html template and serve it to the client
    return render_template("experiment.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        files = request.files.getlist("file")  # Get a list of uploaded files

        predictions = []  # List to store the predictions for each image

        for file in files:
            if file.filename == "":
                return jsonify({"error": "One of the files has no selected file"}), 400

            # Convert Path 
            temp_image_path = Path(file.filename)

            # Save the uploaded image temporarily 
            file.save(temp_image_path)

            # temp_path = "C:/Users/acer/Downloads/w1867202/SDGP/test/Monkeypox/Monkeypox3.png"
            # prediction = predict_image(temp_path)

            prediction = predict_image(temp_image_path)

            # Remove the temporary image file 
            temp_image_path.unlink()
            print(prediction)
            predictions.append(prediction)

        # Return the predictions as the response
        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
