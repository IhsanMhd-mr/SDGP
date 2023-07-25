import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATADIR = r"C:\Users\acer\Downloads\w1867202\SDGP\Dataset"
CATEGORIES = ['Chickenpox','Mild','Monkeypox','Normal','Severe']
            
img_size = (224, 224)  # Specify the desired image size
batch_size = 32  

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

y_train = CATEGORIES

# Create a mapping from class labels to integers
class_labels = np.unique(y_train)
label_to_integer = {label: idx for idx, label in enumerate(class_labels)}

# Convert the string labels to integers
y_train_int = [label_to_integer[label] for label in y_train]

# Calculate the class weights
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(class_labels), y= class_labels)
class_weights_dict = dict(zip(np.unique(y_train_int), class_weights))


import tensorflow as tf

# Define custom loss function with class weights
def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Convert class weights to float32
        class_weights_float32 = tf.cast(class_weights, dtype=tf.float32)

        # Calculate the cross-entropy loss for each sample
        ind_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Apply class weights to the loss for each sample
        weights = tf.gather(class_weights_float32, tf.argmax(y_true, axis=1))
        weighted_loss = ind_loss * weights

        # Return the average loss over all samples
        return tf.reduce_mean(weighted_loss)

    return loss

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


CATEGORIES = ['Chickenpox', 'Mild', 'Monkeypox', 'Normal', 'Severe']
IMG_SIZE = 224

xxx = "C:/Users/acer/Downloads/w1867202/SDGP/test/Monkeypox/Monkeypox3.png"

# Load the pre-trained model
with tf.keras.utils.custom_object_scope({"loss": weighted_categorical_crossentropy}):
    model = load_model('saved_model.h5')

# Load and preprocess the image
img = image.load_img(xxx, target_size=(IMG_SIZE, IMG_SIZE))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype('float32') / 255.0

# Get the prediction
predicted_img = model.predict(img)
index_predicted_img = np.argmax(predicted_img)
predict = CATEGORIES[index_predicted_img]
print(predict)
