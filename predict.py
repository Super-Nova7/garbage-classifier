import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.keras")

img = tf.keras.utils.load_img(
    "test.jpg",
    target_size=(224,224)
)

img = tf.keras.utils.img_to_array(img)

img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

print(prediction.argmax())

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]