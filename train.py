import kagglehub
import tensorflow as tf
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
print("Path to dataset files:", path)
import os

datasetpath ="/kaggle/input/garbage-classification/Garbage classification/Garbage classification"
#print(os.listdir(datasetpath))

tdata = tf.keras.utils.image_dataset_from_directory(
datasetpath,
validation_split= 0.2,
seed = 42,
subset = "training",
image_size = (224, 224),

batch_size = 32)


vdata = tf.keras.utils.image_dataset_from_directory(
datasetpath,
validation_split= 0.2,
seed = 42,
subset = "validation",
image_size = (224, 224),

batch_size = 32)
print(tdata.class_names)
autotune = tf.data.AUTOTUNE
tdata = tdata.prefetch(autotune)
vdata = vdata.prefetch(autotune)

for images, labels in tdata.take(1):
    print(images.shape)
    print(labels.shape)



basemodel = tf.keras.applications.MobileNetV2(
weights="imagenet",
input_shape = (224, 224,3) ,
include_top = False,
)
#tdata = tf.keras.applications.mobilenet_v2.preprocess_input(tdata)

def process (image, label) :
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

tdata = tdata.map(process)
vdata = vdata.map(process)
basemodel.trainable = False
x = basemodel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu") (x)
output = tf.keras.layers.Dense(6,activation="softmax") (x)
model = tf.keras.Model(basemodel.input, output)
model.compile(
metrics = ["accuracy"],
loss = "sparse_categorical_crossentropy",
optimizer = "adam"
)



model.fit(tdata,validation_data=vdata, epochs=5)
model.evaluate(vdata)
model.save("model.keras")
from google.colab import files
files.download("model.keras")