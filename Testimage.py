import os
import tensorflow as tf
import numpy as np
import pathlib
import json

# Load the trained model architecture from the JSON file
with open("model_arch.json", "r") as json_file:
    model_json = json_file.read()

# Create the model from the loaded architecture
model = tf.keras.models.model_from_json(model_json)

# Load the trained weights
model.load_weights("model_weights.h5")

# Set the path to the data directory
data_dir = pathlib.Path('C:\\Users\\User\\Documents\\FYP\\FYP\\data')

img_height = 180
img_width = 180

# Recreate the training dataset to access class names
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=32
)

# Get class names from the training dataset
class_names = train_ds.class_names

# Set the path to the directory containing test images
test_directory = "C:\\Users\\User\\Documents\\FYP\\FYP\\test"

# Get a list of all JPEG files in the directory
image_count = len(list(data_dir.glob('*/*.jpeg')))
test_image_paths = [os.path.join(test_directory, f) for f in os.listdir(test_directory) if f.lower().endswith('.jpeg')]

# Ensure there are images in the directory
if not test_image_paths:
    print("No JPEG images found in the specified test directory.")
    exit()

# Iterate over each test image
for test_image_path in test_image_paths:
    # Load the test image
    try:
        img = tf.keras.utils.load_img(
            test_image_path, target_size=(img_height, img_width)
        )
    except Exception as e:
        print(f"Error loading the image {test_image_path}: {e}")
        continue

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions using the trained model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display the prediction results with class name
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(
        f"The image most likely belongs to class {predicted_class} with a {confidence:.2f}% confidence."
    )
