import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

def train_model():
    # Define the directory to the training image folders
    image_directory = 'data/training_set'

    # List of face shape categories (subfolder names)
    face_shape_categories = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

    # Prepare training data and labels
    X_train = []
    y_train = []

    # Loop through each category (subfolder)
    for category in face_shape_categories:
        # Full path to the subfolder
        category_path = os.path.join(image_directory, category)
        
        # Get list of all images in the subfolder
        image_paths = [os.path.join(category_path, filename) 
                    for filename in os.listdir(category_path) 
                    if filename.endswith('.jpg') or filename.endswith('.png')]
        
        # Preprocess images and append to X_train
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (224, 224))  # Resize to MobileNetV2 input size
            img_array = np.array(img_resized) / 255.0  # Normalize the image to [0, 1] range
            X_train.append(img_array)
            
            # Append the corresponding label (index of the category)
            y_train.append(face_shape_categories.index(category))

    # Convert X_train and y_train to NumPy arrays
    X_train = np.array(X_train)
    y_train = to_categorical(y_train, num_classes=len(face_shape_categories))  # One-hot encode the labels

    # Load MobileNetV2 pre-trained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(face_shape_categories), activation='softmax')  # Number of categories
    ])

    # Compile and train model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    model.save('model/face_shape_classifier.h5')

    if __name__ == "__main__":
        train_model()