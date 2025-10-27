#!/usr/bin/env python3
"""
Standalone Irish Sign Language Recognition System
Real-time gesture recognition without Flask
"""

import cv2
import numpy
import math
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# ==================== CONFIGURATION ====================
# Path to your dataset - UPDATE THIS PATH
DATASET_PATH = './ISL_dataset'  # Change this to your dataset location
MODEL_PATH = './keras_model.h5'  # Where to save/load the trained model

# Model parameters
TOTAL_DATASET = 2515
img_rows, img_cols = 400, 400
img_channels = 3
batch_size = 32
nb_epoch = 100
nb_classes = 36

# Dictionary for classes from char to numbers
classes = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17,
    'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25,
    'q': 26, 'r': 27, 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33,
    'y': 34, 'z': 35,
}


# ==================== MODEL FUNCTIONS ====================
def load_dataset(dataset_path):
    """Load the dataset and populate training arrays"""
    x_train = []
    y_train = []
    
    print("Loading dataset...")
    for root, directories, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(".jpeg"):
                fullpath = os.path.join(root, filename)
                try:
                    img = load_img(fullpath)
                    img = img_to_array(img)
                    x_train.append(img)
                    
                    # Extract class label from path
                    t = fullpath.rindex('/')
                    fullpath_dir = fullpath[0:t]
                    n = fullpath_dir.rindex('/')
                    y_train.append(classes[fullpath_dir[n + 1:t]])
                except Exception as e:
                    print(f"Error loading {fullpath}: {e}")
    
    print(f"Loaded {len(x_train)} images")
    return x_train, y_train


def create_model(input_shape):
    """Create CNN model for training"""
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    return model


def train_model(model, X_train, Y_train):
    """Train the model using SGD"""
    print("Training model...")
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch)
    
    return model


def load_or_train_model():
    """Load existing model or train a new one"""
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    
    # Train new model
    print("No existing model found. Training new model...")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path '{DATASET_PATH}' does not exist!")
        print("Please update DATASET_PATH in the script to point to your ISL_dataset folder")
        return None
    
    x_train, y_train = load_dataset(DATASET_PATH)
    
    if len(x_train) == 0:
        print("ERROR: No training data found!")
        return None
    
    # Prepare training data
    a = numpy.asarray(y_train)
    y_train_new = a.reshape(a.shape[0], 1)
    
    X_train = numpy.asarray(x_train).astype('float32')
    X_train = X_train / 255.0
    Y_train = to_categorical(y_train_new, nb_classes)
    
    # Create and train model
    model = create_model(X_train.shape[1:])
    model = train_model(model, X_train, Y_train)
    
    # Save model
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    return model


def identify_gesture(handTrainImage, model):
    """Identify the gesture from the cropped hand image"""
    try:
        # Convert to RGB
        handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(handTrainImage)
        img_w, img_h = img.size
        
        # Pad to square
        M = max(img_w, img_h)
        background = Image.new('RGB', (M, M), (0, 0, 0))
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(img, offset)
        
        # Resize to 400x400
        size = 400, 400
        background = background.resize(size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        open_cv_image = numpy.array(background)
        background = open_cv_image.astype('float32')
        background = background / 255
        background = background.reshape((1,) + background.shape)
        
        # Predict
        predictions = model.predict(background, verbose=0)
        predicted_class = numpy.argmax(predictions, axis=1)[0]
        
        # Get class name
        for key, value in classes.items():
            if value == predicted_class:
                return key
        
        return None
    except Exception as e:
        print(f"Error in identify_gesture: {e}")
        return None


# ==================== MAIN RECOGNITION LOOP ====================
def main():
    """Main function to run gesture recognition"""
    print("=" * 60)
    print("Irish Sign Language Recognition System")
    print("=" * 60)
    
    # Load or train model
    model = load_or_train_model()
    if model is None:
        print("Failed to load or train model. Exiting.")
        return
    
    print("\nModel loaded successfully!")
    print("\nStarting camera...")
    print("Controls:")
    print("  - Adjust trackbars to calibrate skin color detection")
    print("  - Press ESC to exit")
    print("=" * 60)
    
    # Create windows
    cv2.namedWindow('Camera Output')
    cv2.namedWindow('Hand Detection')
    cv2.namedWindow('Hand Train (Contour)')
    
    # Create trackbars for skin color calibration
    def nothing(x):
        pass
    
    cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
    cv2.createTrackbar('G for min', 'Camera Output', 133, 255, nothing)
    cv2.createTrackbar('R for min', 'Camera Output', 103, 255, nothing)
    cv2.createTrackbar('B for max', 'Camera Output', 255, 255, nothing)
    cv2.createTrackbar('G for max', 'Camera Output', 182, 255, nothing)
    cv2.createTrackbar('R for max', 'Camera Output', 130, 255, nothing)
    
    # Initialize video capture
    videoFrame = cv2.VideoCapture(0)
    
    if not videoFrame.isOpened():
        print("ERROR: Cannot open camera!")
        return
    
    # Initialize variables
    x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0
    _, prevHandImage = videoFrame.read()
    prevcnt = numpy.array([], dtype=numpy.int32)
    gestureStatic = 0
    gestureDetected = 0
    letterDetected = None
    
    # Main loop
    keyPressed = -1
    while keyPressed < 0:
        # Get skin color range from trackbars
        min_YCrCb = numpy.array([
            cv2.getTrackbarPos('B for min', 'Camera Output'),
            cv2.getTrackbarPos('G for min', 'Camera Output'),
            cv2.getTrackbarPos('R for min', 'Camera Output')
        ], numpy.uint8)
        
        max_YCrCb = numpy.array([
            cv2.getTrackbarPos('B for max', 'Camera Output'),
            cv2.getTrackbarPos('G for max', 'Camera Output'),
            cv2.getTrackbarPos('R for max', 'Camera Output')
        ], numpy.uint8)
        
        # Read frame
        readSuccess, sourceImage = videoFrame.read()
        if not readSuccess:
            print("Failed to read frame")
            break
        
        # Convert to YCrCb and blur
        imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
        imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)
        
        # Find skin region
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        
        # Find contours
        contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            keyPressed = cv2.waitKey(30)
            continue
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get largest contour
        cnt = contours[0]
        
        # Compare with previous contour
        if len(prevcnt) > 0:
            ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
            if ret > 0.70:
                gestureStatic = 0
            else:
                gestureStatic += 1
        
        prevcnt = cnt
        
        # Create mask for hand
        stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil, [cnt], color)
        handTrainImage = cv2.bitwise_and(sourceImage, stencil)
        
        # Get bounding rectangle
        x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)
        
        # Draw rectangle
        cv2.rectangle(sourceImage, (x_crop, y_crop), 
                     (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)
        
        # Update crop if changed significantly
        if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
            abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
            x_crop_prev = x_crop
            y_crop_prev = y_crop
            h_crop_prev = h_crop
            w_crop_prev = w_crop
        
        # Create cropped images
        handImage = sourceImage.copy()[
            max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
            max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50
        ]
        
        handTrainImage = handTrainImage[
            max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
            max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15
        ]
        
        # Detect gesture if static
        if gestureStatic == 10:
            gestureDetected = 10
            print("Gesture Detected - Identifying...")
            letterDetected = identify_gesture(handTrainImage, model)
            if letterDetected:
                print(f"Detected Letter: {letterDetected.upper()}")
        
        # Display detected letter
        if gestureDetected > 0:
            if letterDetected is not None:
                cv2.putText(sourceImage, letterDetected.upper(), (10, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            gestureDetected -= 1
        
        # Draw contours
        cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)
        
        # Display windows
        cv2.imshow('Camera Output', sourceImage)
        if handImage.size > 0:
            cv2.imshow('Hand Detection', handImage)
        if handTrainImage.size > 0:
            cv2.imshow('Hand Train (Contour)', handTrainImage)
        
        # Check for ESC key
        keyPressed = cv2.waitKey(30)
    
    # Cleanup
    print("\nCleaning up...")
    cv2.destroyAllWindows()
    videoFrame.release()
    print("Done!")


if __name__ == "__main__":
    main()
