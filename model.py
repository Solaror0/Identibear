import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomBrightness, RandomRotation
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

img_path = "th_wu_man.jpg"
img = load_img(img_path, target_size=(37, 50))  # Resize the image
img_array = img_to_array(img)  # Convert the image to an array
img_array = img_array 

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

def preprocess_image(image_path, target_size=(37, 50), color_mode='grayscale'):
    img = cv2.imread(image_path)
    if color_mode == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    if color_mode == 'grayscale':
        img = img[..., np.newaxis]  # Add channel dimension for grayscale images
    img = img / 255.0  # Normalize to [0, 1]
    return img

def predict_image(model, image_path, target_size=(37, 50), color_mode='grayscale'):
    img = preprocess_image(image_path, target_size, color_mode)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]


def load_images_and_labels(image_dir, label_dir, target_size=(37, 50), color_mode='grayscale'):
    images = []
    labels = []
    for file in sorted(os.listdir(image_dir)):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if color_mode == 'grayscale':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, target_size)
            images.append(img)
            
            # Read the corresponding label
            label_file = os.path.join(label_dir, os.path.splitext(file)[0] + '.txt')
         
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label = int(f.read().strip())
                
            else:
                label = 0  # Default label if no label file is found
        
            labels.append(label)
    
    images = np.array(images)
    if color_mode == 'grayscale':
        images = images[..., np.newaxis]  # Add channel dimension for grayscale images
    labels = np.array(labels)
    return images, labels

# Define directories
train_images_dir = os.path.join(os.getcwd(), 'train', 'images')
train_labels_dir = os.path.join(os.getcwd(), 'train', 'labels')
test_images_dir = os.path.join(os.getcwd(), 'test', 'images')
test_labels_dir = os.path.join(os.getcwd(), 'test', 'labels')

# Load images and labels
X_train, y_train = load_images_and_labels(train_images_dir, train_labels_dir)
X_test, y_test = load_images_and_labels(test_images_dir, test_labels_dir)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]


# Define the CNN model
model = Sequential()
model.add(RandomBrightness(0.3))
model.add(RandomRotation(0.3))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(37, 50, 1)))  # Input shape for grayscale images
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))  # Adjust output layer for number of classes


#model = load_model('model.keras')
model.summary()
# Compile the model

optim = Adam(learning_rate=0.0025)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=16, shuffle=True)
model.evaluate(X_test, y_test)

image_path = 'th_wu_man.jpg'  # Replace with the path to the image you want to predict




predicted_class = predict_image(model, image_path)
print(predicted_class)

model.save("model.keras")
