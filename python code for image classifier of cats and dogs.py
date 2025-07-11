import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# --- 1. Data Preparation ---

# Define paths to your dataset
dataset_dir = 'data'  # Assuming 'data' is in the same directory as your script
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

# Image dimensions and batch size
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data augmentation and rescaling for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=40,       # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,   # Randomly shift width by up to 20%
    height_shift_range=0.2,  # Randomly shift height by up to 20%
    shear_range=0.2,         # Apply shearing transformations
    zoom_range=0.2,          # Apply random zoom
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill in new pixels created by transformations
)

# Only rescale validation data (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # 'binary' for 2 classes (cats/dogs)
)

# Flow validation images in batches from directory
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 2. Model Definition (Convolutional Neural Network - CNN) ---

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Fourth Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten the output for the Dense layers
    Flatten(),
    Dropout(0.5),  # Dropout for regularization to prevent overfitting

    # Dense layers
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])

# --- 3. Compile the Model ---

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', # Appropriate loss for binary classification
              metrics=['accuracy'])

# Display model summary
model.summary()

# --- 4. Train the Model ---

EPOCHS = 20 # You can adjust this based on your dataset size and performance

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Evaluate and Visualize Results ---

# Plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 6. Make Predictions (Example with a single image) ---

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = predictions[0]

    if score > 0.5:
        print(f"This image is a DOG with {score[0]*100:.2f}% confidence.")
    else:
        print(f"This image is a CAT with {(1-score[0])*100:.2f}% confidence.")

# Example usage: Replace 'path/to/your/test_image.jpg' with an actual image path
# Make sure this image is NOT from your training or validation set for a fair test.
# predict_image('path/to/your/test_image.jpg')