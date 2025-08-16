import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# --- Missing Keras/TensorFlow Imports ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
# ----------------------------------------

# --- 1. Load and Preprocess Data ---
DATA_DIR = "game_data"
IMG_WIDTH, IMG_HEIGHT = 150, 112 # Resize images to a smaller, consistent size
ACTIONS = ["left", "right", "up", "down"] # Must match VALID_KEYS from collection

images = []
labels = []

# Check if the data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found.")
    print("Please run the data collection script first.")
    exit()

print("Loading dataset...")
for folder_name in os.listdir(DATA_DIR):
    # Extract action from filename (e.g., 'frame_0_left.png')
    try:
        folder_dir = os.path.join(DATA_DIR, folder_name)
        for filename in os.listdir(folder_dir):
            if not filename.endswith('.png'):
                continue
            action = filename.split('_')[-1].replace('.png', '')
            if action in ACTIONS:
                # Load and resize image
                img_path = os.path.join(folder_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Grayscale is often enough
                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                
                # Convert action string to a number
                labels.append(ACTIONS.index(action))
    except IndexError:
        print(f"Warning: Skipping file with unexpected name format: {filename}")


if not images:
    print("Error: No valid images found in the data directory. Cannot train the model.")
    exit()

# Convert to NumPy arrays and normalize pixel values
# The shape for CNN input should be (num_samples, height, width, channels)
X = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0 
y = to_categorical(np.array(labels), num_classes=len(ACTIONS)) # One-hot encode labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Build the CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax') # Output layer: probabilities for each action
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\nTraining model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# --- 4. Save the Trained Model ---
# Using the recommended Keras format instead of the legacy .h5
model.save("game_agent_model.keras") 
print("\nModel saved as game_agent_model.keras")