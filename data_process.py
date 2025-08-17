from __init__ import *
import os
import cv2
import numpy as np
from keras.utils import to_categorical

def save_data():    # --- 1. Load and Preprocess Data ---

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
    # Save data for future use
    np.savez_compressed('game_data.npz', X=X, y=y)
    print(f"Loaded {len(X)} images with shape {X.shape} and labels with shape {y.shape}.")
    print("Data loading complete.")
    print("Data saved to 'game_data.npz'.")

def load_data():
    """
    Load the preprocessed data from the saved .npz file.
    """
    if not os.path.exists('game_data.npz'):
        print("Error: Data file 'game_data.npz' not found. Please run the data collection script first.")
        exit()
    
    data = np.load('game_data.npz')
    X = data['X']
    y = data['y']
    print(f"Loaded data with shapes: X={X.shape}, y={y.shape}")
    return X, y