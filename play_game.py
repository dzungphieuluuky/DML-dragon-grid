import mss
import cv2
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
import time

# --- Configuration ---
MODEL_PATH = "game_agent_model.h5"
ACTIONS = ["left", "right", "up"]
IMG_WIDTH, IMG_HEIGHT = 150, 112
monitor = {"top": 40, "left": 0, "width": 800, "height": 600} # Same as collection

# --- Load the Model ---
model = load_model(MODEL_PATH)
print("Model loaded. Agent is ready.")

print("Press 's' to start the agent.")
print("Press 'q' to quit.")
keyboard.wait('s')
print("...Agent starting in 3 seconds...")
time.sleep(3)

with mss.mss() as sct:
    while True:
        # 1. Capture and preprocess the screen
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(frame_gray, (IMG_WIDTH, IMG_HEIGHT))
        img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

        # 2. Get the model's prediction
        predictions = model.predict(img)[0]
        
        # 3. Choose the best action
        action_index = np.argmax(predictions)
        predicted_action = ACTIONS[action_index]
        
        print(f"Prediction: {predicted_action} (Confidence: {predictions[action_index]:.2f})")

        # 4. Execute the action
        pyautogui.press(predicted_action)
        
        # Small delay to prevent crazy fast inputs
        time.sleep(0.1)

        # Check for quit signal
        if keyboard.is_pressed('q'):
            print("...Stopping agent...")
            break