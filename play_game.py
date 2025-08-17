import mss
import cv2
import numpy as np
import pyautogui
import time
import keyboard

from keras.models import load_model

# --- Configuration ---
MODEL_PATH = "game_agent_model.keras"
ACTIONS = ["left", "right", "up", "down"]
IMG_WIDTH, IMG_HEIGHT = 150, 112
monitor = {"top": 230, "left": 550, "width": 825, "height": 825}

if __name__ == "__main__":
    execute = input("Do you want to run the agent? (y/n): ").lower()
    if execute == 'y':
        execute = True
    else:
        execute = False

        # --- Load the Model ---
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}. Agent is ready.")

    print("Press 's' to start the agent.")
    print("Press 'q' to quit.")
    keyboard.wait('s')
    print("...Agent starting in 3 seconds...")
    time.sleep(3)
    start_time = time.time()
    with mss.mss() as sct:
        while True:
            if time.time() - start_time > 0.5:
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
                if execute:
                    pyautogui.press(predicted_action)
                start_time = time.time()
            
            # Check for quit signal
            if keyboard.is_pressed('q'):
                print("...Stopping agent...")
                break