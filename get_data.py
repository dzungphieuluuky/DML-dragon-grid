import mss
import cv2
import numpy as np
import os
import time
import keyboard

# --- Configuration ---
# Define the keys your agent can press
# For Dragon Runner:
VALID_KEYS = ["left", "right", "up", "down"] # 'up' might be for shooting
# For Dragon Grid:
# VALID_KEYS = ["up", "down", "left", "right"]

# Folder to save the dataset
DATA_DIR = "game_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Region of the screen to capture (adjust these to your game window)
# You need to find these coordinates manually
monitor = {"top": 230, "left": 550, "width": 825, "height": 825}

# --- Data Collection Logic ---
print("Data Collection Script")
print("----------------------")
iter = int(input("Enter iteration number (0 for first run): "))
print("Press 's' to start collecting data.")
print("Press 'q' to quit.")
folder_path = os.path.join(DATA_DIR, f"game{iter}")
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# Wait for the start signal
keyboard.wait('s')
print("...Starting data collection in 3 seconds...")
time.sleep(3)

file_index = 0
collecting = True
print("Ready to collect data")
with mss.mss() as sct:
    while collecting:
        # 1. Capture the screen
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Convert to standard color format

        # 2. Check which key is being pressed
        action = None
        for key in VALID_KEYS:
            if keyboard.is_pressed(key):
                action = key
                break
        
        # 3. If a valid key was pressed, save the frame and action
        if action:
            time.sleep(0.1)
            # Save the image
            image_path = os.path.join(folder_path, f"frame_{file_index}_{action}.png")
            cv2.imwrite(image_path, frame)
            
            print(f"Saved: frame_{file_index} -> Action: {action}")
            file_index += 1
            
            # Small delay to avoid saving hundreds of frames for one key press
            time.sleep(0.1)

        # 4. Check for the quit signal
        if keyboard.is_pressed('q'):
            print("...Stopping data collection...")
            collecting = False

print(f"Data collection complete. {file_index} samples saved.")