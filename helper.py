import mss
import cv2
import numpy as np
import tkinter as tk
import pytesseract
from PIL import Image

# Define the coordinates of the 3x3 grid on your screen
# (left, top, width, height)
def get_image(left, top, width, height, print=False):
    grid_coords = {'left': left, 'top': top, 'width': width, 'height': height}

    with mss.mss() as sct:
        # Grab the data
        sct_img = sct.grab(grid_coords)

        # Convert to an OpenCV-readable format (NumPy array)
        grid_image = np.array(sct_img)
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGRA2BGR) # Convert to BGR

    # For debugging, you can display the captured image
    if print:
        cv2.imshow('Captured Grid', grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return grid_image

def split_grid_into_tiles(grid_img, rows=3, cols=3, print=False):
    """Splits an image into a grid of smaller images."""
    height, width, _ = grid_img.shape
    tile_height = height // rows
    tile_width = width // cols
    
    tiles = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            # Define the bounding box for the tile
            start_x = c * tile_width
            start_y = r * tile_height
            end_x = (c + 1) * tile_width
            end_y = (r + 1) * tile_height
            
            # Slice the image to get the tile
            tile = grid_img[start_y:end_y, start_x:end_x]
            row_tiles.append(tile)
        tiles.append(row_tiles)
    if print:
        for i in range(rows):
            for j in range(cols):
                # Display each tile for debugging
                cv2.imshow(f'Tile {i},{j}', tiles[i][j])
                cv2.waitKey(0)
        cv2.destroyAllWindows()
    return tiles
# Example: Display the top-left tile for verification

import os

def load_templates(template_folder):
    """Loads all template images from a folder."""
    templates = {}
    for filename in os.listdir(template_folder):
        if filename.endswith(('.png', '.jpg')):
            # Use the filename (without extension) as the key
            name = os.path.splitext(filename)[0]
            template_path = os.path.join(template_folder, filename)
            # Load in grayscale for template matching
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[name] = template_img
    return templates

def identify_tile(tile_image, templates, threshold=0.8):
    """Identifies the content of a single tile using template matching."""
    tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
    
    best_match = None
    max_score = -1

    for name, template in templates.items():
        # Compare the tile with the current template
        result = cv2.matchTemplate(tile_gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        
        if score > max_score:
            max_score = score
            best_match = name
            
    if max_score >= threshold:
        return best_match
    else:
        return 'empty' # Or 'unknown' if no template matches well enough

def read_number_from_icon(image_path):
    """
    Reads the number from the top-right corner of the provided image icon.
    """
    # --- Step 1: Load and Isolate Region of Interest (ROI) ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Crop the image to the top-right corner where the number is.
    # These coordinates may need slight adjustments for different icon sizes.
    h, w, _ = img.shape
    roi = img[5:h//3, w//2:w-5]

    # --- Step 2: Convert to HSV for Color Segmentation ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # --- Step 3: Create a Mask for the White/Gray Number ---
    # Define a range for light gray to white colors in HSV space.
    # These values target the number's color specifically.
    lower_bound = np.array([0, 0, 150])
    upper_bound = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # --- Step 4: Clean the Mask ---
    # The red slash is connected to the number. We can use morphological
    # operations to try and separate them or clean up noise.
    # An "opening" operation (erosion followed by dilation) can remove small noise.
    kernel = np.ones((2, 2), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Step 5: Invert and Prepare for OCR ---
    # Tesseract generally works best with black text on a white background.
    # Our mask is white text on a black background, so we'll invert it.
    final_image = cv2.bitwise_not(cleaned_mask)

    # Optional: Display intermediate steps for debugging
    # cv2.imshow("Original ROI", roi)
    # cv2.imshow("Initial Mask", mask)
    # cv2.imshow("Cleaned Mask", cleaned_mask)
    # cv2.imshow("Final Image for OCR", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # --- Step 6: Perform OCR with Pytesseract ---
    try:
        # Use specific configuration for OCR'ing a single digit
        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
        
        # Convert OpenCV image (NumPy array) to a PIL Image
        pil_img = Image.fromarray(final_image)
        
        text = pytesseract.image_to_string(pil_img, config=custom_config)
        
        # Clean up the output
        number = text.strip()
        return number
        
    except pytesseract.TesseractNotFoundError:
        print("Tesseract Error: Tesseract is not installed or not in your PATH.")
        print("Please install Tesseract and/or configure the pytesseract.pytesseract.tesseract_cmd path.")
        return None
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return None