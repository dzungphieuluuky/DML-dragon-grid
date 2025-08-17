import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model

from data_process import load_data
from __init__ import IMG_WIDTH, IMG_HEIGHT, ACTIONS
# ----------------------------------------

X, y = load_data()
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    retrain = input("Do you want to retrain the model? (y/n): ").lower()
    if retrain == 'y':
        retrain = True
    else:
        retrain = False
        
    if not retrain:
        MODEL_PATH = input("Enter the path to the existing model (default: game_agent_model.keras): ") or "game_agent_model.keras"
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
            print("Training the model...")
            model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test))

            # --- 4. Save the Model ---
            MODEL_PATH = input("Enter the path to save the model (default: game_agent_model.keras): ") or "game_agent_model.keras"
            model.save(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

            # Optionally evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        else:
            print(f"Model file '{MODEL_PATH}' not found. Please run the training script first.")
            exit()
    else:
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
        print("Training the model...")
        model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test))

        # --- 4. Save the Model ---
        MODEL_PATH = input("Enter the path to save the model (default: game_agent_model.keras): ") or "game_agent_model.keras"
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # Optionally evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")