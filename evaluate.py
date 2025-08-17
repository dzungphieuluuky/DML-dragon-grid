from keras.models import load_model
from train_model import load_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

if __name__ == "__main__":
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    X, y = load_data()

    # --- 2. Load the existing model ---
    MODEL_PATH = input("Enter the path to save the model (default: game_agent_model.keras): ") or "game_agent_model.keras"
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # --- 3. Evaluate the model ---
    print("Evaluating model...")
    evaluate_model(model, X, y)
    print("Evaluation complete.")
