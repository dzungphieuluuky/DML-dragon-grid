# DML Dragon Grid

DML Dragon Grid is a Python automation project for the Dragon Grid mini-game event in Dragon Mania Legends. It enables you to collect gameplay data, preprocess it, train a neural network, and run an agent that plays the game automatically using screen capture and keyboard control.

## Overview

This project provides a full pipeline for automating the Dragon Grid mini-game:
- **Data Collection:** Record your own gameplay to create a dataset.
- **Data Processing:** Clean and format the data for machine learning.
- **Model Training:** Train a convolutional neural network (CNN) to predict game actions.
- **Agent Play:** Use the trained model to play the game in real time.
- **Evaluation:** Assess the agent's performance.

## Directory Structure

```
DML-dragon-grid/
├── __init__.py
├── main.py
├── get_data.py
├── data_process.py
├── train_model.py
├── play_game.py
├── evaluate.py
├── get_state.py
├── helper.py
├── game_data.npz
├── game_agent_model.keras
├── game_data/           # Raw gameplay data
├── templates/
├── LICENSE
├── README.md
└── ...
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Recommended: Use a virtual environment (e.g., `venv`, `uv`)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd DML-dragon-grid
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or use your preferred Python environment manager.

### Usage Workflow

1. **Collect gameplay data**
   ```sh
   python get_data.py
   ```
   Follow the prompts to record your actions while playing the game.

2. **Process and save data**
   ```sh
   python data_process.py
   ```
   This script preprocesses and saves your data as `game_data.npz`.

3. **Train the model**
   ```sh
   python train_model.py
   ```
   Trains a CNN on your processed data and saves the model as `game_agent_model.keras`.

4. **Run the agent**
   ```sh
   python play_game.py
   ```
   The agent will use screen capture and keyboard automation to play the game. Make sure the game window is visible and focused.

5. **Evaluate the agent**
   ```sh
   python evaluate.py
   ```
   Run evaluation scripts to measure the agent's performance.

## Notes

- Models are saved in Keras format (`.keras`).
- Processed datasets are saved as `.npz` files.
- The agent uses screen capture and keyboard automation; ensure the game window is not obstructed.
- For AMD GPU acceleration, see [TensorFlow-DirectML](https://github.com/microsoft/tensorflow-directml) or ROCm for Linux.

## License

This project is licensed under the MIT License.