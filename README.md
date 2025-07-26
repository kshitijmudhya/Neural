
# 🧠 Neural Network from Scratch - MNIST Classifier

This project demonstrates how to build, train, and test a fully connected feedforward neural network from scratch (no deep learning libraries) using Python and NumPy. The network is trained on the MNIST dataset of handwritten digits and then tested on example digit images.

---

## 📁 Project Structure

```
.
├── learn.py         # Trains a neural network from scratch on the MNIST dataset
├── testing.py       # Loads the saved model and predicts digit(s) from example image(s)
├── mymodel.pkl      # Trained neural network model saved via pickle
├── img_1.jpg        # Example test image of a handwritten digit
├── img_5.jpg        # Another test image
└── README.md        # This file
```

---

## 📚 Requirements

Make sure you have the following Python packages installed:

```bash
pip install numpy opencv-python tensorflow
```

---

## 🏋️‍♂️ Training the Model (`learn.py`)

To train the model:

```bash
python learn.py
```

- Loads the MNIST dataset using `tensorflow.keras.datasets`.
- Builds a feedforward network with:
  - Input layer (784 neurons)
  - Hidden layers (256 and 128 neurons with ReLU)
  - Output layer (10 neurons with sigmoid activation)
- Uses manual backpropagation and gradient descent.
- One-hot encodes labels.
- Trains for 10 epochs with a batch size of 32.
- Saves the trained model as `mymodel.pkl`.

---

## 🧪 Testing the Model (`testing.py`)

To test the trained model:

```bash
python testing.py
```

- Loads a grayscale test image (e.g., `img_5.jpg`).
- Loads the trained model from `mymodel.pkl`.
- Normalizes and reshapes the image for prediction.
- Prints the predicted digit to the console.

📸 **Note:** Input test images should be:
- 28x28 pixels
- Grayscale
- White digit on a black background

---

## 🔍 Example Output

When you run `testing.py`, you’ll get output like:

```
3
```

This indicates the model predicted the digit "3" in the input image.

---

## 📌 Notes

- All layers and training logic are implemented manually (no TensorFlow/PyTorch used for model logic).
- The model uses sigmoid activation in the output for simplicity.
- Saved model files are in Pickle format (`.pkl`).

---

## 🧑‍💻 Author

This project is a minimalist demonstration of core neural network principles — ideal for learning and experimentation.
