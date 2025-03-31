# README for Word Prediction Model

This project implements a basic word prediction model using TensorFlow and Keras. The model utilizes an LSTM (Long Short-Term Memory) neural network to predict the next word in a sequence of text, trained on a small text dataset. The code includes preprocessing steps such as tokenization, sequence generation, and padding, followed by the model training and word prediction functionality.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

Install the required packages using:

```bash
pip install tensorflow numpy
```

## Code Explanation

### 1. Import Libraries
- `numpy`: For numerical operations.
- `tensorflow`: For building and training the neural network.
- `Tokenizer`, `pad_sequences`, and other Keras utilities are used for text preprocessing and neural network layers.

### 2. Dataset
- A small dataset of text is defined with sentences like "hello world", "hello tensorflow", etc.
   
### 3. Tokenization
- The `Tokenizer` is used to convert the text dataset into a sequence of integers, where each integer represents a unique word in the dataset.
   
### 4. Sequence Preparation
- Input-output pairs are created by slicing each sequence, where the input is a portion of the sequence and the output is the next word in the sequence.

### 5. Padding
- The sequences are padded to ensure that all input sequences are of the same length.

### 6. Model Construction
- A simple neural network model is built using Keras. The model consists of:
  - An embedding layer that converts word indices into dense vectors.
  - An LSTM layer for sequence learning.
  - A Dense output layer with a softmax activation for predicting the next word in the sequence.

### 7. Model Training
- The model is compiled using `sparse_categorical_crossentropy` loss and the Adam optimizer, then trained on the input-output pairs for 50 epochs.

### 8. Prediction Function
- The function `predict_next_word` takes a sequence of text, converts it into indices, and uses the trained model to predict the next word in the sequence.

## Example Usage

```python
# Predict the next word for the input "hello"
next_word = predict_next_word(model, tokenizer, "hello")
print(f"Predicted next word: {next_word}")
```

The output will be the predicted next word for the given input text.

## How to Run

1. Ensure you have the required libraries installed.
2. Copy the code into a Python script or Jupyter Notebook.
3. Run the script to train the model and predict the next word for an input sequence.

## Notes

- The dataset used in this model is very small and simple. For a more robust model, a larger and more diverse dataset is recommended.
- The model can be further enhanced by adding more layers, changing the architecture, or using more advanced techniques like attention mechanisms.
  
## License

This project is open-source and available under the MIT License.

