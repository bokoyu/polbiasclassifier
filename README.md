# Political Bias Classification Project

## Overview
This project is a political bias classification system that uses BERT to classify the bias of a given text as "Left", "Right", or "Neutral". The project consists of several Python scripts for training, evaluating, and predicting using the BERT-based classifier.

### Project Files
- **`data_loader.py`**: Handles loading and preprocessing the dataset, including label encoding and data splitting.
- **`train.py`**: Contains the training logic for fine-tuning a BERT model on the dataset to classify political bias.
- **`evaluate.py`**: Used to evaluate the model's performance on validation data.
- **`predict.py`**: Used to predict the political bias of new input text.
- **`main.py`**: The main entry point for the project, allowing users to train, evaluate, or predict using command line arguments.

## Requirements
The project requires the following Python packages, which can be installed using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- `joblib`

## Usage
The main script `main.py` can be used to train, evaluate, or predict the political bias of text. Below are the usage instructions:

### Training the Model
To train the model, provide the dataset path using the `--data` argument:

```bash
python main.py --mode train --data path/to/your/dataset.csv
```

### Evaluating the Model
To evaluate the model, use the `--data` argument to provide the dataset:

```bash
python main.py --mode evaluate --data path/to/your/dataset.csv
```

### Predicting Bias
To predict the political bias of new text, use the `--text` argument:

```bash
python main.py --mode predict --text "Your text here to predict bias"
```

## Dataset
The dataset should be a CSV file with the following columns:
- **`PHRASE`**: The text or phrase to classify.
- **`calculated_bias`**: The bias label, which can be "Left", "Right", or "Neutral".

The script will preprocess the dataset, encode the labels, and split the data into training and validation sets.

## Model Training
The training script fine-tunes a BERT model using PyTorch. The model can be trained on a GPU for better performance. After training, the model, tokenizer, and label encoder are saved in the `savedmodels/mediabias_bert_model` directory.

## Tips for Speeding Up Training
- **Use a GPU**: Training BERT can be slow on a CPU. Using a GPU will significantly speed up training.
- **Mixed Precision Training**: Consider using mixed precision to reduce memory usage and speed up training.
- **Freeze Lower Layers**: You can freeze some of the BERT layers to speed up training.

## Authors
This project was developed by Boris Petkov (bokoyu). Feel free to contribute or provide feedback.


