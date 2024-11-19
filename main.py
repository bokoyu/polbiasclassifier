import argparse
from train import train_model
from evaluate import evaluate_model
from predict import predict_bias

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Political Bias Classification")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], required=True, help="Mode to run the script.")
    parser.add_argument('--data', type=str, help="Path to the dataset (only required for train and evaluate modes).")
    parser.add_argument('--text', type=str, help="Text to predict bias for (only in predict mode).")
    args = parser.parse_args()

    if args.mode == 'train':
        if args.data:
            train_model(args.data)
        else:
            print("Please provide the dataset path using --data argument.")
    elif args.mode == 'evaluate':
        if args.data:
            evaluate_model(args.data)
        else:
            print("Please provide the dataset path using --data argument.")
    elif args.mode == 'predict':
        if args.text:
            bias = predict_bias(args.text)
            print(f"Predicted Bias: {bias}")
        else:
            print("Please provide text input using --text argument.")