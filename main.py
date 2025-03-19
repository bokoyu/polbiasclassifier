import argparse
from train import train_model, custom_clean
from evaluate import evaluate_model
from predict import predict #change this
from explain import kernel_shap_explain_bias, lime_explain_bias

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Political Bias Classification (2-Step)")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'shap', 'lime'], required=True,
                        help="Mode to run the script.")
    parser.add_argument('--data', type=str, help="Path to the dataset (for train/evaluate).")
    parser.add_argument('--text', type=str, help="Text to predict bias for (predict mode).")
    parser.add_argument('--cleaning', action='store_true', help="Whether to apply cleaning.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--overwrite', action='store_true', help="If set, overwrite existing model folder when training.")

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            print("Please provide the dataset path using --data argument.")
        else:
            train_model(
                data_path=args.data,
                do_cleaning=args.cleaning,
                cleaning_func=custom_clean,
                epochs=args.epochs,
                batch_size=args.batch_size,
                overwrite=args.overwrite
            )

    elif args.mode == 'evaluate':
        if not args.data:
            print("Please provide the dataset path using --data argument.")
        else:
            evaluate_model(
                data_path=args.data,
                do_cleaning=args.cleaning,
                cleaning_func=None,
                batch_size=args.batch_size
            )

    

    elif args.mode == 'predict':
        if not args.text:
            print("Please provide text input using --text argument.")
        else:
            final_prediction = predict(args.text)
            print(f"Final Prediction:\n{final_prediction}")
            

    elif args.mode == 'shap':
        if args.text:
            print(f"Running SHAP on user text: {args.text}")
            shap_values = kernel_shap_explain_bias([args.text])
        else:
            sample_texts = [
                "This is a suspiciously liberal statement!",
                "The left is misguided."
            ]
            print("No --text provided, using sample_texts.")
            shap_values = kernel_shap_explain_bias(sample_texts)

        print(shap_values)
        print("Generated SHAP values.")

    elif args.mode == 'lime':
        if args.text:
            print(f"Running LIME on user text: {args.text}")
            exp = lime_explain_bias(args.text)
        else:
            # Default sample text
            sample_text = "This is a suspiciously liberal statement!"
            print(f"No --text provided, using default sample: '{sample_text}'")
            exp = lime_explain_bias(sample_text)

        print("LIME Weights:", exp.as_list())
