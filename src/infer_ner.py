import argparse
import json
from ner_model import NERModel

def main(args):
    # Load trained model
    ner_model = NERModel(data_path=args.data_path, model_save_path=args.model_path)
    ner_model.load_trained_model()

    if args.text:
        # Run inference on a single text input
        predictions = ner_model.predict_from_text(args.text)
        print("\n📌 Predicted Named Entities:")
        print(", ".join(predictions) if predictions else "No entities found.")

    elif args.dataset_filename:
        # Run inference on a dataset file (e.g., validation/test set)
        ner_model.predict(args.dataset_filename)
        print(f"✅ Predictions saved in {ner_model.predictions_path}/predictions.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained NER model.")
    
    parser.add_argument("--data_path", type=str, default="data", help="Path to the training dataset")
    parser.add_argument("--model_path", type=str, default="models/ner_model", help="Path to the trained model")
    parser.add_argument("--text", type=str, help="Input text for NER prediction (single sentence)")
    parser.add_argument("--dataset_filename", type=str, help="Filename of the dataset for batch prediction")

    args = parser.parse_args()
    main(args)
