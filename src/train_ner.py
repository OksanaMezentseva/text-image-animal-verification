import argparse
from ner_model import NERModel

def main(args):
    # Initialize NER model with specified parameters
    ner_model = NERModel(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    if args.train:
        
        # Train the model
        ner_model.train()
        
        # Save the trained model
        ner_model.save_model()
        print("✅ Model training complete and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Named Entity Recognition (NER) model.")
    
    # Training arguments
    parser.add_argument("--data_path", type=str, default="data", help="Path to the training dataset")
    parser.add_argument("--model_save_path", type=str, default="models/ner_model", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")

    args = parser.parse_args()
    main(args)
