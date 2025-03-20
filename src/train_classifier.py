import argparse
from image_classifier_model import ImageClassifier

def main(args):
    # Initialize classifier with parameters from command line
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Load the dataset and splits data into train, valid and test sets
    classifier.load_data()

    # Initializes the ResNet18 model
    classifier.build_model()

    # Train the model
    classifier.train()

    # Save the trained model
    classifier.save_model_and_class_map()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classifier.")

    parser.add_argument("--data_dir", type=str, default="data/processed-img", help="Path to processed images")
    parser.add_argument("--model_save_path", type=str, default="models/classifier_model.pth", help="Path to save the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)
