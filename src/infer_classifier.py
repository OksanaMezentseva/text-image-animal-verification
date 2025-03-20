import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from image_classifier_model import ImageClassifier

def main(args):
    # Load class mapping
    with open(args.class_map_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Load model
    classifier = ImageClassifier(data_dir='None', model_save_path=args.model_path, class_map_path = args.class_map_path)
    classifier.load_trained_model()

    if args.image_path:
        # Predict a single image
        predicted_class = classifier.predict_from_image(args.image_path, idx_to_class)
        print(f"\n📌 Predicted Class: {predicted_class}")

    elif args.dataset_filename:
        # Run prediction on a dataset
        classifier.predict(args.dataset_filename)
        print(f"✅ Predictions saved in {classifier.data_dir}/{args.dataset_filename.replace('.json', '_predictions.json')}")

    if args.evaluate:
        # Evaluate model performance
        classifier.evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained image classifier.")
    
    parser.add_argument("--class_map_path", type=str, default="models/classifier_model/classifier_model_class_map.json", help="Path to class mapping JSON file")
    parser.add_argument("--model_path", type=str, default="models/classifier_model/classifier_model.pth", help="Path to trained model")
    parser.add_argument("--image_path", type=str, help="Path to a single image for prediction")
    parser.add_argument("--dataset_filename", type=str, help="Filename of the dataset for batch prediction")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance on test set")

    args = parser.parse_args()
    main(args)
