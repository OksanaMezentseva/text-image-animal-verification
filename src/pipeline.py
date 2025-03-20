import torch
import json
import argparse
from image_classifier_model import ImageClassifier
from ner_model import NERModel

class MultimodalPipeline:
    """
    A pipeline that integrates a text-based Named Entity Recognition (NER) model
    with an image classification model to verify the correctness of user queries.
    """
    def __init__(self, ner_model_path, classifier_model_path, class_map_path):
        """
        Initializes the multimodal pipeline by loading pre-trained models.

        Parameters:
        - ner_model_path (str): Path to the trained NER model.
        - classifier_model_path (str): Path to the trained image classification model.
        - class_map_path (str): Path to the JSON file containing class-to-index mapping.
        """
        self.ner_model = NERModel(data_path=None, model_save_path=ner_model_path)
        self.ner_model.load_trained_model()
        
        self.classifier = ImageClassifier(data_dir=None, model_save_path=classifier_model_path)
        self.classifier.load_trained_model()
        
        # Load class mapping
        with open(class_map_path, "r") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def process(self, text, image_path):
        """
        Processes a user query by extracting animal entities from text and classifying the image.

        Parameters:
        - text (str): User input text containing an animal mention.
        - image_path (str): Path to the image file.

        Returns:
        - bool: True if the extracted animal entity matches the classified image, False otherwise.
        """
        animal_names = self.ner_model.predict_from_text(text)
        print(f"Extracted entities: {animal_names}")

        # If no animals found, return False
        if not animal_names:
            print("⚠️ No animal entities found in the text.")
            return False

        predicted_class = self.classifier.predict_from_image(image_path, self.idx_to_class)
        # Check for invalid image format
        if predicted_class == "Invalid Image Format":
            print("⚠️ Error: Invalid image format detected.")
            return False

        print(f"📌 Predicted class: {predicted_class}")

        # Compare extracted entity with predicted image class
        return any(predicted_class.lower() == animal.lower() for animal in animal_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multimodal pipeline for text and image verification.")
    parser.add_argument("--text", type=str, required=True, help="Input text containing an animal mention.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--ner_model", type=str, default="models/ner_model", help="Path to the trained NER model.")
    parser.add_argument("--classifier_model", type=str, default="models/classifier_model/classifier_model.pth", help="Path to the trained image classification model.")
    parser.add_argument("--class_map", type=str, default="models/classifier_model/classifier_model_class_map.json", help="Path to the class-to-index mapping JSON file.")

    args = parser.parse_args()
    
    pipeline = MultimodalPipeline(
        ner_model_path=args.ner_model,  
        classifier_model_path=args.classifier_model,
        class_map_path=args.class_map
    )
    
    result = pipeline.process(args.text, args.image)
    print(f"\n🔍 **Matching result:** {result}")