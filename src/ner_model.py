import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import os
from sklearn.metrics import classification_report

class NERDataset(Dataset):
    """ Custom Dataset class for Named Entity Recognition (NER) """

    def __init__(self, data_path, tokenizer, max_length=128):
        """
        Initializes the dataset by loading and processing JSON data.

        Parameters:
        - data_path (str): Path to the dataset file.
        - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text.
        - max_length (int): Maximum sequence length for tokenized inputs.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}  # Label mapping for entities

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entities = item["entities"]

        # Tokenize with offsets
        tokens = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_offsets_mapping=True
        )

        labels = [0] * len(tokens["input_ids"])  # Initialize all tokens as "O" (no entity)
        token_word_map = {i: False for i in range(len(tokens["input_ids"]))}  # Track first token of each word

        # Assign labels to tokens
        for start, end, entity in entities:
            entity_tokens = []
            for i, (token_start, token_end) in enumerate(tokens["offset_mapping"]):
                if token_start >= start and token_end <= end:
                    entity_tokens.append(i)

            if entity_tokens:
                labels[entity_tokens[0]] = self.label_map["B-ANIMAL"]  # First token gets B-ANIMAL
                for idx in entity_tokens[1:]:  # All following tokens get I-ANIMAL
                    labels[idx] = self.label_map["I-ANIMAL"]
                    token_word_map[idx] = True  # Mark them as part of entity

        for i in range(1, len(labels)):
            token_str = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"][i])
            if token_str.startswith("##") and labels[i] == 0:  # Ensure subword inherits label
                labels[i] = labels[i - 1]

        tokens["labels"] = labels
        tokens.pop("offset_mapping") 

        return {key: torch.tensor(val) for key, val in tokens.items()}

class NERModel:
    """
    Named Entity Recognition (NER) Model using BERT.
    """

    def __init__(self, data_path, model_save_path, batch_size=16, epochs=3, learning_rate=5e-5, predictions_path=None):
        """
        Initializes the NERModel.

        Parameters:
        - data_path (str): Path to the dataset.
        - model_save_path (str): Path to save trained model.
        - batch_size (int): Batch size for training.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for the optimizer.
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.predictions_path = predictions_path if predictions_path else os.path.join(model_save_path, "predictions")  
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and dataset
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

        # Load data only if data_path is provided
        if self.data_path:
            self.train_loader, self.val_loader, self.test_loader, self.label_map = self.load_data()
        else:
            self.train_loader = self.val_loader = self.test_loader = None
            self.label_map = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
        
        # Load model
        self.model = self.build_model()

    def load_data(self):
        """
        Loads the dataset from separate train/validation/test JSON files.
    
        Returns:
        - train_dataset (Dataset): Training dataset.
        - val_dataset (Dataset): Validation dataset.
        - test_dataset (Dataset): Test dataset.
        - label_map (dict): Mapping of labels to integer values.
        """
        train_path = os.path.join(self.data_path, "train_ner_dataset.json")
        val_path = os.path.join(self.data_path, "val_ner_dataset.json")
        test_path = os.path.join(self.data_path, "test_ner_dataset.json")
    
        # Load datasets separately
        train_dataset = NERDataset(train_path, self.tokenizer)
        val_dataset = NERDataset(val_path, self.tokenizer)
        test_dataset = NERDataset(test_path, self.tokenizer)
    
        # Use label map from any dataset (they are identical)
        label_map = train_dataset.label_map
    
        return train_dataset, val_dataset, test_dataset, label_map

    def build_model(self):
        """
        Initializes a BERT model for token classification.

        Returns:
        - model (BertForTokenClassification): BERT model for NER.
        """
        model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(self.label_map))
        return model.to(self.device)

    def train(self):
        """
        Trains the NER model using Hugging Face Trainer API.
        """
        # Load datasets
        train_dataset, val_dataset, _, _ = self.load_data()  # We don't need test dataset for training
    
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            report_to="none"
        )
    
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  
            tokenizer=self.tokenizer
        )
    
        print("Starting training...")
        trainer.train()
    
        # Print loss history after training
        for log in trainer.state.log_history:
            if "loss" in log:
                print(f"Epoch {log.get('epoch', 'N/A')} - Training Loss: {log['loss']:.4f}")
            if "eval_loss" in log:
                print(f"Epoch {log.get('epoch', 'N/A')} - Validation Loss: {log['eval_loss']:.4f}")
    
        print("✅ Training complete!")
        

    def save_model(self):
        """
        Saves the trained model and label mapping.
        Ensures the save directory exists before saving.
        """
        # Ensure the directory exists
        os.makedirs(self.model_save_path, exist_ok=True)  # ✅ Create directory if it does not exist
    
        # Save model and tokenizer
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
    
        # Save label mapping
        label_map_path = os.path.join(self.model_save_path, "label_map.json")
        with open(label_map_path, "w") as f:
            json.dump(self.label_map, f)
    
        print(f"✅ Model saved at {self.model_save_path}")


    def load_trained_model(self):
        """
        Loads a trained NER model.
        """
        self.model = BertForTokenClassification.from_pretrained(self.model_save_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_save_path)
        with open(f"{self.model_save_path}/label_map.json", "r") as f:
            self.label_map = json.load(f)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")

    def predict(self, dataset_filename):
        """
        Generates predictions for the given dataset file and saves them to a JSON file.
    
        Parameters:
        - dataset_filename (str): Name of the dataset file to predict (e.g., "val_ner_dataset.json" or "test_ner_dataset.json").
    
        Returns:
        - None (predictions are saved to a file).
        """
        dataset_path = os.path.join(self.data_path, dataset_filename)  # Construct full dataset path
    
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset file not found: {dataset_path}")
            return
    
        print(f"🔍 Loading dataset for prediction: {dataset_filename}")
    
        # Load dataset
        dataset = NERDataset(dataset_path, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
        self.model.eval()
        id_to_label = {v: k for k, v in self.label_map.items()}  # Reverse mapping
    
        predictions = []
    
        with torch.no_grad():
            for batch in data_loader:
                batch = {key: val.to(self.device) for key, val in batch.items()}
                outputs = self.model(**batch).logits
                batch_predictions = torch.argmax(outputs, dim=2).cpu().numpy()
                batch_true_labels = batch["labels"].cpu().numpy()
    
                for i, (sentence_preds, sentence_true) in enumerate(zip(batch_predictions, batch_true_labels)):
                    tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][i].cpu().tolist())
    
                    sentence_result = {
                        "tokens": [],
                        "pred_labels": [],
                        "true_labels": []
                    }
    
                    word = ""
                    word_pred_label = None
                    word_true_label = None
    
                    for j, (token, pred_label, true_label) in enumerate(zip(tokens, sentence_preds, sentence_true)):
                        if token in ["[CLS]", "[SEP]", "[PAD]"]:
                            continue  # Ignore special tokens
                        
                        if token.startswith("##"):  
                            word += token[2:]  # Append to previous token (subword)
                        else:
                            if word:  # If a word was built, save its label
                                sentence_result["tokens"].append(word)
                                sentence_result["pred_labels"].append(id_to_label[word_pred_label])
                                sentence_result["true_labels"].append(id_to_label[word_true_label])
    
                            word = token  # Start a new word
                            word_pred_label = pred_label
                            word_true_label = true_label
    
                    # Append the last word
                    if word:
                        sentence_result["tokens"].append(word)
                        sentence_result["pred_labels"].append(id_to_label[word_pred_label])
                        sentence_result["true_labels"].append(id_to_label[word_true_label])
    
                    predictions.append(sentence_result)
        
        # Ensure predictions directory exists
        os.makedirs(self.predictions_path, exist_ok=True)  # ✅ Create directory if it does not exist
    
        # Save predictions in JSON format
        predictions_file = os.path.join(self.predictions_path, "predictions.json")
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=4)
    
        print(f"📁 Predictions saved at {predictions_file}")

    def predict_from_text(self, text):
        """
        Predicts named entities from a given text input.

        Parameters:
        - text (str): Input sentence.

        Returns:
        - List of recognized animals.
        """
        self.model.eval()
        id_to_label = {v: k for k, v in self.label_map.items()}  # Reverse mapping

        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokens).logits

        predictions = torch.argmax(outputs, dim=2).cpu().numpy()[0]
        
        token_ids = tokens["input_ids"].squeeze(0).tolist()  # Перетворюємо в список
        words = self.tokenizer.convert_ids_to_tokens(token_ids)

        recognized_animals = []
        current_animal = ""

        for token, label_id in zip(words, predictions):
            label = id_to_label[label_id]

            if label == "B-ANIMAL":
                if current_animal:
                    recognized_animals.append(current_animal)  # Додаємо попереднє слово
                current_animal = token  # Починаємо нове слово
            elif label == "I-ANIMAL" and current_animal:
                current_animal += token.replace("##", "")  # Видаляємо субтокени
            else:
                if current_animal:
                    recognized_animals.append(current_animal)
                    current_animal = ""

        if current_animal:
            recognized_animals.append(current_animal)

        return recognized_animals


    def evaluate_predictions(self):
        """
        Evaluates the predictions using Precision, Recall, and F1-score.

        Returns:
        - None (prints evaluation metrics).
        """
        # Load predictions
        predictions_file = os.path.join(self.predictions_path, "predictions.json")
        if not os.path.exists(predictions_file):
            print(f"❌ Predictions file not found at {predictions_file}. Run `predict()` first.")
            return

        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        true_labels = []
        pred_labels = []

        # Extract labels
        for sentence in predictions:
            true_labels.append(sentence["true_labels"])
            pred_labels.append(sentence["pred_labels"])

        # Flatten lists
        true_labels_flat = [label for sentence in true_labels for label in sentence]
        pred_labels_flat = [label for sentence in pred_labels for label in sentence]

        # Compute classification report
        report = classification_report(true_labels_flat, pred_labels_flat, digits=4)

        print("\n📊 Evaluation Results:")
        print(report)
