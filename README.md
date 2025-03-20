# Text-Image Animal Verification

## Overview
This project implements a **multimodal machine learning pipeline** that integrates **Natural Language Processing (NLP) and Computer Vision (CV)** to verify whether a user's textual statement about an image is correct.

### **How It Works**
1. **Named Entity Recognition (NER) Model** extracts animal names from the input text.
2. **Image Classification Model** determines which animal is present in the image.
3. The system **compares the extracted entity with the classified object** and returns `True` if they match, otherwise `False`.

### **Motivation**
The idea for this project was inspired by the need for automated verification of text-image pairs. Ensuring that text descriptions match images is crucial in various applications, such as:
- 🛒 **E-commerce Moderation** – Preventing incorrect product listings.
- 📚 **Educational Apps** – Helping students verify their understanding of animals.
- 🏞 **Wildlife Research** – Assisting in the identification of species from camera-trap images.
- 📢 **Content Verification for Social Media** – Detecting mismatched captions and preventing misinformation.

---

## **Project Structure**
```
text-image-animal-verification/
│── data/                     # Data directory (not included in GitHub, available for download)
│   ├── raw-img/              # Raw images
│   ├── test_images/          # Test images
│   ├── train_ner_dataset.json # Training dataset for NER model
│   ├── val_ner_dataset.json   # Validation dataset for NER model
│   ├── test_ner_dataset.json  # Test dataset for NER model
│
│── models/                    # Pretrained models (available for download)
│   ├── classifier_model/       # Image classifier model
│   ├── ner_model/              # NER model
│
│── src/                        # Source code
│   ├── image_classifier_model.py # Image classification model
│   ├── infer_classifier.py     # Inference script for image classification
│   ├── infer_ner.py            # Inference script for NER model
│   ├── ner_model.py            # NER model implementation
│   ├── pipeline.py             # Multimodal pipeline script
│   ├── train_classifier.py     # Training script for the image classifier
│   ├── train_ner.py            # Training script for the NER model
│
│── notebooks/                  # Jupyter Notebooks
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   ├── demo.ipynb              # Demonstration of the pipeline
│
│── README.md                    # Project documentation
│── requirements.txt              # Python dependencies
│── generate_ner_dataset.py       # Script for generating the NER dataset
```

---

## **Installation**

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Download Models and Data
Since the `models/` and `data/` directories are too large for GitHub, you can download them from the provided links:
- **[Download Models](https://drive.google.com/file/d/1FUwW9qnwdjlq23rY6LW_5K7Xe9U3xdG3/view?usp=sharing)**
- **[Download Data](https://drive.google.com/file/d/1-mBFHxxfikUYI6LupYPPvNsLgO0HoaWX/view?usp=sharing)**

Extract them into the project root:
```sh
cd text-image-animal-verification
```

---

## **Training the Models**
### Train the NER Model
```sh
python3 src/train_ner.py --data_path data/train_ner_dataset.json --model_save_path models/ner_model --batch_size 16 --epochs 3 --learning_rate 5e-5
```

### Train the Image Classification Model
```sh
python3 src/train_classifier.py --data_dir data/processed-img --model_save_path models/classifier_model/classifier_model.pth --batch_size 32 --epochs 20 --learning_rate 0.001
```

---

## **Running the Pipeline**
To test the pipeline with a text input and an image, run:
```sh
python3 src/pipeline.py --text "There is a cat in the picture." --image data/test_images/cat.jpg
```

---

## **Pipeline Workflow**
1. The user provides a **text input** and an **image**.
2. The **NER model** extracts potential animal names from the text.
3. The **Image classifier model** predicts the class of the animal in the image.
4. The pipeline checks if the extracted animal name matches the predicted class.
5. The system outputs `True` (if the match is correct) or `False` (if it is incorrect).

### **Example**
#### **Input:**
```sh
python3 src/pipeline.py --text "I see a cat." --image data/test_images/cow.jpg
```
#### **Output:**
```
Extracted entities: ['cat']
📌 Predicted class: cow
🔍 Matching result: False
```

---

## **Technologies Used**
- **Python 3.8+**
- **Hugging Face Transformers** (NER model)
- **PyTorch / TensorFlow** (Image classification)
- **spaCy / NLTK** (Text preprocessing)
- **OpenCV** (Image processing)
- **Pandas, NumPy, Scikit-learn** (Data handling)
- **Matplotlib, Seaborn** (Exploratory Data Analysis)

---

## **Future Improvements**
- 🔹 **Improve NER accuracy** by fine-tuning with domain-specific training data.
- 🔹 **Enhance image classification** by incorporating more diverse datasets.
- 🔹 **Introduce out-of-distribution (OOD) detection** to filter unrelated images.
- 🔹 **Deploy the model** as an API for real-world applications.

---

📌 **If you find this project interesting, feel free to star ⭐ the repository and contribute!** 🚀