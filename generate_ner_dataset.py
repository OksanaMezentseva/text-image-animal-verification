import json
import random
import os

# Define the path to save datasets
BASE_PATH = "NER_ImageClassification/data"
os.makedirs(BASE_PATH, exist_ok=True)  # Ensure the directory exists

# Define animals for each dataset
train_animals = ["dog", "dolphin", "cat", "elephant", "spider", "lion", "horse", "chicken", "squirrel", "parrot", "cow"]
val_animals = ["penguin", "sheep", "dolphin", "owl", "butterfly"]
test_animals = ["bear", "fox", "zebra", "monkey", "eagle"]

# Sentence templates for each dataset
train_templates = [
    "I saw a {} in the park.",
    "Have you ever seen a {} before?",
    "The {} is my favorite animal.",
    "There is a {} in the picture.",
    "Look at that {}! It's so beautiful.",
    "A {} appeared in my backyard yesterday.",
    "They found a {} near the river.",
    "A {} was spotted in the zoo.",
    "One of the most fascinating creatures is the {}.",
    "In the wild, a {} can run incredibly fast.",
    "My neighbor owns a pet {}, and it's very friendly.",
    "The children were amazed when they saw a {} for the first time.",
    "A {} jumped out of the bush and ran across the road.",
    "During our trip to Australia, we saw a wild {} in its natural habitat.",
    "Scientists have been studying the behavior of {}s for decades.",
    "I met a friend who owns a pet {}."  # Non-obvious case where the animal is mentioned indirectly
]

val_templates = [
    "A {} was spotted near the river.",
    "Scientists study {}s to learn more about wildlife.",
    "In some cultures, {}s are considered sacred.",
    "The {} at the zoo was very active today.",
    "Can you identify the {} in this picture?",
    "A {} is known for its unique behavior.",
    "Some {}s are nocturnal and hunt at night.",
    "The {} was running across the road.",
    "A {} appeared in my backyard yesterday.",
    "The documentary featured a rare {} in its natural habitat."
]

test_templates = [
    "A {} was seen roaming in the mountains.",
    "Experts say {}s are highly intelligent creatures.",
    "A {} is often associated with strength and courage.",
    "The {} was resting under the shade of a large tree.",
    "Have you ever heard the sound of a {}?",
    "A {} can survive in extreme weather conditions.",
    "The {} is an important part of the ecosystem.",
    "A {} was found in an unexpected place today.",
    "There was a story about a lost {} on the news.",
    "The {} is a symbol of freedom and power."
]

# Function to generate dataset
def generate_dataset(file_name, num_samples, animals, templates):
    dataset = []
    for _ in range(num_samples):
        animal = random.choice(animals)
        sentence_template = random.choice(templates)
        sentence = sentence_template.format(animal)

        # Find entity position
        start_idx = sentence.index(animal)
        end_idx = start_idx + len(animal)

        # Add to dataset
        dataset.append({
            "text": sentence,
            "entities": [[start_idx, end_idx, "ANIMAL"]]
        })

     # Save as JSON in the target directory
    file_path = os.path.join(BASE_PATH, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ Dataset created: {file_path}")

# Generate train, validation, and test datasets
generate_dataset("train_ner_dataset.json", num_samples=500, animals=train_animals, templates=train_templates)
generate_dataset("val_ner_dataset.json", num_samples=100, animals=val_animals, templates=val_templates)
generate_dataset("test_ner_dataset.json", num_samples=100, animals=test_animals, templates=test_templates)