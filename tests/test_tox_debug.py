from transformers import pipeline
from config import TOXICITY_MODEL_NAME

print(f"Using model: {TOXICITY_MODEL_NAME}")

clf = pipeline("text-classification", model=TOXICITY_MODEL_NAME)

texts = [
    "What was the company revenue in 2024?",
    "You are a complete idiot and I hate you",
    "I will find you and hurt you badly",
    "I hope you die",
]

for text in texts:
    raw = clf(text)
    print(f"RAW: {raw}")
    print(f"TEXT: {text[:60]}\n")