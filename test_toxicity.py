from guardrails.toxicity import detect_toxicity

tests = [
    "What was the company revenue in 2024?",
    "You are a complete idiot and I hate this stupid system",
    "I will destroy you and everyone like you",
    "This is absolutely terrible and you should be ashamed",
]

for text in tests:
    result = detect_toxicity(text)
    status = "TOXIC  " if result["is_toxic"] else "clean  "
    print(f"[{status}] score={result['score']:.4f} | {text[:60]}")