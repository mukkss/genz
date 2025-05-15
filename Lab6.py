from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
reviews = [
    "The product quality is amazing! I absolutely love it.",
    "Terrible experience. The item arrived broken and customer service was unhelpful.",
    "It's okay, not great but not terrible either.",
    "Fast delivery and excellent packaging. Highly recommend!",
    "I’m very disappointed. The product did not match the description."
]
# Analyze sentiment and print results
for review, result in zip(reviews, sentiment_pipeline(reviews)):
    sentiment_type = "Positive" if result['label'].startswith('5') else "Negative" if result['label'].startswith('1') else "Neutral"
    print(f"Review: {review}\nSentiment: {result['label']}, Confidence: {result['score']:.2f} ({sentiment_type})\n")
