import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("../models/sentiment_model")
tokenizer = BertTokenizer.from_pretrained("../models/sentiment_model")

# Set the model to evaluation mode
model.eval()

# Define stronger positive and negative keywords
positive_keywords = ["amazing", "great", "fantastic", "excellent", "love", "wonderful", "best", "awesome", "outstanding", "recommend"]
negative_keywords = ["terrible", "awful", "worst", "bad", "dislike", "hate", "poor", "horrible", "wouldn't recommend", "not recommend", "disappointing"]

def rule_based_classification(text):
    text_lower = text.lower()
    if any(word in text_lower for word in negative_keywords):
        return "Negative"  # Prioritize negative if any negative keyword matches
    elif any(word in text_lower for word in positive_keywords):
        return "Positive"
    return None  # No keyword match

def predict_sentiment(text, threshold=0.6):
    # Apply rule-based classification first
    rule_based_sentiment = rule_based_classification(text)
    if rule_based_sentiment:
        return rule_based_sentiment
    
    # Tokenize and run the model if no rule applies
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        positive_prob = probabilities[0][1].item()

    # Threshold-based classification
    return "Positive" if positive_prob >= threshold else "Negative"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Predict sentiment of input texts.")
parser.add_argument("texts", type=str, nargs='+', help="Texts to analyze for sentiment.")
parser.add_argument("--output", type=str, default="predictions.txt", help="File to save the predictions.")
args = parser.parse_args()

# Run predictions and save results
with open(args.output, "w") as f:
    for text in args.texts:
        sentiment = predict_sentiment(text)
        f.write(f"Input: {text}\nPredicted Sentiment: {sentiment}\n\n")
        print(f"Input: {text}")
        print(f"Predicted Sentiment: {sentiment}")
        print("-" * 30)
    
print(f"Predictions saved to {args.output}")

