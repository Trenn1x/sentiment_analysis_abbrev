import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def load_data(path):
    # Load CSV data without specifying dtypes to prevent casting issues
    df = pd.read_csv(path, header=None, names=['stars', 'text', 'sentiment'], low_memory=False)
    df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    df = df.dropna(subset=['stars', 'sentiment'])
    df['sentiment'] = df['sentiment'].astype(int)
    return df['text'].tolist(), df['sentiment'].tolist()

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    train_texts, train_labels = load_data("../data/train_sample.csv")
    test_texts, test_labels = load_data("../data/test.csv")

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    # Set training parameters with limited epochs and disabled evaluation strategy
    training_args = TrainingArguments(
        output_dir="models",
        eval_strategy="no",  # Disable intermediate evaluation
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,  # Set to 1 epoch to reduce training time
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    # Start training
    trainer.train()

    # Save the trained model and tokenizer
    model.save_pretrained("../models/sentiment_model")
    tokenizer.save_pretrained("../models/sentiment_model")

