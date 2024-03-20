import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load the dataset
data = pd.read_csv("poids.csv")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text descriptions
train_encodings = tokenizer(train_data["Description"].tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_data["Description"].tolist(), truncation=True, padding=True)

# Convert the weights to numeric values
train_labels = train_data["Poids"].tolist()
val_labels = val_data["Poids"].tolist()

# Convert the encodings and labels to datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# Fine-tune BERT on the dataset
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
training_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Evaluate the fine-tuned model
trainer.evaluate()

