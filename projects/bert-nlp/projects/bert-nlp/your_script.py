import os
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# Suppress Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to CPU to avoid MPS errors on Mac
device = torch.device('cpu')

# Step 1: Load and preprocess data
def load_data():
    df = pd.read_csv("candidates_dataset.csv")
    df['text'] = df['Summary'] + " Skills: " + df['Skills']
    label_map = {label: idx for idx, label in enumerate(df['Discipline'].unique())}
    df['label'] = df['Discipline'].map(label_map)
    print(df['Discipline'].value_counts())
    return df, label_map

# Step 2: Prepare datasets and tokenize
def prepare_datasets(df):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=42, stratify=df['label']
    )
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels
    })
    return train_dataset, test_dataset, tokenizer

# Step 3: Define compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Step 4: Train model
def train_model(train_dataset, test_dataset, label_map, tokenizer):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    return model

# Step 5: Predict on new texts
def predict(texts, model, tokenizer, label_map):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1).tolist()
    
    readable_preds = [list(label_map.keys())[list(label_map.values()).index(pred)] for pred in preds]
    return readable_preds

# Optional: Predict from CSV
def predict_from_file(file_path, model, tokenizer, label_map):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    preds = predict(texts, model, tokenizer, label_map)
    df['predicted_label'] = preds
    df.to_csv('predictions_output.csv', index=False)
    print("Saved predictions to predictions_output.csv")

if __name__ == "__main__":
    df, label_map = load_data()
    train_dataset, test_dataset, tokenizer = prepare_datasets(df)
    model = train_model(train_dataset, test_dataset, label_map, tokenizer)
    
    # Example predictions
    test_samples = [
        "Experienced data scientist with Python and SQL skills",
        "Mechanical engineer with CAD and manufacturing background"
    ]
    predictions = predict(test_samples, model, tokenizer, label_map)
    print("Predicted Labels:", predictions)
