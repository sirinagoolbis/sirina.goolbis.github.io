import os
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use CPU for compatibility (change to 'cuda' if GPU available)
device = torch.device('cpu')

# === Step 1: Load and preprocess dataset ===
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Combine relevant text columns to form the input
    df['text'] = df['Summary'] + " Skills: " + df['Skills']
    
    # Map each discipline (label) to a numeric id
    label_map = {label: idx for idx, label in enumerate(df['Discipline'].unique())}
    df['label'] = df['Discipline'].map(label_map)
    
    print("Class distribution:")
    print(df['Discipline'].value_counts())
    
    return df, label_map

# === Step 2: Prepare datasets and tokenize text ===
def prepare_datasets(df):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Tokenize texts (pad and truncate automatically)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    
    # Create Hugging Face Datasets
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

# === Step 3: Define evaluation metric ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# === Step 4: Train the BERT classification model ===
def train_bert_model(train_dataset, test_dataset, label_map):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_map)
    )
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
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
    
    # Save the best model and tokenizer
    trainer.save_model('./saved_model')
    trainer.tokenizer.save_pretrained('./saved_model')
    
    return model, trainer.tokenizer

# === Step 5: Predict on new texts ===
def predict(texts, model, tokenizer, label_map):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).tolist()
    
    # Map numeric predictions back to original labels
    idx_to_label = {v: k for k, v in label_map.items()}
    readable_preds = [idx_to_label[p] for p in preds]
    
    return readable_preds

# === Main execution ===
if __name__ == "__main__":
    # Load and preprocess your data CSV file here
    data_csv_path = "candidates_dataset.csv"
    
    df, label_map = load_and_preprocess(data_csv_path)
    train_dataset, test_dataset, tokenizer = prepare_datasets(df)
    model, tokenizer = train_bert_model(train_dataset, test_dataset, label_map)
    
    # Example: predict new samples
    sample_texts = [
        "Experienced software engineer skilled in Python, Java, and cloud computing.",
        "Marketing professional with expertise in social media and content creation."
    ]
    predictions = predict(sample_texts, model, tokenizer, label_map)
    print("Predictions:", predictions)
